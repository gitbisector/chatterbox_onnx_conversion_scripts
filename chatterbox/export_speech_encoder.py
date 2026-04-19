"""Export Chatterbox's speech encoder (reference-audio → speaker embedding + conditioning).

This module exports the ``PrepareConditionalsModel`` wrapper around Chatterbox's
speaker encoder, S3 tokenizer, and mel-spectrogram extractor to a single ONNX
graph ``speech_encoder.onnx`` that produces:

- ``audio_features``  — speaker conditioning embedding for the T3 prefill
- ``audio_tokens``    — speech tokens from the reference audio (prepended to generated tokens before the vocoder)
- ``speaker_embeddings`` — 192-D CAMPPlus speaker vector for the vocoder
- ``speaker_features`` — reference mel-spectrogram for the vocoder

The code below is an extract of the original monolithic
``chatterbox_to_onnx_conversion_script.py``; this split separates the speech
encoder export from the embed-tokens and conditional-decoder exports without
changing behavior.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio as ta
from torchaudio.compliance.kaldi import get_mel_banks

from s3tokenizer.model import Conv1d, LayerNorm, Linear, MultiHeadAttention
from s3tokenizer.utils import mask_to_bias

import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
import math

from ._constants import (
    S3GEN_SR, S3_SR, S3_HOP, S3_TOKEN_HOP, S3_TOKEN_RATE, SPEECH_VOCAB_SIZE,
    MILLISECONDS_TO_SECONDS, START_SPEECH_TOKEN, STOP_SPEECH_TOKEN,
    EXAGGERATION_TOKEN, ENC_COND_LEN, DEC_COND_LEN,
)


@dataclass
class ModelConfig:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 3**8

    use_sdpa: bool = False

def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    1 for non-padded part and 0 for padded part.

    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B,).

    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, max_T).

    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    batch_size = lengths.size(0)
    # max_len = max_len if max_len > 0 else lengths.max()
    max_len_2 = lengths.max()
    seq_range = torch.arange(0,
                             max_len_2,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len_2)
    seq_length_expand = lengths.unsqueeze(-1)
    return seq_range_expand < seq_length_expand


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0,
                         scaling=None):
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    if scaling is not None:
        t = t * scaling
    freqs = torch.outer(t, freqs).float()  # type: ignore
    
    cos = freqs.cos()
    sin = freqs.sin()
    
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)
    return cos, sin


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    D = xq.shape[-1]
    half_l, half_r = xq[:, :, :, :D // 2], xq[:, :, :, D // 2:]
    xq_r = torch.cat((-half_r, half_l), dim=-1)

    D = xk.shape[-1]

    half_l, half_r = xk[:, :, :, :D // 2], xk[:, :, :, D // 2:]
    xk_r = torch.cat((-half_r, half_l), dim=-1)

    return xq * cos + xq_r * sin, xk * cos + xk_r * sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)


class FSQCodebook(nn.Module):

    def __init__(self, dim: int, level: int = 3):
        super().__init__()
        self.project_down = nn.Linear(dim, 8)
        self.level = level
        self.embed = None

    @torch.inference_mode()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x = rearrange(x, "... d -> (...) d")
        x = x.reshape(-1, x.shape[-1])
        return x

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        h = self.project_down(x).float()
        h = h.tanh()
        h = h * 0.9990000128746033
        h = h.round() + 1
        # h = ((self.level - 1) * h).round()  # range [-k, k]
        powers = torch.pow(
            self.level,
            torch.arange(2**self.level, device=x.device, dtype=h.dtype))
        mu = torch.sum(h * powers.unsqueeze(0), dim=-1)
        ind = mu.reshape(x_shape[0], x_shape[1])
        return ind

    @torch.inference_mode()
    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            'There is no official up project component provided')


class FSQVectorQuantization(nn.Module):
    """Vector quantization implementation (inference-only).
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
    ):
        super().__init__()
        assert 3**8 == codebook_size
        self._codebook = FSQCodebook(dim=dim, level=3)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._codebook.encode(x)

    @torch.inference_mode()
    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        quantize = self._codebook.decode(embed_ind)
        # quantize = rearrange(quantize, "b n d -> b d n")
        quantize = quantize.permute(0, 2, 1)
        return quantize


class FSMNMultiHeadAttention(MultiHeadAttention):

    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
        use_sdpa: bool = False,
    ):
        super().__init__(n_state, n_head)

        self.fsmn_block = nn.Conv1d(n_state,
                                          n_state,
                                          kernel_size,
                                          stride=1,
                                          padding=0,
                                          groups=n_state,
                                          bias=False)
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding
        self.pad_fn = nn.ConstantPad1d(
            (self.left_padding, self.right_padding), 0.0)

        self.use_sdpa = use_sdpa

    def forward_fsmn(self,
                     inputs: torch.Tensor,
                     mask: Optional[torch.Tensor] = None):
        b, t, _, _ = inputs.size()
        inputs = inputs.view(b, t, -1)
        if mask is not None and mask.size(2) > 0:  # time2 > 0
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        return x * mask

    def qkv_attention(self,
                      q: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor,
                      mask: Optional[torch.Tensor] = None,
                      mask_pad: Optional[torch.Tensor] = None,
                      cos: Optional[torch.Tensor] = None,
                      sin: Optional[torch.Tensor] = None):
        _, _, D = q.shape
        scale = (D // self.n_head)**-0.25
        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)

        if cos is not None and sin is not None:
            q, k = apply_rotary_emb(q, k, cos=cos, sin=sin)

        fsm_memory = self.forward_fsmn(v, mask_pad)

        q = q.permute(0, 2, 1, 3) * scale
        v = v.permute(0, 2, 1, 3)

        if not self.use_sdpa:
            k = k.permute(0, 2, 3, 1) * scale
            qk = q @ k  # (B, n_head, T, T)
            if mask is not None:
                qk = qk + mask
            qk = qk.float()
            w = F.softmax(qk, dim=-1).to(q.dtype)
            return (w @ v).permute(
                0, 2, 1, 3).flatten(start_dim=2), qk.detach(), fsm_memory
        else:
            k = k.permute(0, 2, 1, 3) * scale
            assert mask is not None
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.,
                scale=1.,
            )
            output = (output.transpose(1,
                                       2).contiguous().view(q.size(0), -1, D)
                      )  # (batch, time1, d_model)
            return output, None, fsm_memory

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                mask_pad: Optional[torch.Tensor] = None,
                cos: Optional[torch.Tensor] = None,
                sin: Optional[torch.Tensor] = None):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk, fsm_memory = self.qkv_attention(q, k, v, mask, mask_pad,
                                                cos, sin)
        return self.out(wv) + fsm_memory, qk


class ResidualAttentionBlock(nn.Module):

    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
        use_sdpa: bool = False,
    ):
        super().__init__()

        self.attn = FSMNMultiHeadAttention(n_state,
                                           n_head,
                                           kernel_size,
                                           use_sdpa=use_sdpa)
        self.attn_ln = LayerNorm(n_state, eps=1e-6)

        n_mlp = n_state * 4

        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(),
                                       Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(
            self.attn_ln(x), mask=mask, mask_pad=mask_pad,
            cos=cos, sin=sin)[0]

        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoderV2(nn.Module):

    def __init__(
        self,
        n_mels: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
        use_sdpa: bool,
    ):
        super().__init__()
        self.stride = stride

        self.conv1 = Conv1d(n_mels,
                            n_state,
                            kernel_size=3,
                            stride=stride,
                            padding=1)
        self.conv2 = Conv1d(n_state,
                            n_state,
                            kernel_size=3,
                            stride=2,
                            padding=1)
        cos, sin = precompute_freqs_cis(64, 1024 * 2)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, use_sdpa=use_sdpa)
            for _ in range(n_layer)
        ])

    def forward(self, x: torch.Tensor,
                x_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, T)
            the mel spectrogram of the audio
        x_len: torch.Tensor, shape = (batch_size,)
            length of each audio in x
        """
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = F.gelu(self.conv1(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = F.gelu(self.conv2(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
        # NOTE: .contiguous() is essential for dynamo export!
        x = x.contiguous()

        
        cos = self.cos[:x.size(1)].to(x.device)
        sin = self.sin[:x.size(1)].to(x.device)

        mask_pad = mask.transpose(1, 2)
        mask = mask_to_bias(mask, x.dtype).unsqueeze(1)

        for block in self.blocks:
            x = block(x, mask, mask_pad, cos, sin)

        return x, x_len


class S3TokenizerV2(nn.Module):
    """S3 tokenizer v2 implementation (inference-only).
    Args:
        config (ModelConfig): Config
    """

    def __init__(self):
        super().__init__()
        # self.name = name  # Store model name for token_rate determination
        self.config = ModelConfig()
        self.encoder = AudioEncoderV2(
            self.config.n_mels,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            2,
            self.config.use_sdpa,
        )
        self.quantizer = FSQVectorQuantization(
            self.config.n_audio_state,
            self.config.n_codebook_size,
        )

    def forward(self, mel: torch.Tensor,
                mel_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.quantize(mel, mel_len)

    @torch.inference_mode()
    def quantize(self, mel: torch.Tensor,
                 mel_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize mel spectrogram to tokens, with automatic long audio handling.

        Args:
            mel: mel spectrogram tensor, shape (batch_size, n_mels, T)
            mel_len: mel length tensor, shape (batch_size,)

        Returns:
            code: quantized tokens, shape (batch_size, T')
            code_len: token length, shape (batch_size,)
        """
        # Check if any audio in the batch exceeds 30 seconds
        # Assuming 16kHz sample rate and hop_length=160, 30s = 30*16000/160 = 3000 frames
        # max_frames = 3000

        # Check which samples are long audio
        # assert (mel_len <= max_frames).all()

        # All short audio - use original method
        hidden, code_len = self.encoder(mel, mel_len)
        code = self.quantizer.encode(hidden).long()
        return code, code_len

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False


class S3Tokenizer(S3TokenizerV2):
    """
    s3tokenizer.S3TokenizerV2 with the following changes:
    - a more integrated `forward`
    - compute `log_mel_spectrogram` using `_mel_filters` and `window` in `register_buffers`
    """

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        config: ModelConfig = ModelConfig()
    ):
        super().__init__()

        self.n_fft = 400
        _mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=config.n_mels
        )
        self.register_buffer(
            "_mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor,
        max_len,
    ) -> torch.Tensor:
        """
        NOTE: mel-spec has a hop size of 160 points (100 frame/sec).

        Args
        ----
        - `wavs`: 16 kHz speech audio
        """
        mels = self.log_mel_spectrogram(wavs)  # [B, F, T]
        if max_len is not None:
            mels = mels[..., :max_len * 4]
        mel_lens = torch.full((mels.shape[0],), mels.shape[-1], dtype=torch.int32, device=self.device)

        speech_tokens, _ = self.quantize(mels, mel_lens)
        return speech_tokens

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        padding: int = 0,
    ):
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: torch.Tensor, shape = (*)
            The path to audio or either a NumPy array or Tensor containing the
            audio waveform in 16 kHz

        padding: int
            Number of zero samples to pad to the right

        Returns
        -------
        torch.Tensor, shape = (128, n_frames)
            A Tensor that contains the Mel spectrogram
        """

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        stft = torch.stft(
            audio, self.n_fft, S3_HOP,
            window=self.window.to(self.device),
            return_complex=False
        )
        # remove Nyquist bin
        stft = stft[..., :-1, :]
        # compute magnitude squared
        magnitudes = stft[..., 0]**2 + stft[..., 1]**2

        mel_spec = self._mel_filters.to(self.device) @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


class SafeDenseLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(SafeDenseLayer, self).__init__()
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = torch.nn.Sequential()
        self.nonlinear.add_module("layernorm", torch.nn.LayerNorm(out_channels))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.linear(x)
        if x.size(-1) == 1:
            x = x[:, :, 0]
        x = self.nonlinear(x)
        return x


class PrepareConditionalsModel(torch.nn.Module):

    speech_cond_prompt_len = 150
    speaker_embed_size = 256

    def __init__(self, chatterbox):
        super().__init__()

        # TODO: Move loading elsewhere
        self.s3 = S3Tokenizer()
        self.s3.load_state_dict(chatterbox.s3gen.tokenizer.state_dict(), strict=False)

        self.speaker_encoder = chatterbox.s3gen.speaker_encoder
        self.flow = chatterbox.s3gen.flow

        self.cond_enc = chatterbox.t3.cond_enc

        self.resampler = ta.transforms.Resample(S3GEN_SR, S3_SR)
        self.eps = torch.tensor(torch.finfo(torch.float).eps)
        self.n_fft = 400
        _mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=128
        )
        self.register_buffer(
            "mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

        self.speech_emb = chatterbox.t3.speech_emb
        self.speech_pos_emb = chatterbox.t3.speech_pos_emb

        # Speaker embedding projection
        # NOTE: From testing, randomly/zero initializing speaker embedding seems to work fine
        # speaker_emb = torch.randn(batch_size, self.speaker_embed_size)
        speaker_emb = torch.zeros(1, self.speaker_embed_size)
        self.cond_spkr = self.cond_enc.spkr_enc(speaker_emb.view(-1, self.speaker_embed_size))[:, None]  # (B, 1, dim)

    def mel_spectrogram(self, y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920,
                    fmin=0, fmax=8000, center=False):
        y = F.pad(
            y.unsqueeze(1),
            ((n_fft - hop_size) // 2, (n_fft - hop_size) // 2),
            mode="reflect",
        )
        y = y.squeeze(1)
        hann_window = torch.hann_window(win_size)
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel = torch.from_numpy(mel).float()
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )
        # real = spec[..., 0]
        # imag = spec[..., 1]
        # spec = torch.sqrt(real**2 + imag**2 + 1e-9)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(mel, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5) * 1) # spectral_normalize_torch

        return spec
    
    def _next_power_of_2(self, x: int) -> int:
        r"""Returns the smallest power of 2 that is greater than x"""
        return 1 if x == 0 else 2 ** (x - 1).bit_length()
    
    def _get_strided(self, waveform: torch.Tensor, window_size: int, window_shift: int) -> torch.Tensor:
        r"""Given a waveform (1D tensor of size ``num_samples``), it returns a 2D tensor (m, ``window_size``)
        representing how the window is shifted along the waveform. Each row is a frame.

        Args:
            waveform (Tensor): Tensor of size ``num_samples``
            window_size (int): Frame length
            window_shift (int): Frame shift
            snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
                in the file, and the number of frames depends on the frame_length.  If False, the number of frames
                depends only on the frame_shift, and we reflect the data at the ends.

        Returns:
            Tensor: 2D tensor of size (m, ``window_size``) where each row is a frame
        """
        num_samples = waveform.size(0)
        strides = (window_shift * waveform.stride(0), waveform.stride(0))

        if num_samples < window_size:
            return torch.empty((0, 0), dtype=waveform.dtype, device=waveform.device)
        else:
            m = 1 + (num_samples - window_size) // window_shift

        sizes = (m, window_size)
        return waveform.as_strided(sizes, strides)

    def _get_log_energy(self, strided_input: torch.Tensor, epsilon: torch.Tensor, energy_floor: float) -> torch.Tensor:
        r"""Returns the log energy of size (m) for a strided_input (m,*)"""
        device, dtype = strided_input.device, strided_input.dtype
        log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()  # size (m)
        if energy_floor == 0.0:
            return log_energy
        return torch.max(log_energy, torch.tensor(math.log(energy_floor), device=device, dtype=dtype))

    def _get_window(
        self,
        waveform: torch.Tensor,
        padded_window_size: int,
        window_size: int,
        window_shift: int,
        preemphasis_coefficient: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Gets a window and its log energy

        Returns:
            (Tensor, Tensor): strided_input of size (m, ``padded_window_size``) and signal_log_energy of size (m)
        """
        device, dtype = waveform.device, waveform.dtype
        # size (m, window_size)
        strided_input = self._get_strided(waveform, window_size, window_shift)

        # Subtract each row/frame by its mean
        row_means = torch.mean(strided_input, dim=1).unsqueeze(1)  # size (m, 1)
        strided_input = strided_input - row_means

        if preemphasis_coefficient != 0.0:
            # strided_input[i,j] -= preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
            offset_strided_input = F.pad(strided_input.unsqueeze(0), (1, 0), mode="replicate").squeeze(
                0
            )  # size (m, window_size + 1)
            strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]

        # Apply window_function to each row/frame
        window_function = torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85).unsqueeze(0)  # size (1, window_size)
        strided_input = strided_input * window_function  # size (m, window_size)

        # Pad columns with zero until we reach size (m, padded_window_size)
        if padded_window_size != window_size:
            padding_right = padded_window_size - window_size
            strided_input = F.pad(
                strided_input.unsqueeze(0), (0, padding_right), mode="constant", value=0
            ).squeeze(0)

        return strided_input

    def _get_waveform_and_window_properties(
        self,
        waveform: torch.Tensor,
        channel: int,
        sample_frequency: float,
        frame_shift: float,
        frame_length: float,
    ) -> Tuple[torch.Tensor, int, int, int]:
        r"""Gets the waveform and window properties"""
        channel = max(channel, 0)
        waveform = waveform[channel, :]  # size (n)
        window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
        window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
        padded_window_size = self._next_power_of_2(window_size)
        return waveform, window_shift, window_size, padded_window_size

    def extract_feature(self, waveform: torch.Tensor,
        channel: int = -1,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        high_freq: float = 0.0,
        low_freq: float = 20.0,
        num_mel_bins: int = 23,
        preemphasis_coefficient: float = 0.97,
        sample_frequency: float = 16000.0,
        vtln_high: float = -500.0,
        vtln_low: float = 100.0,
        vtln_warp: float = 1.0):

        device, dtype = waveform.device, waveform.dtype
        waveform, window_shift, window_size, padded_window_size = self._get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length)

        # strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
        strided_input = self._get_window(
            waveform,
            padded_window_size,
            window_size,
            window_shift,
            preemphasis_coefficient,
        )

        # size (m, padded_window_size // 2 + 1)
        spec = torch.stft(
            strided_input,
            n_fft=512,
            hop_length=512,
            center=False,
            window=None,
            return_complex=False
        )   # shape: [..., freq, 2]  (last dim = [real, imag])

        # Compute magnitude manually
        real = spec[..., 0]
        imag = spec[..., 1]
        spectrum = torch.sqrt(real**2 + imag**2).squeeze(-1)
        spectrum = spectrum.pow(2.0)

        # size (num_mel_bins, padded_window_size // 2)
        mel_energies, _ = get_mel_banks(
            num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq, vtln_low, vtln_high, vtln_warp
        )
        mel_energies = mel_energies.to(device=device, dtype=dtype)

        # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
        mel_energies = F.pad(mel_energies, (0, 1), mode="constant", value=0)

        # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
        mel_energies = torch.matmul(spectrum, mel_energies.T)

        # avoid log of zero (which should be prevented anyway by dithering)
        mel_energies = torch.max(mel_energies, self.eps).log()
        return mel_energies

    def prepare_conditions_from_audio(self, audio_values):
        batch_size = audio_values.shape[0]

        # Compute embed_ref
        ref_wav_24 = audio_values[..., :DEC_COND_LEN]
        speaker_features = self.mel_spectrogram(ref_wav_24).transpose(1, 2)

        # Resample to 16kHz
        ref_wav_16 = self.resampler(audio_values) # resample uncropped audio

        # Speech cond prompt tokens
        # TODO START REMOVE
        # -- AT EXPORT, WE MUST SWAP THIS WITH self.resampler(audio_values)
        # ref_wav_16 = librosa.resample(audio_values.cpu().numpy(), orig_sr=S3GEN_SR, target_sr=S3_SR)
        # ref_wav_16 = torch.from_numpy(ref_wav_16).to(audio_values.device)
        # TODO END REMOVE

        feature = self.extract_feature(ref_wav_16, num_mel_bins=80) # == Kaldi.fbank(ref_wav_16, num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        speaker_embeddings = self.speaker_encoder(feature.unsqueeze(0))

        t3_cond_prompt_tokens = self.s3(ref_wav_16[..., :ENC_COND_LEN], max_len=self.speech_cond_prompt_len)

        resampled_wav_16 = self.resampler(ref_wav_24) # resample uncropped audio

        # NOTE: For some reason, we do two passes of the s3 tokenizer
        # TODO: Try reduce this?
        # Tokenize 16khz reference
        prompt_token = self.s3(resampled_wav_16, max_len=None)

        cond_prompt_speech_emb = self.speech_emb(t3_cond_prompt_tokens) + \
                     self.speech_pos_emb(t3_cond_prompt_tokens)

        # Cond prompt
        cond_prompt_speech_emb = self.cond_enc.perceiver(cond_prompt_speech_emb)

        expanded_cond_spkr = self.cond_spkr.expand(batch_size, -1, -1)  # (B, 1, dim)

        # Concat and return
        cond_emb = torch.cat((
            expanded_cond_spkr,
            cond_prompt_speech_emb,
        ), dim=1)  # (B, len_cond, dim)
        # assert cond_emb.dim() == 3
        return cond_emb, prompt_token, speaker_embeddings, speaker_features

    def forward(
        self,
        audio_values: torch.Tensor, # NOTE: Must have sample rate of S3GEN_SR=24000
    ):
        cond_emb, prompt_token, speaker_embeddings, speaker_features = self.prepare_conditions_from_audio(audio_values)
        return cond_emb, prompt_token, speaker_embeddings, speaker_features

