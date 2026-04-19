"""Export Chatterbox's conditional decoder (speech tokens → waveform).

This module combines three stages into a single ONNX graph
``conditional_decoder.onnx``:

1. **Encoder** — upsample speech tokens to mel-spectrogram conditioning
2. **CFM (Conditional Flow Matching)** — N Euler steps that denoise a Gaussian
   into a clean mel-spectrogram. The step count is fixed at export time.
3. **HiFi-GAN** — convert the mel-spectrogram to 24 kHz waveform. Uses a local
   ``ISTFT`` implementation to avoid the non-traceable torch.istft.

The original monolithic export lived in ``chatterbox_to_onnx_conversion_script.py``;
this split separates it from the speech encoder and embed tokens exports without
changing behavior.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s3tokenizer.model import Conv1d, LayerNorm, Linear, MultiHeadAttention
from s3tokenizer.utils import mask_to_bias

from ._constants import (
    S3GEN_SR, S3_SR, S3_HOP, S3_TOKEN_HOP, S3_TOKEN_RATE, SPEECH_VOCAB_SIZE,
    START_SPEECH_TOKEN, STOP_SPEECH_TOKEN, EXAGGERATION_TOKEN,
    CFM_PARAMS, ISTFT_PARAMS,
)




class ISTFT(torch.nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        assert n_fft >= win_length
        super().__init__()

        self.filter_length = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = self.filter_length // 2 + 1
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])


        self.window = torch.hann_window(win_length)

        # Center pad the window to the size of n_fft
        pad_length = n_fft - self.window.size(0)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left

        torch_fft_window = F.pad(self.window, (pad_left, pad_right), mode='constant', value=0)
        inverse_basis *= torch_fft_window

        self.register_buffer('inverse_basis', inverse_basis.float())

    @staticmethod
    def window_sumsquare(
        window,
        n_frames,
        hop_length,
        win_length,
        n_fft,
    ):
        if win_length is None:
            win_length = n_fft

        n = n_fft + hop_length * (n_frames - 1)

        # Compute the squared window at the desired length
        win_sq = window ** 2

        # Center pad the window to the size of n_fft
        pad_length = n_fft - win_sq.size(0)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        win_sq = F.pad(win_sq, (pad_left, pad_right), mode='constant', value=0)

        # Prepare the kernel for conv_transpose1d
        win_sq = win_sq.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_fft)

        # Create the input signal: ones of shape (1, 1, n_frames)
        s = torch.ones(1, 1, n_frames, dtype=window.dtype, device=window.device)

        # Perform conv_transpose1d with stride=hop_length
        x = F.conv_transpose1d(s, win_sq, stride=hop_length).squeeze()

        # Adjust x to have length n
        x = x[:n]

        return x

    def forward(self, recombine_magnitude_phase):
        assert recombine_magnitude_phase.dim() == 3, 'must be [B, 2 * N, T]'
        num_frames = recombine_magnitude_phase.size(-1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )

        window_sum = self.window_sumsquare(
            self.window,
            n_frames=num_frames,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.filter_length,
        )

        tiny_value = torch.finfo(window_sum.dtype).tiny

        denom = torch.where(
            window_sum > tiny_value,
            window_sum,
            torch.tensor(1.0, dtype=window_sum.dtype, device=window_sum.device),
        )
        # Apply the transformation
        inverse_transform /= denom

        # scale by hop ratio
        inverse_transform *= self.filter_length / self.hop_length

        q = self.filter_length // 2
        inverse_transform = inverse_transform[:, 0, q:-q]
        return inverse_transform

istft = ISTFT(ISTFT_PARAMS["n_fft"], ISTFT_PARAMS["hop_len"], ISTFT_PARAMS["n_fft"])


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max()
    seq_range = torch.arange(0,
                            max_len,
                            dtype=torch.int64,
                            device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -1.0e+10
    return mask


class ConditionalDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.output_size = model.s3gen.flow.output_size
        self.input_embedding = model.s3gen.flow.input_embedding
        self.spk_embed_affine_layer = model.s3gen.flow.spk_embed_affine_layer
        self.encoder = model.s3gen.flow.encoder
        self.encoder_proj = model.s3gen.flow.encoder_proj
        self.time_embeddings = model.s3gen.flow.decoder.estimator.time_embeddings
        self.time_mlp = model.s3gen.flow.decoder.estimator.time_mlp
        self.up_blocks = model.s3gen.flow.decoder.estimator.up_blocks
        self.static_chunk_size = model.s3gen.flow.decoder.estimator.static_chunk_size
        self.mid_blocks = model.s3gen.flow.decoder.estimator.mid_blocks
        self.down_blocks = model.s3gen.flow.decoder.estimator.down_blocks
        self.final_block = model.s3gen.flow.decoder.estimator.final_block
        self.final_proj = model.s3gen.flow.decoder.estimator.final_proj
        self.n_fft = ISTFT_PARAMS["n_fft"]
        self.hop_len = ISTFT_PARAMS["hop_len"]
        self.n_trim = S3GEN_SR // 50
        self.stft_window = model.s3gen.mel2wav.stft_window
        self.f0_predictor = model.s3gen.mel2wav.f0_predictor
        self.f0_upsamp = model.s3gen.mel2wav.f0_upsamp
        self.m_source = model.s3gen.mel2wav.m_source
        self.inference_cfg_rate = 0.7
        self.conv_pre = model.s3gen.mel2wav.conv_pre
        self.lrelu_slope = model.s3gen.mel2wav.lrelu_slope
        self.reflection_pad = model.s3gen.mel2wav.reflection_pad
        self.ups = model.s3gen.mel2wav.ups
        self.source_downs = model.s3gen.mel2wav.source_downs
        self.source_resblocks = model.s3gen.mel2wav.source_resblocks
        self.resblocks = model.s3gen.mel2wav.resblocks
        self.conv_post = model.s3gen.mel2wav.conv_post
        self.istft = istft
    
    def cond_forward(self, x, mask, mu, t, spks, cond) -> torch.Tensor:
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.
        """

        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        x = torch.cat([x, mu], dim=1)
        spks = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.cat([x, spks], dim=1)
        x = torch.cat([x, cond], dim=1)

        masks = [mask]
        resnet, transformer_blocks, downsample = self.down_blocks[0]
        mask_down = masks[-1]
        x = resnet(x, mask_down, t)
        x = x.permute(0, 2, 1).contiguous()
        attn_mask = mask_to_bias(mask_down.bool() == 1, x.dtype)
        for transformer_block in transformer_blocks:
            x = transformer_block(
                hidden_states=x,
                attention_mask=attn_mask,
                timestep=t,
            )
        x = x.permute(0, 2, 1).contiguous()
        residual = x  # Save hidden states for skip connections
        x = downsample(x * mask_down)
        masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = x.permute(0, 2, 1).contiguous()
            attn_mask = mask_to_bias(mask_mid.bool() == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = x.permute(0, 2, 1).contiguous() 

        resnet, transformer_blocks, upsample = self.up_blocks[0]
        mask_up = masks.pop()
        x = torch.cat([x[:, :, :residual.shape[-1]], residual], dim=1)
        x = resnet(x, mask_up, t)
        x = x.permute(0, 2, 1).contiguous()
        attn_mask = mask_to_bias(mask_up.bool() == 1, x.dtype)
        for transformer_block in transformer_blocks:
            x = transformer_block(
                hidden_states=x,
                attention_mask=attn_mask,
                timestep=t,
            )
        x = x.permute(0, 2, 1).contiguous()
        x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output

    def flow_forward(self, speech_tokens, token_len, mask, embedding, prompt_feat):
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        speech_tokens = self.input_embedding(torch.clamp(speech_tokens, min=0))
        speech_tokens = speech_tokens * mask

        # text encode
        text_encoded, _ = self.encoder(speech_tokens, token_len)
        mel_len1, mel_len2 = prompt_feat.shape[1], text_encoded.shape[1] - prompt_feat.shape[1]
        text_encoded = self.encoder_proj(text_encoded)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size]).to(text_encoded.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mu = text_encoded
        spks = embedding
        if not isinstance(mel_len1, torch.Tensor):
            mel_len1 = torch.tensor(mel_len1, device=speech_tokens.device)
        if not isinstance(mel_len2, torch.Tensor):
            mel_len2 = torch.tensor(mel_len2, device=speech_tokens.device)
        return mel_len1, mel_len2, mu, spks, conds

    def decode(self, x: torch.Tensor, s_stft: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)

        # ---- Upsample 0 ----
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[0](x)

        si = self.source_downs[0](s_stft)
        si = self.source_resblocks[0](si)
        x = x + si

        xs0 = self.resblocks[0](x) + self.resblocks[1](x) + self.resblocks[2](x)
        x = xs0 / 3

        # ---- Upsample 1 ----
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[1](x)

        si = self.source_downs[1](s_stft)
        si = self.source_resblocks[1](si)
        x = x + si

        xs1 = self.resblocks[3](x) + self.resblocks[4](x) + self.resblocks[5](x)
        x = xs1 / 3

        # ---- Upsample 2 ----
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[2](x)
        x = self.reflection_pad(x)

        si = self.source_downs[2](s_stft)
        si = self.source_resblocks[2](si)
        x = x + si

        xs2 = self.resblocks[6](x) + self.resblocks[7](x) + self.resblocks[8](x)
        x = xs2 / 3

        # ---- Final layers ----
        x = F.leaky_relu(x)
        x = self.conv_post(x)

        magnitude = torch.exp(x[:, :self.n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.n_fft // 2 + 1:, :])

        return magnitude, phase

    def forward(self, speech_tokens, speaker_embeddings, speaker_features):
        token_len = torch.full((speech_tokens.size(0),), speech_tokens.size(1), dtype=torch.long, device=speech_tokens.device)
        mask = (~make_pad_mask(token_len)).unsqueeze(-1)
        mel_len1, mel_len2, mu, spks, cond = self.flow_forward(speech_tokens, token_len, mask, speaker_embeddings, speaker_features)
        mu = mu.transpose(1, 2).contiguous()
        total_len = mel_len1.add(mel_len2).unsqueeze(0)
        mask = (~make_pad_mask(total_len)).unsqueeze(0)
        n_timesteps = 10
        temperature = 1.0
        x = torch.randn_like(mu, dtype=mu.dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps+1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        dt_all = t_span[1:] - t_span[:-1]

        t = t_span[0:1]
        dt = dt_all[0:1]

        x_in = torch.cat([x, torch.zeros_like(x)], dim=0) 
        mask_in = torch.cat([mask, torch.zeros_like(mask)], dim=0) 
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0) 
        t_in = torch.cat([t, torch.zeros_like(t)], dim=0) 
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0) 
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        ## Classifier-Free Guidance inference introduced in VoiceBox
        # step 1
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[1 + 1] - t

        # step 2
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[2 + 1] - t

        # step 3
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[3 + 1] - t

        # step 4
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[4 + 1] - t

        # step 5
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[5 + 1] - t

        # step 6
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[6 + 1] - t

        # step 7
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[7 + 1] - t

        # step 8
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[8 + 1] - t

        # step 9
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        t = t + dt
        dt = t_span[9 + 1] - t

        # step 10
        x_in[:].copy_(x.squeeze(0)) 
        mask_in[:].copy_(mask.squeeze(0))
        mu_in[0].copy_(mu.squeeze(0))
        t_in[:].copy_(t.squeeze(0))
        spks_in[0].copy_(spks.squeeze(0))
        cond_in[0].copy_(cond.squeeze(0))
        dphi_dt = self.cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
        x = x + dt * dphi_dt
        output = x.float()
        speech_feat = torch.narrow(output, dim=2, start=mel_len1, length=output.size(2) - mel_len1)
        #mel->f0
        f0 = self.f0_predictor(speech_feat)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        output_sources = s.transpose(1, 2).squeeze(1)
        spec = torch.stft(
            output_sources,
            self.n_fft, 
            self.hop_len, 
            self.n_fft, 
            window=self.stft_window.to(output_sources.device),
            return_complex=False)
        s_stft_real, s_stft_imag = spec[..., 0], spec[..., 1]
        output_sources = torch.cat([s_stft_real, s_stft_imag], dim=1)
        magnitude, phase = self.decode(x=speech_feat, s_stft=output_sources)
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        recombine_magnitude_phase = torch.cat([real, img], dim=1)
        output_wavs = self.istft(recombine_magnitude_phase)
        trim_fade = torch.zeros(2 * self.n_trim)
        cosine_window = (torch.cos(torch.linspace(torch.pi, 0, self.n_trim)) + 1) / 2
        trim_fade[self.n_trim:] = cosine_window
        output_wavs[:, :trim_fade.size(0)] *= trim_fade
        return output_wavs


