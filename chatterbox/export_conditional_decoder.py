"""Export Chatterbox's conditional decoder (speech tokens → waveform).

This module wraps Chatterbox's S3Gen vocoder (flow encoder + CFM UNet + HiFi-GAN)
for end-to-end ONNX export and differs from the original monolithic export in
three ways:

1. **Parameterized CFM step count.**  The original export baked exactly
   10 Euler steps for the diffusion-like flow-matching loop into the graph.
   This implementation accepts ``n_cfm_timesteps`` at construction so the
   caller can export variants for different speed/quality trade-offs
   (e.g. N=4 for real-time, N=10 for reference quality).

2. **Scatter-free graph.**  The upstream ``solve_euler`` used ``x_in[:].copy_(x)``
   inside a runtime Python loop, producing ``ScatterND`` nodes that prevent
   ORT CUDA-graph capture.  Here the loop is Python-unrolled at trace time
   (N steps baked in) and the batched CFG tensors are rebuilt every step
   via ``torch.cat([..., zeros_like(...)], dim=0)`` — scatter-free.

3. **Scatter-free SineGen.**  The HiFi-GAN's original ``SineGen`` built the
   harmonic frequency matrix via a per-row in-place loop
   (``F_mat[:, i:i+1, :] = f0 * (i+1) / sr``), yielding ``harmonic_num+1``
   ScatterND nodes on its own.  Here we replace it with a single broadcast
   multiply over a ``(1, H, 1)`` harmonic-index tensor.

4. **Custom torch-exportable ISTFT / STFT.**  ``torch.istft`` does not export
   cleanly; we replace it with ``CustomISTFT`` (conv_transpose1d-based).
   ``torch.stft(..., return_complex=True) + view_as_real`` is replaced with
   ``torch.stft(..., return_complex=False)``.

Interface: constructor ``ConditionalDecoder(chatterbox_model, n_cfm_timesteps=10)``;
forward takes ``(speech_tokens, speaker_embeddings, speaker_features)`` and
returns ``waveform``.  Backward-compatible with the original 1-arg
``ConditionalDecoder(chatterbox_model)`` call used by the driver.
"""
import math
from typing import Tuple

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


# Derived constants — pulled from CFM_PARAMS / ISTFT_PARAMS so we read the
# authoritative values and can't drift from them silently.
INFERENCE_CFG_RATE = float(CFM_PARAMS["inference_cfg_rate"])  # 0.7
ISTFT_N_FFT = int(ISTFT_PARAMS["n_fft"])                      # 16
ISTFT_HOP_LEN = int(ISTFT_PARAMS["hop_len"])                  # 4
N_TRIM = S3GEN_SR // 50                                       # 480 samples = 20ms fade-in


class CustomISTFT(nn.Module):
    """
    ConvTranspose1d-based inverse STFT. Exports cleanly to ONNX (unlike the
    native torch.istft, which emits a custom op). Adapted from the reference
    conversion script at:
      /tmp/onnx_conversion_scripts/chatterbox/chatterbox_to_onnx_conversion_script.py
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        assert n_fft >= win_length
        self.filter_length = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = self.filter_length // 2 + 1
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        window = torch.hann_window(win_length)
        pad_length = n_fft - window.size(0)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        torch_fft_window = F.pad(window, (pad_left, pad_right), mode="constant", value=0)
        inverse_basis *= torch_fft_window

        self.register_buffer("inverse_basis", inverse_basis.float(), persistent=False)
        self.register_buffer("window", window, persistent=False)

    @staticmethod
    def _window_sumsquare(window, n_frames, hop_length, win_length, n_fft):
        win_sq = window**2
        pad_length = n_fft - win_sq.size(0)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        win_sq = F.pad(win_sq, (pad_left, pad_right), mode="constant", value=0)
        win_sq = win_sq.unsqueeze(0).unsqueeze(0)

        s = torch.ones(1, 1, n_frames, dtype=window.dtype, device=window.device)
        x = F.conv_transpose1d(s, win_sq, stride=hop_length).squeeze()
        n = n_fft + hop_length * (n_frames - 1)
        return x[:n]

    def forward(self, recombine_magnitude_phase: torch.Tensor) -> torch.Tensor:
        assert recombine_magnitude_phase.dim() == 3, "must be [B, 2*N, T]"
        num_frames = recombine_magnitude_phase.size(-1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )

        window_sum = self._window_sumsquare(
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
        inverse_transform = inverse_transform / denom
        inverse_transform = inverse_transform * (self.filter_length / self.hop_length)

        q = self.filter_length // 2
        return inverse_transform[:, 0, q:-q]



class ConditionalDecoder(nn.Module):
    """
    Wraps the S3Gen flow encoder + CFM UNet estimator + HiFi-GAN vocoder for
    end-to-end ONNX export.

    Key differences from chatterbox.models.s3gen's runtime path:
      * CFM loop is Python-unrolled N times at trace time (`n_timesteps`
        fixed at construction); this replaces the runtime `solve_euler` loop
        which uses in-place `.copy_()` and produces ScatterND.
      * The CFG-batched tensors `x_in, mask_in, mu_in, ...` are rebuilt each
        step via `torch.cat(...,[zeros_like(...)], dim=0)`. This is cheap
        (all tensors are cond + zeros) and scatter-free.
      * `torch.stft(..., return_complex=True) + view_as_real` is replaced
        with `torch.stft(..., return_complex=False)`.
      * `torch.istft` is replaced with `CustomISTFT` (conv_transpose1d).
      * The per-CFG-step estimator forward skips `add_optional_chunk_mask`
        (static_chunk_size == 0, so it's equivalent to just the mask).
    """

    def __init__(self, chatterbox_or_s3gen, n_cfm_timesteps: int = 10):
        """Accept either a full chatterbox TTS model (``.s3gen`` will be extracted)
        or a bare ``s3gen`` module, for backward compatibility with the driver's
        ``ConditionalDecoder(chatterbox_model)`` invocation."""
        super().__init__()
        # Figure out whether we were handed chatterbox or its s3gen sub-module.
        s3gen = getattr(chatterbox_or_s3gen, "s3gen", chatterbox_or_s3gen)

        self.n_timesteps = int(n_cfm_timesteps)
        self.inference_cfg_rate = INFERENCE_CFG_RATE
        self.n_trim = N_TRIM
        self.n_fft = ISTFT_N_FFT
        self.hop_len = ISTFT_HOP_LEN

        # ---- Flow encoder (speech_tokens -> mu) ----
        flow = s3gen.flow
        self.output_size = flow.output_size          # == 80
        self.input_embedding = flow.input_embedding
        self.spk_embed_affine_layer = flow.spk_embed_affine_layer
        self.encoder = flow.encoder
        self.encoder_proj = flow.encoder_proj
        self.pre_lookahead_len = flow.pre_lookahead_len
        self.token_mel_ratio = flow.token_mel_ratio

        # ---- CFM UNet estimator ----
        est = flow.decoder.estimator
        self.time_embeddings = est.time_embeddings
        self.time_mlp = est.time_mlp
        self.down_blocks = est.down_blocks            # 1 block for channels=[256]
        self.mid_blocks = est.mid_blocks              # 12 blocks
        self.up_blocks = est.up_blocks                # 1 block
        self.final_block = est.final_block
        self.final_proj = est.final_proj
        # meanflow / static chunk are both unused here (static_chunk_size==0)

        # ---- HiFi-GAN mel2wav ----
        mel2wav = s3gen.mel2wav
        self.conv_pre = mel2wav.conv_pre
        self.lrelu_slope = mel2wav.lrelu_slope
        self.reflection_pad = mel2wav.reflection_pad
        self.ups = mel2wav.ups
        self.num_upsamples = mel2wav.num_upsamples
        self.num_kernels = mel2wav.num_kernels
        self.source_downs = mel2wav.source_downs
        self.source_resblocks = mel2wav.source_resblocks
        self.resblocks = mel2wav.resblocks
        self.conv_post = mel2wav.conv_post
        self.f0_predictor = mel2wav.f0_predictor
        self.f0_upsamp = mel2wav.f0_upsamp
        self.m_source = mel2wav.m_source
        self.audio_limit = mel2wav.audio_limit
        self.register_buffer(
            "stft_window", mel2wav.stft_window.float().clone(), persistent=False
        )

        # trim_fade ramp (fixed)
        trim_fade = torch.zeros(2 * self.n_trim)
        trim_fade[self.n_trim :] = (
            torch.cos(torch.linspace(torch.pi, 0, self.n_trim)) + 1
        ) / 2
        self.register_buffer("trim_fade", trim_fade.float(), persistent=False)

        # Custom ISTFT replacing torch.istft
        self.istft = CustomISTFT(self.n_fft, self.hop_len, self.n_fft)

        # Precomputed sine-gen phase offset. Upstream samples this at runtime
        # from U(-pi, pi). We bake it in as a constant buffer so the trace
        # does not emit aten::uniform (which opset 17 cannot export).
        # Index 0 stays at 0 (matches upstream's `phase_vec[:, 0, :] = 0`).
        sg = self.m_source.l_sin_gen
        H = sg.harmonic_num + 1
        g = torch.Generator().manual_seed(17)
        phase_vec = torch.empty(1, H, 1).uniform_(
            -float(torch.pi), float(torch.pi), generator=g
        )
        phase_vec[:, 0, :] = 0
        self.register_buffer("sine_phase_vec", phase_vec.float(), persistent=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mask_to_bias(mask_bool: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        mask = mask_bool.to(dtype)
        return (1.0 - mask) * -1.0e10

    def _cond_forward(self, x, mask, mu, t, spks, cond) -> torch.Tensor:
        """Single UNet estimator call (equivalent to decoder.py ConditionalDecoder.forward
        with static_chunk_size==0, meanflow=False).
        """
        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        x = torch.cat([x, mu], dim=1)
        spks_exp = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.cat([x, spks_exp], dim=1)
        x = torch.cat([x, cond], dim=1)

        masks = [mask]
        resnet, transformer_blocks, downsample = self.down_blocks[0]
        mask_down = masks[-1]
        x = resnet(x, mask_down, t)
        x = x.permute(0, 2, 1).contiguous()
        attn_mask = self._mask_to_bias(mask_down.bool(), x.dtype)
        for transformer_block in transformer_blocks:
            x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
        x = x.permute(0, 2, 1).contiguous()
        skip = x
        x = downsample(x * mask_down)
        masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = x.permute(0, 2, 1).contiguous()
            attn_mask = self._mask_to_bias(mask_mid.bool(), x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x, attention_mask=attn_mask, timestep=t
                )
            x = x.permute(0, 2, 1).contiguous()

        resnet, transformer_blocks, upsample = self.up_blocks[0]
        mask_up = masks.pop()
        x = torch.cat([x[:, :, : skip.shape[-1]], skip], dim=1)
        x = resnet(x, mask_up, t)
        x = x.permute(0, 2, 1).contiguous()
        attn_mask = self._mask_to_bias(mask_up.bool(), x.dtype)
        for transformer_block in transformer_blocks:
            x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
        x = x.permute(0, 2, 1).contiguous()
        x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        return self.final_proj(x * mask_up)

    def _flow_encode(
        self,
        speech_tokens: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        speaker_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the flow encoder to get mu, spks, cond, mask, mel_len1."""
        B = speech_tokens.size(0)
        # xvec projection
        embedding = F.normalize(speaker_embeddings, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)  # (B, 80)

        # token embedding
        token_len = torch.full(
            (B,), speech_tokens.size(1), dtype=torch.long, device=speech_tokens.device
        )
        # 1 for non-pad (B, T, 1); we assume no padding inside the batch.
        tmask = torch.ones(
            B, speech_tokens.size(1), 1, dtype=speaker_embeddings.dtype,
            device=speech_tokens.device,
        )
        tok_emb = self.input_embedding(torch.clamp(speech_tokens, min=0).long())
        tok_emb = tok_emb * tmask

        # conformer encoder -> (B, T*ratio, C)
        h, h_masks = self.encoder(tok_emb, token_len)
        # (skip pre-lookahead trim; for inference `finalize=True`)
        h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)
        mel_len1 = speaker_features.shape[1]
        h = self.encoder_proj(h)  # (B, T_mel, 80)

        mel_total = h.shape[1]
        # conds[:, :mel_len1] = speaker_features
        # NOTE: we build it scatter-free via cat(prompt, zeros)
        pad_feat = torch.zeros(
            B, mel_total - mel_len1, self.output_size,
            dtype=h.dtype, device=h.device,
        )
        conds = torch.cat([speaker_features, pad_feat], dim=1).transpose(1, 2)  # (B, 80, T_mel)

        mu = h.transpose(1, 2).contiguous()  # (B, 80, T_mel)

        # mask for the mel: (B, 1, T_mel), 1 for non-pad. h_lengths==T_mel in
        # our single-batch inference.
        mask = torch.ones(
            B, 1, mel_total, dtype=h.dtype, device=h.device,
        )
        return mel_len1, mu, embedding, conds, mask

    # ------------------------------------------------------------------
    # CFM Euler solver, Python-unrolled and scatter-free.
    # ------------------------------------------------------------------
    def _cfm_solve(self, mu, mask, spks, cond):
        """
        Args:
          mu:   (B, 80, T_mel)
          mask: (B, 1,  T_mel)
          spks: (B, 80)
          cond: (B, 80, T_mel)
        Returns:
          x: (B, 80, T_mel)
        """
        x = torch.randn_like(mu) * 1.0
        t_span = torch.linspace(
            0, 1, self.n_timesteps + 1, device=mu.device, dtype=mu.dtype
        )
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # Zero buffers for the uncond half (matches upstream's initial
        # torch.zeros([2*B, ...]) with only the cond half getting written).
        # Upstream DOES duplicate x, mask and t across both halves (see
        # `x_in[:B] = x_in[B:] = x` pattern in
        # chatterbox.models.s3gen.flow_matching.ConditionalCFM.solve_euler).
        # Only mu, spks and cond are zeroed in the uncond half.
        zeros_mu = torch.zeros_like(mu)
        zeros_spks = torch.zeros_like(spks)
        zeros_cond = torch.zeros_like(cond)

        # Python-unrolled loop -- n_timesteps fixed at construction time.
        # Each iteration builds fresh batched CFG tensors via cat, which
        # avoids the ScatterND produced by the upstream in-place .copy_().
        for step in range(self.n_timesteps):
            t = t_span[step : step + 1]
            dt = t_span[step + 1 : step + 2] - t

            # Build (2B, ...) CFG-batched tensors.
            # cond half  =  (x,   mask, mu,      t, spks,      cond)
            # uncond half = (x,   mask, zeros,   t, zeros,     zeros)
            t_single = t.expand(x.size(0))
            x_in = torch.cat([x, x], dim=0)
            mask_in = torch.cat([mask, mask], dim=0)
            mu_in = torch.cat([mu, zeros_mu], dim=0)
            t_in = torch.cat([t_single, t_single], dim=0)
            spks_in = torch.cat([spks, zeros_spks], dim=0)
            cond_in = torch.cat([cond, zeros_cond], dim=0)

            dphi_dt = self._cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt_cond, dphi_dt_uncond = torch.split(
                dphi_dt, [x.size(0), x.size(0)], dim=0
            )
            combined = (1.0 + self.inference_cfg_rate) * dphi_dt_cond \
                       - self.inference_cfg_rate * dphi_dt_uncond
            x = x + dt * combined

        return x

    # ------------------------------------------------------------------
    # Scatter-free SineGen. Upstream does `F_mat[:, i:i+1, :] = f0 * (i+1)/sr`
    # inside a Python loop, which produces harmonic_num+1 ScatterND nodes.
    # We replace that with torch.arange + broadcast to build the full harmonic
    # bank in one multiply. Logically identical.
    # ------------------------------------------------------------------
    def _sine_gen(self, f0: torch.Tensor) -> torch.Tensor:
        """f0: (B, 1, N). Returns sine_waves: (B, harmonic+1, N).

        Exactly reproduces chatterbox.models.s3gen.hifigan.SineGen.forward but
        with no in-place scatter writes. `phase_vec` is still baked in from
        the traced random sample (same as upstream at export time -- the
        reference script uses the same approach).
        """
        sg = self.m_source.l_sin_gen
        H = sg.harmonic_num + 1
        # harmonics = [1, 2, ..., H], broadcast over (B, H, N)
        harmonics = torch.arange(1, H + 1, device=f0.device, dtype=f0.dtype).view(1, H, 1)
        F_mat = f0 * harmonics / sg.sampling_rate  # (B, H, N)

        theta_mat = 2 * torch.pi * (torch.cumsum(F_mat, dim=-1) % 1)

        # Deterministic phase offset -- precomputed at __init__ to avoid
        # aten::uniform in the trace (opset 17 cannot export it).
        phase_vec = self.sine_phase_vec.to(dtype=f0.dtype, device=f0.device)
        sine_waves = sg.sine_amp * torch.sin(theta_mat + phase_vec)

        uv = (f0 > sg.voiced_threshold).to(f0.dtype)
        noise_amp = uv * sg.noise_std + (1 - uv) * sg.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv

    def _source_module(self, x: torch.Tensor) -> torch.Tensor:
        """Scatter-free replacement for SourceModuleHnNSF.forward."""
        sine_wavs, uv = self._sine_gen(x.transpose(1, 2))  # (B, H, N)
        sine_wavs = sine_wavs.transpose(1, 2)  # (B, N, H)
        sine_merge = self.m_source.l_tanh(self.m_source.l_linear(sine_wavs))
        return sine_merge

    # ------------------------------------------------------------------
    # HiFi-GAN forward -- mel -> waveform
    # ------------------------------------------------------------------
    def _hifigan_decode(self, speech_feat: torch.Tensor) -> torch.Tensor:
        """
        speech_feat: (B, 80, T_mel_gen)
        returns:     (B, N_samples)
        """
        # mel -> f0 -> sine source (via our scatter-free SineGen)
        f0 = self.f0_predictor(speech_feat)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # (B, N_samples_raw, 1)
        s = self._source_module(s)
        output_sources = s.transpose(1, 2).squeeze(1)  # (B, N_samples_raw)

        # real-valued STFT (return_complex=False, for ONNX export)
        spec = torch.stft(
            output_sources,
            self.n_fft,
            self.hop_len,
            self.n_fft,
            window=self.stft_window.to(output_sources.device),
            return_complex=False,
        )
        s_stft_real, s_stft_imag = spec[..., 0], spec[..., 1]
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)  # (B, n_fft+2, T)

        # HiFiGAN body
        x = self.conv_pre(speech_feat)

        # upsample 0
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[0](x)
        si = self.source_downs[0](s_stft)
        si = self.source_resblocks[0](si)
        x = x + si
        xs = (
            self.resblocks[0](x) + self.resblocks[1](x) + self.resblocks[2](x)
        )
        x = xs / 3

        # upsample 1
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[1](x)
        si = self.source_downs[1](s_stft)
        si = self.source_resblocks[1](si)
        x = x + si
        xs = (
            self.resblocks[3](x) + self.resblocks[4](x) + self.resblocks[5](x)
        )
        x = xs / 3

        # upsample 2 (last: reflection pad)
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[2](x)
        x = self.reflection_pad(x)
        si = self.source_downs[2](s_stft)
        si = self.source_resblocks[2](si)
        x = x + si
        xs = (
            self.resblocks[6](x) + self.resblocks[7](x) + self.resblocks[8](x)
        )
        x = xs / 3

        # final
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, : self.n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.n_fft // 2 + 1 :, :])

        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        rebuild = torch.cat([real, img], dim=1)
        wavs = self.istft(rebuild)

        # fade-in on first 2*n_trim samples, hard-limited
        trim_fade = self.trim_fade.to(dtype=wavs.dtype, device=wavs.device)
        # Build a per-sample scale: first len(trim_fade) samples are fade-in,
        # the rest are 1.0. We do this scatter-free by cat.
        ones = torch.ones(wavs.size(1) - trim_fade.size(0), dtype=wavs.dtype, device=wavs.device)
        scale = torch.cat([trim_fade, ones], dim=0)
        wavs = wavs * scale.unsqueeze(0)
        wavs = torch.clamp(wavs, -self.audio_limit, self.audio_limit)
        return wavs

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(
        self,
        speech_tokens: torch.Tensor,       # (B, N_tok)  int64
        speaker_embeddings: torch.Tensor,  # (B, 192)    float
        speaker_features: torch.Tensor,    # (B, N_mel_prompt, 80) float
    ) -> torch.Tensor:
        mel_len1, mu, spks, cond, mask = self._flow_encode(
            speech_tokens, speaker_embeddings, speaker_features
        )
        mel = self._cfm_solve(mu, mask, spks, cond)  # (B, 80, T_mel_total)
        # trim off the prompt portion
        mel_gen = mel[:, :, mel_len1:]
        wavs = self._hifigan_decode(mel_gen)
        return wavs


# ---------------------------------------------------------------------------
# Build dummy speaker inputs by running the reference embed_ref on a reference
# audio clip. Used both to drive the torch trace and to build ORT smoke inputs.
# ---------------------------------------------------------------------------
