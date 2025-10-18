# !pip install --upgrade torch==2.6.0 torchaudio==2.6.0 resemble-perth==1.0.1 numpy==1.25.0 onnx==1.19.0 onnxslim==0.1.71
# recommend to use python version 3.10 as some libraries do not support a higher one

import torch
from torch import nn
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker
from perth.perth_net.perth_net_implicit.utils import denormalize_spectrogram, normalize
import torchaudio as ta
import numpy as np
from torch.functional import F

ISTFT_PARAMS = {"n_fft": 2048, "hop_len": 320, "win_len": 2048}

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

istft = ISTFT(ISTFT_PARAMS["n_fft"], ISTFT_PARAMS["hop_len"], ISTFT_PARAMS["win_len"])

class ImplicitWatermarker(nn.Module):
    def __init__(self, model, fixed_sample_rate: int):
        super().__init__()
        self.perth_net = model.perth_net
        self.default_sample_rate = model.perth_net.hp.sample_rate
        self.fixed_sample_rate = fixed_sample_rate
        self.istft = istft
        self.hp = model.perth_net.hp
        self.n_fft = model.perth_net.hp.n_fft
        self.win_length = model.perth_net.hp.window_size
        self.hop_length = model.perth_net.hp.hop_size
        self.window = model.perth_net.ap.spectrogram.window
    
    def spectrogram(
        self,
        waveform: torch.Tensor,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        ) -> torch.Tensor:

        # Pack batch
        shape = waveform.size()
        waveform = waveform.reshape(-1, shape[-1])

        # Perform STFT with return_complex=False
        spec_f = torch.stft(
            input=waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            return_complex=False,
        )

        # Unpack batch
        spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-3:])
        return spec_f

    def magphase_to_realimag(self, hp, magspec, phases):
        """
        Convert magnitude + phase (radians) to real and imaginary components
        without using complex dtype.
        """
        magspec = denormalize_spectrogram(hp, magspec)
        magspec = 10. ** ((magspec / 20).clamp(max=10))

        real = magspec * torch.cos(phases)
        imag = magspec * torch.sin(phases)
        return real, imag
    
    def cx_to_magphase(self, hp, spec):
        """
        ONNX-safe replacement for converting complex spectrograms
        to magnitude and phase tensors.
        """
        # Convert complex to real/imag
        real = spec[..., 0]
        imag = spec[..., 1]

        # ONNX-safe equivalents of abs() and angle()
        mag = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag, real)

        # Convert to log magnitude (dB)
        mag = 20 * torch.log10(torch.clamp(mag, min=hp.stft_magnitude_min))
        mag = normalize(hp, mag)
        return mag, phase

    def forward(self, audio_values):
        audio_values = ta.functional.resample(audio_values, orig_freq=self.fixed_sample_rate, new_freq=self.default_sample_rate)
        audio_values = audio_values.to(self.perth_net.device)
        signal = audio_values.float()
        spec = self.spectrogram(signal).squeeze(0)

        # split signal into magnitude and phase
        magspec, phase = self.cx_to_magphase(self.hp, spec)
        magspec = magspec.unsqueeze(0)
        phase = phase.unsqueeze(0)

        # encode the watermark
        wm_magspec, _ = self.perth_net.encoder(magspec)
        wm_magspec = wm_magspec[0]

        real, imag = self.magphase_to_realimag(self.hp, wm_magspec, phase )
        recombine_magnitude_phase = torch.cat([real, imag], dim=1)
        wm_signal = self.istft(recombine_magnitude_phase)
        # assemble back into watermarked signal
        wm_signal = ta.functional.resample(wm_signal, self.default_sample_rate, self.fixed_sample_rate)
        return wm_signal


def export_model_to_onnx(output_export_dir="output", sample_rates_for_export=[16000, 24000, 44100, 48000]):
    import onnx
    if output_export_dir:
        import os
        os.makedirs(output_export_dir, exist_ok=True)
    watermarker = PerthImplicitWatermarker()
    dummy_audio_values = torch.randn(1, 312936)
    for sr in sample_rates_for_export:
        implicit_watermarker = ImplicitWatermarker(watermarker, sr)
        torch.onnx.export(
            implicit_watermarker,
            (dummy_audio_values),
            f"{output_export_dir}/implicit_watermarker_{sr}.onnx",
            export_params=True,
            opset_version=20,
            input_names=["audio_values"],
            output_names=["watermarked_audio_values"],
            dynamic_axes={
                "audio_values": {0: "batch_size", 1: "num_samples"},
                "watermarked_audio_values": {0: "batch_size", 1: "num_samples"},
            },
        )
        print(f"✅ Implicit Watermarker ONNX export for {sr} is completed. Model saved as 'implicit_watermarker_{sr}.onnx'")

    import onnxslim
    import onnx
    for f in os.listdir(output_export_dir):
        if not f.endswith(".onnx"):
            continue
        save_path = os.path.join(output_export_dir, f)
        model = onnxslim.slim(save_path)
        onnx.save_model(model, save_path, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(save_path) + "_data")


if __name__ == "__main__":
    export_model_to_onnx()
