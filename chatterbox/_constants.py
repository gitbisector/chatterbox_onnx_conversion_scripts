"""Shared constants used across the ONNX export modules.

Extracted verbatim from the original ``chatterbox_to_onnx_conversion_script.py``.
"""
import torch

# Sampling rate of the inputs to S3TokenizerV2
S3GEN_SR = 24000
S3_SR = 16_000
S3_HOP = 160
S3_TOKEN_HOP = 640
S3_TOKEN_RATE = 25  # 25 tokens/sec
SPEECH_VOCAB_SIZE = 6561
MILLISECONDS_TO_SECONDS = 0.001

START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
EXAGGERATION_TOKEN = 6563

ENC_COND_LEN = 6 * S3_SR
DEC_COND_LEN = 10 * S3GEN_SR

CFM_PARAMS = {
    "sigma_min": 1e-06,
    "solver": "euler",
    "t_scheduler": "cosine",
    "training_cfg_rate": 0.2,
    "inference_cfg_rate": 0.7,
    "reg_loss_type": "l1",
}
ISTFT_PARAMS = {"n_fft": 16, "hop_len": 4}

# override certain torch functions — shared across all export modules
# so they all see the same patched Tensor behavior during tracing
torch.Tensor.item = lambda x: x  # no-op
