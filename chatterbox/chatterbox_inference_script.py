"""Reference ONNX inference script for the v2 Chatterbox export.

Drives the four graphs produced by ``chatterbox_to_onnx_conversion_script``:

- ``speech_encoder.onnx``      (unchanged from upstream)
- ``embed_tokens.onnx``        (batch=2 CFG-aware, scatter-free)
- ``language_model.onnx``      (CFG baked in, alignment attention exposed, fp16)
- ``conditional_decoder.onnx`` (parameterized CFM steps, scatter-free)

The language model inputs/outputs are fp16 so inputs_embeds, cfg_weight, and
past_key_values are all cast to fp16 before each call.  The alignment-attention
output drives :class:`alignment_runtime.AlignmentStreamAnalyzer`, which forces
EOS on short utterances to prevent trailing-speech hallucinations
(cf. resemble-ai/chatterbox#97).

Run ``chatterbox_to_onnx_conversion_script.export_model_to_onnx`` first to
produce the .onnx files, then point ``models_dir`` at the same output folder.
"""
# !pip install --upgrade onnxruntime==1.22.1 huggingface_hub==0.34.4 transformers==4.46.3 numpy==2.2.6 tqdm==4.67.1 librosa==0.11.0 soundfile==0.13.1 resemble-perth==1.0.1

import os
from pathlib import Path

import onnxruntime

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf

from ._constants import S3GEN_SR, START_SPEECH_TOKEN, STOP_SPEECH_TOKEN
from .alignment_runtime import AlignmentStreamAnalyzer

NUM_HIDDEN_LAYERS = 30
NUM_KEY_VALUE_HEADS = 16
HEAD_DIM = 64


class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` must be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


def _sample(logits: np.ndarray, generate_tokens: np.ndarray, temperature: float,
            rep_processor: RepetitionPenaltyLogitsProcessor) -> np.ndarray:
    logits = rep_processor(generate_tokens, logits.astype(np.float32, copy=False))
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    if temperature > 0:
        lmax = np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits - lmax)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        return np.array([[np.random.choice(probs.shape[-1], p=probs[0])]], dtype=np.int64)
    return np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)


def run_inference(
    text="The Lord of the Rings is the greatest work of literature.",
    target_voice_path=None,
    max_new_tokens=800,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    repetition_penalty=2.0,
    models_dir="converted",
    output_file_name="output.wav",
    apply_watermark=False,
):
    """Run the v2 ONNX inference pipeline end-to-end.

    Parameters
    ----------
    models_dir : str
        Directory containing the four .onnx files + sidecars produced by
        ``chatterbox_to_onnx_conversion_script.export_model_to_onnx``.
    cfg_weight : float
        CFG strength; 0.5 matches the PyTorch defaults.
    temperature : float
        Sampling temperature.  ``0`` disables sampling (argmax).
    repetition_penalty : float
        2.0 matches the PyTorch inference defaults.
    """
    models_dir = Path(models_dir)
    tokenizer_id = "onnx-community/chatterbox-onnx"  # tokenizer only — graphs come from models_dir
    if not target_voice_path:
        target_voice_path = hf_hub_download(
            repo_id=tokenizer_id, filename="default_voice.wav", local_dir=str(models_dir)
        )

    speech_encoder_path = models_dir / "speech_encoder.onnx"
    embed_tokens_path = models_dir / "embed_tokens.onnx"
    language_model_path = models_dir / "language_model.onnx"
    conditional_decoder_path = models_dir / "conditional_decoder.onnx"
    for p in (speech_encoder_path, embed_tokens_path, language_model_path, conditional_decoder_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing ONNX file: {p}.  Run "
                f"chatterbox_to_onnx_conversion_script.export_model_to_onnx(output_export_dir=...) first."
            )

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers() \
        else ["CPUExecutionProvider"]

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    speech_encoder_session = onnxruntime.InferenceSession(str(speech_encoder_path), sess_options, providers=providers)
    embed_tokens_session = onnxruntime.InferenceSession(str(embed_tokens_path), sess_options, providers=providers)
    language_model_session = onnxruntime.InferenceSession(str(language_model_path), sess_options, providers=providers)
    cond_decoder_session = onnxruntime.InferenceSession(str(conditional_decoder_path), sess_options, providers=providers)

    lm_output_names = [o.name for o in language_model_session.get_outputs()]

    def _run_lm(inputs_embeds, attention_mask, cfg_scalar, past_kv):
        feed = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cfg_weight": cfg_scalar,
        }
        feed.update(past_kv)
        outs = language_model_session.run(lm_output_names, feed)
        return dict(zip(lm_output_names, outs))

    def _present_to_past(out_dict):
        return {
            f"past_key_values.{n}.{kv}": out_dict[f"present.{n}.{kv}"]
            for n in range(NUM_HIDDEN_LAYERS) for kv in ("key", "value")
        }

    def execute_text_to_audio_inference(text):
        print("Start inference script...")

        audio_values, _ = librosa.load(target_voice_path, sr=S3GEN_SR)
        audio_values = audio_values[np.newaxis, :].astype(np.float32)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        input_ids = tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
        position_ids = np.where(
            input_ids >= START_SPEECH_TOKEN,
            0,
            np.arange(input_ids.shape[1])[np.newaxis, :] - 1,
        ).astype(np.int64)

        # Speech encoder (one-shot).
        cond_emb_np, prompt_token, ref_x_vector, prompt_feat = speech_encoder_session.run(
            None, {"audio_values": audio_values}
        )
        # Broadcast cond_emb to batch=2 for CFG and cast to fp16.
        cond_emb_b2 = np.broadcast_to(
            cond_emb_np.astype(np.float16),
            (2, cond_emb_np.shape[1], cond_emb_np.shape[2]),
        ).copy()
        cond_len = cond_emb_b2.shape[1]

        # Embed text tokens with BOS appended — matches PyTorch t3.inference()
        # which concatenates BOS onto the text embeddings before prefill.
        bos_ids = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
        input_ids_with_bos = np.concatenate([input_ids, bos_ids], axis=1)
        position_ids_with_bos = np.concatenate([position_ids, np.array([[0]], dtype=np.int64)], axis=1)

        input_ids_b2 = np.concatenate([input_ids_with_bos, input_ids_with_bos], axis=0)
        position_ids_b2 = np.concatenate([position_ids_with_bos, position_ids_with_bos], axis=0)
        text_embeds = embed_tokens_session.run(None, {
            "input_ids": input_ids_b2,
            "position_ids": position_ids_b2,
            "exaggeration": np.array([exaggeration], dtype=np.float32),
        })[0]
        text_len = input_ids.shape[1]

        # Prefill inputs: concatenate speaker-cond embeddings with text-with-BOS.
        prefill_embeds = np.concatenate([cond_emb_b2, text_embeds], axis=1)
        batch_size, prefill_len, _ = prefill_embeds.shape
        attention_mask = np.ones((batch_size, prefill_len), dtype=np.int64)
        past_kv = {
            f"past_key_values.{l}.{kv}": np.zeros((batch_size, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM), dtype=np.float16)
            for l in range(NUM_HIDDEN_LAYERS) for kv in ("key", "value")
        }
        cfg_scalar = np.array(cfg_weight, dtype=np.float16)

        rep_proc = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)

        # Text span inside the prefill is [cond_len, cond_len + text_len).
        analyzer = AlignmentStreamAnalyzer(
            text_tokens_slice=(cond_len, cond_len + text_len),
            eos_idx=STOP_SPEECH_TOKEN,
        )

        # Prefill.
        out = _run_lm(prefill_embeds, attention_mask, cfg_scalar, past_kv)
        past_kv = _present_to_past(out)
        logits_step = out["logits"][:, -1, :].astype(np.float32)
        logits_step = analyzer.step(logits_step, out["attn_layers"], next_token=None)
        next_token = _sample(logits_step, generate_tokens, temperature, rep_proc)
        generate_tokens = np.concatenate([generate_tokens, next_token], axis=-1)

        if int(next_token[0, 0]) != STOP_SPEECH_TOKEN:
            for i in tqdm(range(1, max_new_tokens), desc="Sampling", dynamic_ncols=True):
                nt_b2 = np.concatenate([next_token, next_token], axis=0)
                pos_ids_b2 = np.full((2, 1), i, dtype=np.int64)
                step_embeds = embed_tokens_session.run(None, {
                    "input_ids": nt_b2,
                    "position_ids": pos_ids_b2,
                    "exaggeration": np.array([exaggeration], dtype=np.float32),
                })[0]
                attention_mask = np.concatenate(
                    [attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1
                )
                out = _run_lm(step_embeds, attention_mask, cfg_scalar, past_kv)
                past_kv = _present_to_past(out)
                logits_step = out["logits"][:, -1, :].astype(np.float32)
                logits_step = analyzer.step(logits_step, out["attn_layers"], next_token=int(next_token[0, 0]))
                next_token = _sample(logits_step, generate_tokens, temperature, rep_proc)
                generate_tokens = np.concatenate([generate_tokens, next_token], axis=-1)
                if int(next_token[0, 0]) == STOP_SPEECH_TOKEN:
                    break

        # Strip BOS + EOS (if present), prepend the reference prompt tokens.
        speech_tokens = generate_tokens[:, 1:]
        if speech_tokens.size > 0 and speech_tokens[0, -1] == STOP_SPEECH_TOKEN:
            speech_tokens = speech_tokens[:, :-1]
        speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)
        return speech_tokens, ref_x_vector, prompt_feat

    speech_tokens, speaker_embeddings, speaker_features = execute_text_to_audio_inference(text)
    wav = cond_decoder_session.run(None, {
        "speech_tokens": speech_tokens,
        "speaker_embeddings": speaker_embeddings,
        "speaker_features": speaker_features,
    })[0]
    wav = np.squeeze(wav, axis=0)

    if apply_watermark:
        import perth
        import torch
        watermarker = perth.PerthImplicitWatermarker()
        # Wrap in torch.no_grad — Perth's apply_watermark doesn't set it
        # internally, and autograd state accumulation leaks ~35 MiB/call.
        with torch.no_grad():
            wav = watermarker.apply_watermark(wav, sample_rate=S3GEN_SR)

    sf.write(output_file_name, wav, S3GEN_SR)
    print(f"{output_file_name} was successfully saved")


if __name__ == "__main__":
    run_inference(
        text="Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.",
        exaggeration=0.5,
        output_file_name="output.wav",
        apply_watermark=False,
    )
