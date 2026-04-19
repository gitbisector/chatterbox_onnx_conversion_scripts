"""Reference ONNX inference script for the v2 Chatterbox Multilingual export.

Same design as ``chatterbox_inference_script`` but with the per-language text
preprocessing (Cangjie for Chinese, kakasi for Japanese, dicta for Hebrew,
Jamo decomposition for Korean) carried over from upstream.

Drives the four graphs produced by ``chatterbox_to_onnx_conversion_script``
with ``multilingual=True``.

Run ``chatterbox_to_onnx_conversion_script.export_model_to_onnx(multilingual=True, ...)``
first to produce the .onnx files, then point ``models_dir`` at the same output folder.
"""
# !pip install --upgrade onnxruntime==1.22.1 huggingface_hub==0.34.4 transformers==4.46.3 numpy==2.2.6 tqdm==4.67.1 librosa==0.11.0 soundfile==0.13.1 resemble-perth==1.0.1
# for Chinese, Japanese additionally pip install pkuseg==0.0.25 pykakasi==2.3.0

import json
from pathlib import Path
from unicodedata import category

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

SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}

_kakasi = None
_dicta = None


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


class ChineseCangjieConverter:
    """Converts Chinese characters to Cangjie codes for tokenization."""

    def __init__(self):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        self._load_cangjie_mapping()
        self._init_segmenter()

    def _load_cangjie_mapping(self):
        """Load Cangjie mapping from HuggingFace model repository."""
        try:
            cangjie_file = hf_hub_download(
                repo_id="onnx-community/chatterbox-multilingual-ONNX",
                filename="Cangjie5_TC.json",
            )

            with open(cangjie_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            for entry in data:
                word, code = entry.split("\t")[:2]
                self.word2cj[word] = code
                if code not in self.cj2word:
                    self.cj2word[code] = [word]
                else:
                    self.cj2word[code].append(word)

        except Exception as e:
            print(f"Could not load Cangjie mapping: {e}")

    def _init_segmenter(self):
        """Initialize pkuseg segmenter."""
        try:
            from pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            print("pkuseg not available - Chinese segmentation will be skipped")
            self.segmenter = None

    def _cangjie_encode(self, glyph: str):
        """Encode a single Chinese glyph to Cangjie code."""
        normed_glyph = glyph
        code = self.word2cj.get(normed_glyph, None)
        if code is None:  # e.g. Japanese hiragana
            return None
        index = self.cj2word[code].index(normed_glyph)
        index = str(index) if index > 0 else ""
        return code + str(index)

    def __call__(self, text):
        """Convert Chinese characters in text to Cangjie tokens."""
        output = []
        if self.segmenter is not None:
            segmented_words = self.segmenter.cut(text)
            full_text = " ".join(segmented_words)
        else:
            full_text = text

        for t in full_text:
            if category(t) == "Lo":
                cangjie = self._cangjie_encode(t)
                if cangjie is None:
                    output.append(t)
                    continue
                code = []
                for c in cangjie:
                    code.append(f"[cj_{c}]")
                code.append("[cj_.]")
                code = "".join(code)
                output.append(code)
            else:
                output.append(t)
        return "".join(output)


def is_kanji(c: str) -> bool:
    """Check if character is kanji."""
    return 19968 <= ord(c) <= 40959


def is_katakana(c: str) -> bool:
    """Check if character is katakana."""
    return 12449 <= ord(c) <= 12538


def hiragana_normalize(text: str) -> str:
    """Japanese text normalization: converts kanji to hiragana; katakana remains the same."""
    global _kakasi

    try:
        if _kakasi is None:
            import pykakasi
            _kakasi = pykakasi.kakasi()

        result = _kakasi.convert(text)
        out = []

        for r in result:
            inp = r['orig']
            hira = r["hira"]

            # Any kanji in the phrase
            if any([is_kanji(c) for c in inp]):
                if hira and hira[0] in ["は", "へ"]:  # Safety check for empty hira
                    hira = " " + hira
                out.append(hira)

            # All katakana
            elif all([is_katakana(c) for c in inp]) if inp else False:  # Safety check for empty inp
                out.append(r['orig'])

            else:
                out.append(inp)

        normalized_text = "".join(out)

        # Decompose Japanese characters for tokenizer compatibility
        import unicodedata
        normalized_text = unicodedata.normalize('NFKD', normalized_text)

        return normalized_text

    except ImportError:
        print("pykakasi not available - Japanese text processing skipped")
        return text


def add_hebrew_diacritics(text: str) -> str:
    """Hebrew text normalization: adds diacritics to Hebrew text."""
    global _dicta

    try:
        if _dicta is None:
            from dicta_onnx import Dicta
            _dicta = Dicta()

        return _dicta.add_diacritics(text)

    except ImportError:
        print("dicta_onnx not available - Hebrew text processing skipped")
        return text
    except Exception as e:
        print(f"Hebrew diacritization failed: {e}")
        return text


def korean_normalize(text: str) -> str:
    """Korean text normalization: decompose syllables into Jamo for tokenization."""

    def decompose_hangul(char):
        """Decompose Korean syllable into Jamo components."""
        if not ('\uac00' <= char <= '\ud7af'):
            return char

        # Hangul decomposition formula
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''

        return initial + medial + final

    # Decompose syllables and normalize punctuation
    result = ''.join(decompose_hangul(char) for char in text)
    return result.strip()


def prepare_language(txt, language_id):
    # Language-specific text processing
    cangjie_converter = ChineseCangjieConverter()
    if language_id == 'zh':
        txt = cangjie_converter(txt)
    elif language_id == 'ja':
        txt = hiragana_normalize(txt)
    elif language_id == 'he':
        txt = add_hebrew_diacritics(txt)
    elif language_id == 'ko':
        txt = korean_normalize(txt)

    # Prepend language token
    if language_id:
        txt = f"[{language_id.lower()}]{txt}"
    return txt


def run_inference(
    text="The Lord of the Rings is the greatest work of literature.",
    language_id="en",
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
    """Run the v2 ONNX multilingual inference pipeline end-to-end.

    See ``chatterbox_inference_script.run_inference`` for parameter docs.
    Extra ``language_id`` (ISO code) routes through the per-language text
    normalization (Cangjie / kakasi / dicta / Jamo) and prepends the language
    token before tokenization.
    """
    if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
        supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise ValueError(
            f"Unsupported language_id '{language_id}'. Supported languages: {supported_langs}"
        )

    models_dir = Path(models_dir)
    tokenizer_id = "onnx-community/chatterbox-multilingual-ONNX"  # tokenizer only
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
                f"chatterbox_to_onnx_conversion_script.export_model_to_onnx(multilingual=True, output_export_dir=...) first."
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
        text = prepare_language(text, language_id)
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
        cond_emb_b2 = np.broadcast_to(
            cond_emb_np.astype(np.float16),
            (2, cond_emb_np.shape[1], cond_emb_np.shape[2]),
        ).copy()
        cond_len = cond_emb_b2.shape[1]

        # Text embeddings — append BOS to match PyTorch t3.inference().
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
                attn_layers = out["attn_layers"]

                logits_step = analyzer.step(logits_step, attn_layers, next_token=int(next_token[0, 0]))
                next_token = _sample(logits_step, generate_tokens, temperature, rep_proc)
                generate_tokens = np.concatenate([generate_tokens, next_token], axis=-1)
                if int(next_token[0, 0]) == STOP_SPEECH_TOKEN:
                    break

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
        with torch.no_grad():
            wav = watermarker.apply_watermark(wav, sample_rate=S3GEN_SR)

    sf.write(output_file_name, wav, S3GEN_SR)
    print(f"{output_file_name} was successfully saved")


if __name__ == "__main__":
    run_inference(
        text="Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues.",
        language_id="fr",
        exaggeration=0.5,
        output_file_name="output.wav",
        apply_watermark=False,
    )
