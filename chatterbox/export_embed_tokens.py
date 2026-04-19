"""Export Chatterbox's input embedding layer (text/speech tokens + positions → embeddings).

The ``InputsEmbeds`` wrapper replaces Chatterbox's multi-step embedding lookup
with a single ONNX-traceable graph. Given ``(input_ids, position_ids, exaggeration)``
it returns the per-token embeddings that feed into the T3 language model.

The original monolithic export lived in ``chatterbox_to_onnx_conversion_script.py``;
this split separates it from the speech encoder and conditional decoder exports
without changing behavior.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._constants import (
    START_SPEECH_TOKEN, STOP_SPEECH_TOKEN, EXAGGERATION_TOKEN,
)




class InputsEmbeds(nn.Module):
    def __init__(self, chatterbox):
        super().__init__()
        self.text_emb = chatterbox.t3.text_emb
        self.text_pos_emb = chatterbox.t3.text_pos_emb.emb

        self.speech_emb = chatterbox.t3.speech_emb
        self.speech_pos_emb = chatterbox.t3.speech_pos_emb.emb

        self.emotion_adv_fc = chatterbox.t3.cond_enc.emotion_adv_fc

    def forward(self, input_ids, position_ids, exaggeration):
        assert position_ids.shape == input_ids.shape
        batch_size, seq_len = input_ids.shape

        x = input_ids
        idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Detect first zero
        is_zero = (x == 0)
        has_zero = is_zero.any(dim=1)
        zero_pos = torch.where(
            has_zero,
            is_zero.float().argmax(dim=1),
            torch.full((batch_size,), -1, device=x.device)  # placeholder
        )

        # Masks
        exaggeration_mask = (x == EXAGGERATION_TOKEN)
        base_text_mask = (idx <= zero_pos.unsqueeze(1)) & has_zero.unsqueeze(1)
        
        text_mask = base_text_mask & ~exaggeration_mask
        speech_mask = ~base_text_mask & ~exaggeration_mask

        # Compute relative positions by multiplying with the masks
        text_pos_ids = position_ids * text_mask
        speech_pos_ids = position_ids * speech_mask

        # Flatten
        flat_x = x.view(-1)
        flat_text_mask = text_mask.view(-1)
        flat_speech_mask = speech_mask.view(-1)
        flat_exaggeration_mask = exaggeration_mask.view(-1)
        flat_text_pos = text_pos_ids.view(-1)
        flat_speech_pos = speech_pos_ids.view(-1)

        # Replace invalid indices with 0 (safe padding idx)
        safe_text_idx = torch.where(flat_text_mask, flat_x, torch.zeros_like(flat_x))
        safe_text_pos = torch.where(flat_text_mask, flat_text_pos, torch.zeros_like(flat_text_pos))

        safe_speech_idx = torch.where(flat_speech_mask, flat_x, torch.zeros_like(flat_x))
        safe_speech_pos = torch.where(flat_speech_mask, flat_speech_pos, torch.zeros_like(flat_speech_pos))

        # Embed everything, but irrelevant positions will become "padding" embeddings
        all_text_emb = self.text_emb(safe_text_idx) + self.text_pos_emb(safe_text_pos)
        all_speech_emb = self.speech_emb(safe_speech_idx) + self.speech_pos_emb(safe_speech_pos)

        # Finally, mask out the padding positions to zero them
        text_emb = all_text_emb * flat_text_mask.unsqueeze(-1)
        speech_emb = all_speech_emb * flat_speech_mask.unsqueeze(-1)

        # Emotion Adv: must provide a value if this model uses emotion conditioning
        emotion_adv = exaggeration.view(-1, 1, 1)
        cond_emotion_adv = self.emotion_adv_fc(emotion_adv)

        # Reshape to [B*L, D] to match masks
        embed_dim = text_emb.size(-1)
        text_emb_full   = text_emb
        speech_emb_full = speech_emb

        # Start with zeros
        out = torch.zeros(batch_size * seq_len, embed_dim, device=x.device, dtype=text_emb.dtype)

        # Where text mask is True → take text_emb, else keep current out
        out = torch.where(flat_text_mask.unsqueeze(-1), text_emb_full, out)

        # Where speech mask is True → take speech_emb, else keep current out
        out = torch.where(flat_speech_mask.unsqueeze(-1), speech_emb_full, out)
        
        # Handle exaggeration tokens
        # We need to expand cond_emotion_adv to match the number of exaggeration tokens
        # This assumes cond_emotion_adv is (batch_size, 1, dim) and we need to map it correctly
        # to the flattened positions.
        # We can create an index mapping from the flattened index to the batch index.
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, seq_len).reshape(-1)
        exaggeration_emb_full = cond_emotion_adv[batch_indices].transpose(0, 1) 

        # Zero out positions where mask is False
        exaggeration_emb = exaggeration_emb_full * flat_exaggeration_mask.unsqueeze(-1)

        out = out + exaggeration_emb
        out = out.view(batch_size, seq_len, embed_dim)
        return out

