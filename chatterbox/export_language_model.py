"""Export Chatterbox's T3 Llama language model to ONNX.

New export (not present in the original conversion script — the upstream
relied on a separately-published Llama backbone at ``vladislavbro/llama_backbone_0.5``).
This module exports the multilingual T3 backbone directly with three
features that the external Llama export cannot provide:

1. **Classifier-free guidance baked into the graph.**  The multilingual T3
   is trained for CFG.  The graph accepts a ``(2, S, 1024)`` batched
   ``inputs_embeds`` (row 0 = conditional, row 1 = unconditional) plus a
   scalar ``cfg_weight`` and emits a single ``logits`` output of shape
   ``(1, S, V)`` computed as ``cond + cfg_weight * (cond - uncond)``.
   Callers don't need to run the model twice or combine logits outside
   the graph.

2. **Exposed alignment attention.**  Layers 9, 12, and 13 are the layers
   consumed by ``AlignmentStreamAnalyzer`` (``LLAMA_ALIGNED_HEADS``).  The
   graph emits their attention weights as an ``attn_layers`` output of
   shape ``(3, num_heads, S, total_S)``.  Without these, an ONNX-based
   inference loop can't reproduce Chatterbox's EOS-forcing behavior and
   the model hallucinates trailing speech past the end of the input
   (cf. ``resemble-ai/chatterbox#97``).

3. **Scatter-free.**  HF's ``DynamicCache.update`` path is what emits
   ``ScatterND`` for KV-cache writes.  This wrapper bypasses the ``Cache``
   object entirely and manages KV growth via
   ``torch.cat([past, current], dim=-2)`` — ORT CUDA-graph capturable.

Plus fp16 weights with fp32 softmax / residual sums for numerical safety.

Interface
---------
Inputs:
    inputs_embeds   float16 (2, S, 1024)  — cond on row 0, uncond on row 1
    attention_mask  int64   (2, total_S)  — 1 = real, 0 = pad
    cfg_weight      float16 ()            — scalar
    past_key_values.{i}.{key,value}  float16 (2, 16, past_S, 64)  for i in 0..29

Outputs:
    logits          float16 (1, S, V)                        — CFG-combined
    attn_layers     float32 (3, 16, S, total_S)              — layers [9, 12, 13], row 0
    present.{i}.{key,value}  float16 (2, 16, total_S, 64)    for i in 0..29
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# T3 multilingual architecture constants — match chatterbox.models.t3.T3Config
# for the multilingual variant.
NUM_LAYERS = 30
NUM_HEADS = 16
NUM_KV_HEADS = 16  # Llama_520M is NOT GQA; kv_heads == heads
HEAD_DIM = 64
HIDDEN_SIZE = 1024
ALIGN_LAYERS = [9, 12, 13]  # chatterbox.models.t3.inference.alignment_stream_analyzer.LLAMA_ALIGNED_HEADS


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q/k shape: (B, H, S, D); cos/sin shape: (B, S, D).  Matches HF Llama's impl."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class LlamaForCFG(nn.Module):
    """T3 Llama backbone with CFG logits combine + alignment attention output.

    Bypasses HF's ``Cache`` abstraction to keep KV cache writes as pure
    ``Concat`` (scatter-free) and to expose per-layer attention for the three
    alignment layers.

    Weights are cast to fp16 at construction time (via the caller calling
    ``model.half()`` before passing in).  Softmax and residual adds run in
    fp32 for stability — we explicitly upcast Q·K before the softmax and
    downcast the attention weights after.
    """

    def __init__(self, t3):
        super().__init__()
        self.t3 = t3  # keep the whole module so parameters stay registered
        self.llama = t3.tfmr  # transformers.LlamaModel
        self.rotary = self.llama.rotary_emb
        self.final_norm = self.llama.norm
        self.speech_head = t3.speech_head
        self.num_layers = NUM_LAYERS
        self.num_heads = NUM_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.head_dim = HEAD_DIM
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.align_layers = ALIGN_LAYERS

    def _layer_attention(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Custom attention: no Cache object, no SDPA, pure ops.

        Returns: (attn_output, attn_weights, present_key, present_value).
        """
        attn = self.llama.layers[layer_idx].self_attn
        bsz, q_len, _ = hidden_states.shape

        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)

        # Append past along sequence axis (pure Concat — no ScatterND).
        k = torch.cat([past_k, k], dim=-2)
        v = torch.cat([past_v, v], dim=-2)

        # num_kv_heads == num_heads for Llama_520M, so no GQA repeat needed.
        # Keep in fp32 for the score + softmax step.
        q32 = q.float()
        k32 = k.float()
        attn_weights = torch.matmul(q32, k32.transpose(2, 3)) * self.scale
        # causal_mask shape: (B, 1, S, T), additive in fp32
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Back to fp16 for the value product.
        attn_weights_cast = attn_weights.to(v.dtype)
        attn_output = torch.matmul(attn_weights_cast, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights, k, v

    def _layer_forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ):
        layer = self.llama.layers[layer_idx]
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_out, attn_weights, present_k, present_v = self._layer_attention(
            layer_idx, hidden_states, cos, sin, causal_mask, past_k, past_v
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights, present_k, present_v

    def forward(
        self,
        inputs_embeds: torch.Tensor,          # (2, S, 1024) — cond row 0, uncond row 1
        attention_mask: torch.Tensor,         # (2, total_S) — 1 = real, 0 = pad
        cfg_weight: torch.Tensor,             # scalar fp16
        *past_key_values: torch.Tensor,       # flat: [pk0, pv0, pk1, pv1, ...]
    ):
        bsz, q_len, _ = inputs_embeds.shape
        past_len = past_key_values[0].shape[-2]
        total_len = past_len + q_len

        # Position ids for the NEW tokens: [past_len, past_len+1, ..., past_len+q_len-1]
        position_ids = torch.arange(
            past_len, total_len, device=inputs_embeds.device
        ).unsqueeze(0).expand(bsz, -1)

        # Additive causal mask in fp32 with shape (B, 1, S, total_S).
        # 1) key-side padding: real=0, padded=-inf
        key_mask = attention_mask[:, None, None, :].to(torch.float32)  # 1 or 0
        key_mask = (1.0 - key_mask) * torch.finfo(torch.float32).min
        # 2) causal mask: query at absolute position (past_len + q) may attend
        # to keys 0..past_len+q.
        q_abs = position_ids[0]  # (S,), identical across batch
        k_pos = torch.arange(total_len, device=inputs_embeds.device)
        causal = (k_pos[None, :] > q_abs[:, None]).to(torch.float32) * torch.finfo(torch.float32).min
        causal = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, S, total_S)
        causal_mask = key_mask + causal  # broadcast add → (B, 1, S, total_S)

        # Rotary frequencies for the NEW token positions.
        cos, sin = self.rotary(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        attn_captures: List[Optional[torch.Tensor]] = [None] * len(self.align_layers)
        presents: List[torch.Tensor] = []

        for i in range(self.num_layers):
            past_k = past_key_values[2 * i]
            past_v = past_key_values[2 * i + 1]
            hidden_states, attn_weights, present_k, present_v = self._layer_forward(
                i, hidden_states, cos, sin, causal_mask, past_k, past_v
            )
            presents.append(present_k)
            presents.append(present_v)

            if i in self.align_layers:
                idx = self.align_layers.index(i)
                # Conditional row only (row 0). attn_weights: (B, H, S, total_S)
                attn_captures[idx] = attn_weights[0]  # (H, S, total_S)

        hidden_states = self.final_norm(hidden_states)
        logits_full = self.speech_head(hidden_states)  # (2, S, V)

        cond = logits_full[0:1]
        uncond = logits_full[1:2]
        logits = cond + cfg_weight * (cond - uncond)

        attn_layers = torch.stack(attn_captures, dim=0)  # (3, H, S, total_S)
        return logits, attn_layers, *presents
