import torch
from torch import nn 
from typing import List
from torch.nn import functional as F 

from block.feed_forward import FeedForward
from norm import AdaLayerNormZero
from flash_attention.flux_attention_processor_2 import FluxAttnProcessor2_0
from attention.attention import Attention



class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6, use_flash_attn=False):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0(use_flash_attn)
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor,
        temb: torch.FloatTensor,
        attention_mask: torch.FloatTensor = None,
        hidden_length: List = None,
        image_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb, hidden_length=hidden_length)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask, 
            attention_mask=attention_mask,
            hidden_length=hidden_length,
            image_rotary_emb=image_rotary_emb,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states