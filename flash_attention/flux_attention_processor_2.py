import torch
from typing import Optional, List

from attention.attention import Attention
from flash_attention.variable_flash_attn_with_t5_mask import VarlenFlashSelfAttentionWithT5Mask
from flash_attention.var_len_self_attn_with_mask import VarlenSelfAttentionWithT5Mask

class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, use_flash_attn=False):
        self.use_flash_attn = use_flash_attn

        if self.use_flash_attn:
            
            self.varlen_flash_attn = VarlenFlashSelfAttentionWithT5Mask()
        else:
            
            self.varlen_attn = VarlenSelfAttentionWithT5Mask()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        hidden_length: List = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(query.shape[0], -1, attn.heads, head_dim)
        key = key.view(key.shape[0], -1, attn.heads, head_dim)
        value = value.view(value.shape[0], -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            encoder_hidden_states_query_proj.shape[0], -1, attn.heads, head_dim
        )
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            encoder_hidden_states_key_proj.shape[0], -1, attn.heads, head_dim
        )
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            encoder_hidden_states_value_proj.shape[0], -1, attn.heads, head_dim
        )

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        if self.use_flash_attn:
            hidden_states, encoder_hidden_states = self.varlen_flash_attn(
                query, key, value, 
                encoder_hidden_states_query_proj, encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj, attn.heads, attn.scale, hidden_length, 
                image_rotary_emb, encoder_attention_mask,
            )
        else:
            hidden_states, encoder_hidden_states = self.varlen_attn(
                query, key, value, 
                encoder_hidden_states_query_proj, encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj, attn.heads, attn.scale, hidden_length, 
                image_rotary_emb, attention_mask,
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states