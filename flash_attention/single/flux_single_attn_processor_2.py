
import torch 
from typing import Optional, List

from attention.attention import Attention
from flash_attention.single.var_len_flash_self_attn_single import VarlenFlashSelfAttnSingle
from flash_attention.single.var_len_self_attn_single import VarlenSelfAttnSingle



class FluxSingleAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """
    def __init__(self, use_flash_attn=False):
        self.use_flash_attn = use_flash_attn

        if self.use_flash_attn:
            
            self.varlen_flash_attn = VarlenFlashSelfAttnSingle()
        else:
            self.varlen_attn = VarlenSelfAttnSingle()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        hidden_length: List = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

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

        if self.use_flash_attn:
            hidden_states = self.varlen_flash_attn(
                query, key, value, 
                attn.heads, attn.scale, hidden_length, 
                image_rotary_emb, encoder_attention_mask,
            )
        else:
            hidden_states = self.varlen_attn(
                query, key, value, 
                attn.heads, attn.scale, hidden_length, 
                image_rotary_emb, attention_mask,
            )

        return hidden_states