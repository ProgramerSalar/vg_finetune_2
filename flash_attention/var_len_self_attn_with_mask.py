import torch 
from einops import rearrange
from torch.nn import functional as F 

from .variable_flash_attn_with_t5_mask import apply_rope


class VarlenSelfAttentionWithT5Mask:

    def __init__(self):
        pass

    def __call__(
            self, query, key, value, encoder_query, encoder_key, encoder_value, 
            heads, scale, hidden_length=None, image_rotary_emb=None, attention_mask=None,
        ):
        assert attention_mask is not None, "The attention mask needed to be set"

        encoder_length = encoder_query.shape[1]
        num_stages = len(hidden_length)        
    
        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2) # [bs, sub_seq, 3, head, head_dim]
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        i_sum = 0
        output_encoder_hidden_list = []
        output_hidden_list = []
    
        for i_p, length in enumerate(hidden_length):
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)  # [bs, tot_seq, 3, nhead, dim]
            
            if image_rotary_emb is not None:
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

            query, key, value = concat_qkv_tokens.unbind(2)   # [bs, tot_seq, nhead, dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            # with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
            stage_hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask[i_p],
            )
            stage_hidden_states = stage_hidden_states.transpose(1, 2).flatten(2, 3)   # [bs, tot_seq, dim]

            output_encoder_hidden_list.append(stage_hidden_states[:, :encoder_length])
            output_hidden_list.append(stage_hidden_states[:, encoder_length:])
            i_sum += length

        output_encoder_hidden = torch.stack(output_encoder_hidden_list, dim=1)  # [b n s d]
        output_encoder_hidden = rearrange(output_encoder_hidden, 'b n s d -> (b n) s d')
        output_hidden = torch.cat(output_hidden_list, dim=1)

        return output_hidden, output_encoder_hidden