import torch 
import torch.nn.functional as F 

from flash_attention.var_len_self_attn_with_mask import apply_rope


class VarlenSelfAttnSingle:

    def __init__(self):
        pass

    def __call__(
            self, query, key, value, heads, scale, 
            hidden_length=None, image_rotary_emb=None, attention_mask=None,
        ):
        assert attention_mask is not None, "The attention mask needed to be set"

        num_stages = len(hidden_length)        
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        i_sum = 0
        output_hidden_list = []
    
        for i_p, length in enumerate(hidden_length):
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            
            if image_rotary_emb is not None:
                qkv_tokens[:,:,0], qkv_tokens[:,:,1] = apply_rope(qkv_tokens[:,:,0], qkv_tokens[:,:,1], image_rotary_emb[i_p])

            query, key, value = qkv_tokens.unbind(2)
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()

            stage_hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask[i_p],
            )
            stage_hidden_states = stage_hidden_states.transpose(1, 2).flatten(2, 3)   # [bs, tot_seq, dim]

            output_hidden_list.append(stage_hidden_states)
            i_sum += length

        output_hidden = torch.cat(output_hidden_list, dim=1)

        return output_hidden