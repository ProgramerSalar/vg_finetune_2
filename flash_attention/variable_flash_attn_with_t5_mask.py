import torch 

from flash_attn.bert_padding import pad_input, index_first_axis
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from einops import rearrange
from torch.nn import functional as F

def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)



class VarlenFlashSelfAttentionWithT5Mask:

    def __init__(self):
        pass

    def __call__(
            self, query, key, value, encoder_query, encoder_key, encoder_value, 
            heads, scale, hidden_length=None, image_rotary_emb=None, encoder_attention_mask=None,
        ):
        assert encoder_attention_mask is not None, "The encoder-hidden mask needed to be set"

        batch_size = query.shape[0]
        output_hidden = torch.zeros_like(query)
        output_encoder_hidden = torch.zeros_like(encoder_query)
        encoder_length = encoder_query.shape[1]

        qkv_list = []
        num_stages = len(hidden_length)        
    
        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2) # [bs, sub_seq, 3, head, head_dim]
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        i_sum = 0
        for i_p, length in enumerate(hidden_length):
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)  # [bs, tot_seq, 3, nhead, dim]
            
            if image_rotary_emb is not None:
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

            indices = encoder_attention_mask[i_p]['indices']
            qkv_list.append(index_first_axis(rearrange(concat_qkv_tokens, "b s ... -> (b s) ..."), indices))
            i_sum += length

        token_lengths = [x_.shape[0] for x_ in qkv_list]
        qkv = torch.cat(qkv_list, dim=0)
        query, key, value = qkv.unbind(1)

        cu_seqlens = torch.cat([x_['seqlens_in_batch'] for x_ in encoder_attention_mask], dim=0)
        max_seqlen_q = cu_seqlens.max().item()
        max_seqlen_k = max_seqlen_q
        cu_seqlens_q = F.pad(torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32), (1, 0))
        cu_seqlens_k = cu_seqlens_q.clone()

        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
        )

        # To merge the tokens
        i_sum = 0;token_sum = 0
        for i_p, length in enumerate(hidden_length):
            tot_token_num = token_lengths[i_p]
            stage_output = output[token_sum : token_sum + tot_token_num]
            stage_output = pad_input(stage_output, encoder_attention_mask[i_p]['indices'], batch_size, encoder_length + length)
            stage_encoder_hidden_output = stage_output[:, :encoder_length]
            stage_hidden_output = stage_output[:, encoder_length:]   
            output_hidden[:, i_sum:i_sum+length] = stage_hidden_output
            output_encoder_hidden[i_p::num_stages] = stage_encoder_hidden_output
            token_sum += tot_token_num
            i_sum += length

        output_hidden = output_hidden.flatten(2, 3)
        output_encoder_hidden = output_encoder_hidden.flatten(2, 3)

        return output_hidden, output_encoder_hidden