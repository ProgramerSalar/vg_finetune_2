import torch 
from flash_attn.bert_padding import index_first_axis, pad_input
from flash_attn.flash_attn_interface import flash_attn_varlen_func
import torch.nn.functional as F 
from einops import rearrange

from flash_attention.var_len_self_attn_with_mask import apply_rope



class VarlenFlashSelfAttnSingle:

    def __init__(self):
        pass

    def __call__(
            self, query, key, value, heads, scale, 
            hidden_length=None, image_rotary_emb=None, encoder_attention_mask=None,
        ):
        assert encoder_attention_mask is not None, "The encoder-hidden mask needed to be set"

        batch_size = query.shape[0]
        output_hidden = torch.zeros_like(query)

        qkv_list = []
        num_stages = len(hidden_length)        
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        i_sum = 0
        for i_p, length in enumerate(hidden_length):
            qkv_tokens = qkv[:, i_sum:i_sum+length]

            if image_rotary_emb is not None:
                qkv_tokens[:,:,0], qkv_tokens[:,:,1] = apply_rope(qkv_tokens[:,:,0], qkv_tokens[:,:,1], image_rotary_emb[i_p])

            indices = encoder_attention_mask[i_p]['indices']
            qkv_list.append(index_first_axis(rearrange(qkv_tokens, "b s ... -> (b s) ..."), indices))
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
            stage_output = pad_input(stage_output, encoder_attention_mask[i_p]['indices'], batch_size, length)
            output_hidden[:, i_sum:i_sum+length] = stage_output
            token_sum += tot_token_num
            i_sum += length

        output_hidden = output_hidden.flatten(2, 3)

        return output_hidden