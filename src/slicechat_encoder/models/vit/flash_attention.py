import torch


flash_attn_func = None
flash_attn_varlen_func = None


if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    try:
        from einops import rearrange
        from flash_attn.bert_padding import pad_input, unpad_input
        from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_qkvpacked_func as _flash_attn_varlen_qkvpacked_func,
        )
    except ImportError:
        pass
    else:

        def flash_attn_varlen_func(
            q,
            k,
            v,
            dropout=0.0,
            bias=None,
            key_padding_mask=None,
            softmax_scale=None,
            is_causal=False,
        ):
            assert bias is None
            if key_padding_mask is None:
                raise ValueError("key_padding_mask is required for variable-length attention")

            padded_qkv = torch.stack([q, k, v], dim=2)
            batch_size, seqlen, _, nheads, _ = padded_qkv.shape

            qkv = rearrange(padded_qkv, "b s three h d -> b s (three h d)")
            qkv, indices, cu_seqlens, max_seqlen, _ = unpad_input(
                qkv, key_padding_mask
            )
            qkv = rearrange(
                qkv,
                "nnz (three h d) -> nnz three h d",
                three=3,
                h=nheads,
            )

            output, softmax_lse, _ = _flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                dropout,
                softmax_scale=softmax_scale,
                causal=is_causal,
                return_attn_probs=True,
            )

            output = rearrange(output, "nnz h d -> nnz (h d)")
            output = pad_input(output, indices, batch_size, seqlen)
            output = rearrange(output, "b s (h d) -> b s h d", h=nheads)

            softmax_lse = rearrange(softmax_lse, "h nnz -> nnz h")
            softmax_lse = pad_input(softmax_lse, indices, batch_size, seqlen)
            softmax_lse = rearrange(softmax_lse, "b s h -> b h s")

            return output, softmax_lse

        def flash_attn_func(
            q,
            k,
            v,
            dropout=0.0,
            bias=None,
            softmax_scale=None,
            is_causal=False,
        ):
            assert bias is None
            output, softmax_lse, _ = _flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=is_causal,
                return_attn_probs=True,
            )
            return output, softmax_lse
