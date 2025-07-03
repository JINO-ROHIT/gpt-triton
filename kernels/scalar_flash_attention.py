import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(q_ptr, k_ptr, v_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    # constant used to compute e^x as exp2(x * log2(e))
    LOG2_E = 1.44269504
    # Outer loop: loop over query blocks
    for q_start in tl.static_range(0, N, BLOCK_SIZE):
        q_idx = q_start + tl.arange(0, BLOCK_SIZE)
        q_mask = q_idx < N
        q_block = tl.load(q_ptr + q_idx, mask=q_mask)
        row_max = (-1e10) + tl.zeros([BLOCK_SIZE], dtype=tl.float32) 
        row_sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)       
        output_block = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  
        # Inner loop: loop over key/value blocks
        for k_start in tl.static_range(0, N, BLOCK_SIZE):
            k_idx = k_start + tl.arange(0, BLOCK_SIZE)
            k_mask = k_idx < N
            k_block = tl.load(k_ptr + k_idx, mask=k_mask)
            # Compute attention scores (dot product): shape (BLOCK_SIZE, BLOCK_SIZE)
            attn_scores = q_block[:, None] * k_block[None, :]
            attn_scores = tl.where(k_mask[None, :], attn_scores, -1e10)
            new_row_max = tl.maximum(row_max, tl.max(attn_scores, axis=1))
            exp_scale = tl.exp2((row_max - new_row_max) * LOG2_E)
            exp_scores = tl.exp2((attn_scores - new_row_max[:, None]) * LOG2_E)
            new_row_sum_exp = row_sum_exp * exp_scale + tl.sum(exp_scores, axis=1)
            attn_probs = exp_scores / new_row_sum_exp[:, None]
            v_block = tl.load(v_ptr + k_idx, mask=k_mask)
            output_block = output_block * (exp_scale * row_sum_exp / new_row_sum_exp)
            output_block += tl.sum(attn_probs * v_block[None, :], axis=1)
            row_max = new_row_max
            row_sum_exp = new_row_sum_exp
        tl.store(z_ptr + q_idx, output_block, mask=q_mask)

def flash_attention(q, k, v, block_size = 64):
    q = q.cuda().contiguous()
    k = k.cuda().contiguous() 
    v = v.cuda().contiguous()
    
    seq_len = q.shape[0]
    
    output = torch.empty_like(q)
    
    grid = (1,)
    
    flash_attention_kernel[grid](
        q_ptr=q,
        k_ptr=k, 
        v_ptr=v,
        z_ptr=output,
        N=seq_len,
        BLOCK_SIZE=block_size
    )
    
    return output


# if __name__ == "__main__":
#     seq_len = 256
#     torch.manual_seed(42)
    
#     q = torch.randn(seq_len, dtype=torch.float32, device='cuda')
#     k = torch.randn(seq_len, dtype=torch.float32, device='cuda') 
#     v = torch.randn(seq_len, dtype=torch.float32, device='cuda')
    
#     print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    
#     block_size = 64
#     output = flash_attention(q, k, v, block_size=block_size)
    
#     print(f"Output shape: {output.shape}")
#     print(f"Output sample: {output[:5]}")
    
#     # Verify against PyTorch implementation
#     def naive_attention(q, k, v):
#         scores = torch.outer(q, k)  # [seq_len, seq_len]
#         attn_weights = torch.softmax(scores, dim=1)
#         return torch.sum(attn_weights * v[None, :], dim=1)

#     naive_output = naive_attention(q, k, v)
    
#     print(f"\nComparison:")
#     print(f"Flash attention output[:5]: {output[:5]}")
#     print(f"Naive attention output[:5]: {naive_output[:5]}")
#     print(f"Max difference: {torch.max(torch.abs(output - naive_output))}")
#     print(f"Are results close? {torch.allclose(output, naive_output, atol=1e-3)}")
