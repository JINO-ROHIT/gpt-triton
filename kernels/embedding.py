import triton
import triton.language as tl
import torch

# a sample input
# batch_size = 2
# seq_len = 512
# hidden_dim = 768
# vocab_size = 50_257
# context_length = 1024

def fused_embedding(token_ids, position_ids, token_embedding_table, position_embedding_table):
    batch_size, seq_len = token_ids.shape
    hidden_dim = token_embedding_table.shape[1]
    
    token_ids = token_ids.contiguous()
    position_ids = position_ids.contiguous()
    
    output = torch.empty(
        (batch_size, seq_len, hidden_dim), 
        dtype = token_embedding_table.dtype,
        device = token_ids.device
    )
    
    total_tokens = batch_size * seq_len
    
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 1024)
    
    # 2d grid
    grid = (total_tokens, triton.cdiv(hidden_dim, BLOCK_SIZE))
    
    fused_embedding_kernel[grid](
        token_ids,
        position_ids,
        token_embedding_table,
        position_embedding_table,
        output,
        total_tokens,
        hidden_dim,
        token_embedding_table.shape[0],  # vocab_size
        position_embedding_table.shape[0],  # context_length
        seed = 123,
        p = 0.1,
        BLOCK_SIZE = BLOCK_SIZE
    )
    
    return output

@triton.jit
def fused_embedding_kernel(
    token_ids_ptr,
    position_ids_ptr,
    token_embedding_table_ptr,
    position_embedding_table_ptr,
    output_ptr,
    total_tokens,
    hidden_dim,
    vocab_size,
    context_length,
    seed,
    p,
    BLOCK_SIZE: tl.constexpr
):
    token_idx = tl.program_id(0)  # token
    block_idx = tl.program_id(1)  # block
    
    hidden_offset = block_idx * BLOCK_SIZE
    hidden_offsets = hidden_offset + tl.arange(0, BLOCK_SIZE)
    
    hidden_mask = hidden_offsets < hidden_dim
    
    if token_idx >= total_tokens:
        return
    
    token_id = tl.load(token_ids_ptr + token_idx)
    position_id = tl.load(position_ids_ptr + token_idx)
    
    token_valid = (token_id >= 0) & (token_id < vocab_size)
    position_valid = (position_id >= 0) & (position_id < context_length)
    
    token_embedding_offsets = token_id * hidden_dim + hidden_offsets
    position_embedding_offsets = position_id * hidden_dim + hidden_offsets
    
    token_embedding = tl.load(
        token_embedding_table_ptr + token_embedding_offsets,
        mask = hidden_mask & token_valid,
        other = 0.0
    )
    
    position_embedding = tl.load(
        position_embedding_table_ptr + position_embedding_offsets,
        mask = hidden_mask & position_valid,
        other = 0.0
    )
    
    fused_embedding = token_embedding + position_embedding

    # fused drop out (check correctness)
    random = tl.rand(seed, position_embedding_offsets)
    dropout_mask = random > p 
    fused_embedding = tl.where(dropout_mask , fused_embedding / (1 - p), 0.0)
    
    output_offsets = token_idx * hidden_dim + hidden_offsets
    
    tl.store(
        output_ptr + output_offsets,
        fused_embedding,
        mask = hidden_mask
    )