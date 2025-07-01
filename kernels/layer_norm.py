import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    input_ptr, output_ptr, gamma_ptr, beta_ptr, mean_ptr, var_ptr, eps,
    num_rows: tl.constexpr, num_cols: tl.constexpr,
    stride_row: tl.constexpr, stride_col: tl.constexpr,
    block_size: tl.constexpr
):
    
    pid = tl.program_id(axis=0)
    row_start = pid * block_size
    row_offsets = row_start + tl.arange(0, block_size)
    col_offsets = tl.arange(0, num_cols)

    mask = (row_offsets[:, None] < num_rows) & (col_offsets[None, :] < num_cols)

    input_ptrs = input_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
    gamma_ptrs = gamma_ptr + col_offsets[None, :]
    beta_ptrs = beta_ptr + col_offsets[None, :]


    x = tl.load(input_ptrs, mask = mask, other=0.0)
    gamma = tl.load(gamma_ptrs)
    beta = tl.load(beta_ptrs)

    # mean across columns for each row
    mean = tl.sum(x, axis=1, keep_dims=True) / num_cols
    mean = mean.to(tl.float32)

    # variance across columns for each row
    squared_diff = (x - mean) * (x - mean)
    var = tl.sum(squared_diff, axis=1, keep_dims=True) / num_cols
    var = var.to(tl.float32)

    # (x - mean) / sqrt(var + eps)
    x_normalized = (x - mean) / tl.sqrt(var + eps)

    # scale (gamma) and shift (beta)
    output = x_normalized * gamma + beta

    output_ptrs = output_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
    tl.store(output_ptrs, output, mask=mask)
    tl.store(mean_ptr + row_offsets, tl.reshape(mean, [block_size]), mask=row_offsets < num_rows)
    tl.store(var_ptr + row_offsets, tl.reshape(var, [block_size]), mask=row_offsets < num_rows)

def layer_norm(input_tensor, gamma, beta, eps):
    assert input_tensor.is_contiguous(), "tensor must be contiguous"

    num_rows, num_cols = input_tensor.shape

    normalized_tensor = torch.empty_like(input_tensor, device = input_tensor.device)
    mean = torch.empty((num_rows, 1), device = input_tensor.device, dtype  = input_tensor.dtype)
    var = torch.empty_like(mean, device = input_tensor.device, dtype = input_tensor.dtype)

    grid = lambda meta: (triton.cdiv(num_rows, meta['block_size']),)

    layer_norm_kernel[grid](
        input_tensor, normalized_tensor, gamma, beta, mean, var, eps,
        num_rows, num_cols,
        input_tensor.stride(0), input_tensor.stride(1),
        block_size = 1024
    )

    return normalized_tensor




### sample usage --> use gamma beta from nn
'''
layer_norm = torch.nn.LayerNorm(input.shape[1], eps = 1e-5, device = 'cuda')
input_normalized = layer_norm(input)
input_norm = layer_norm_forward(input, layer_norm.weight, layer_norm.bias, layer_norm.eps)
'''