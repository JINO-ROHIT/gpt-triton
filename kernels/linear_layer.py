import torch
import triton
import triton.language as tl


@triton.jit
def triton_linear(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """C = A @ B.T + bias"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        mask_b = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask = mask_a, other = 0.0)
        b = tl.load(b_ptrs, mask = mask_b, other = 0.0)
        
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    if bias_ptr is not None:
        bias_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        bias_mask = bias_offs < N
        bias = tl.load(bias_ptr + bias_offs, mask=bias_mask, other=0.0)
        accumulator = accumulator + bias[None, :]
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    c = accumulator.to(tl.float16)
    tl.store(c_ptrs, c, mask=c_mask)


def linear_layer(input_tensor, weight, bias = None, 
                 block_size_m = 16, block_size_n = 16, block_size_k = 16):
    original_shape = input_tensor.shape
    input_2d = input_tensor.view(-1, original_shape[-1])
    M, K = input_2d.shape
    N, K_weight = weight.shape
    
    assert K == K_weight, f"Dimension mismatch: input {K} vs weight {K_weight}"
    
    output = torch.empty((M, N), device=input_tensor.device, dtype=input_tensor.dtype)
    
    grid = (triton.cdiv(M, block_size_m), triton.cdiv(N, block_size_n))
    
    triton_linear[grid](
        input_2d, weight, output, bias,
        M, N, K,
        input_2d.stride(0), input_2d.stride(1),
        weight.stride(1), weight.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M = block_size_m,
        BLOCK_SIZE_N = block_size_n,
        BLOCK_SIZE_K = block_size_k,
    )
    
    output_shape = list(original_shape[:-1]) + [N]
    return output.view(output_shape)


# torch version --> later on check 
def pytorch_linear(input_tensor, weight, bias=None):
    return torch.nn.functional.linear(input_tensor, weight, bias)