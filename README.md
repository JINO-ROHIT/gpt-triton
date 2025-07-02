# gpt-triton

This repository is an experimental worklog to implementing the gpt architecture using triton. 

## GPT-2 LAYER SKELETON

1. Embedding layer
2. Dropout
3. Tranformer Block ( x 12 )
   - layer norm
   - MHA
   - layer norm
   - linear

4. Layer Norm
5. Linear Layer


## Implemented Layers

1. Embedding layer - [x]
2. Dropout - [x]
3. Transfomer Block - []
4. Layer Norm - [x]
5. Linear Layer - []


## Benchmark Results (WIP)

### Custom Triton Fused Embedder vs Pytorch Embedder layer

| Batch Size | Sequence Length | Hidden Dim | PyTorch (ms) | Triton (ms) | Speedup (x) |
|------------|------------------|-------------|---------------|--------------|--------------|
| 32         | 128              | 1024        | 0.09          | 0.06         | 1.50         |
| 32         | 512              | 768         | 0.27          | 0.21         | 1.31         |
| 32         | 512              | 1024        | 0.36          | 0.21         | 1.73         |
| 32         | 1024             | 768         | 0.43          | 0.22         | 1.98         |
| 32         | 1024             | 1024        | 0.56          | 0.21         | 2.66         |
| 64         | 128              | 768         | 0.11          | 0.06         | 1.64         |
| 64         | 128              | 1024        | 0.14          | 0.07         | 2.10         |
| 64         | 512              | 768         | 0.42          | 0.21         | 2.00         |
| 64         | 512              | 1024        | 0.56          | 0.21         | 2.66         |
| 64         | 1024             | 768         | 0.85          | 0.41         | 2.09         |
| 64         | 1024             | 1024        | 1.13          | 0.41         | 2.76         |
| 128        | 128              | 768         | 0.21          | 0.12         | 1.80         |
| 128        | 128              | 1024        | 0.28          | 0.12         | 2.35         |
| 128        | 512              | 768         | 0.83          | 0.40         | 2.06         |
| 128        | 512              | 1024        | 1.12          | 0.39         | 2.89         |
| 128        | 1024             | 768         | 1.68          | 0.81         | 2.07         |
| 128        | 1024             | 1024        | 2.25          | 0.83         | 2.72         |
