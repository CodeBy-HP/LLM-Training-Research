# nanoGPT (Tiny Shakespeare)

## Purpose
This project is an exploration into building and training a Generative Pre-trained Transformer (GPT) from scratch. The goal was to deeply understand the inner workings of Large Language Models by implementing a character-level GPT to generate Shakespearean text, heavily inspired by Andrej Karpathy's codebase.

## Model Configuration
- **Parameters**: ~10.7M
- **Vocabulary Size**: 65 (Character-level)
- **Context Window**: 256 tokens
- **Layers**: 6
- **Heads**: 6
- **Embedding Dimension**: 384

## Training Configuration
- **Dataset**: Tiny Shakespeare (~1MB)
- **Iterations**: 5,000 steps
- **Batch Size**: 64
- **Optimizer**: Fused AdamW
- **Learning Rate**: 1e-3 (Cosine Decay with Warmup)
- **Hardware**: Trained on Google Colab (Tesla T4 GPU)

## Optimization Tricks Implemented
To train efficiently on a free T4 GPU, several modern PyTorch optimizations were integrated:
- **PyTorch 2.0 `torch.compile`**: JIT-compiles the model into optimized C++/CUDA kernels.
- **Mixed Precision (FP16)**: Uses `float16` and a `GradScaler` to accelerate matrix multiplications via Tensor Cores.
- **Flash Attention**: Uses `F.scaled_dot_product_attention` for highly optimized, memory-efficient attention computation.
- **Fused AdamW**: Executes the optimizer step in a single fused CUDA kernel to reduce overhead.
- **Gradient Accumulation**: Decouples forward/backward passes from the optimizer step to simulate larger batch sizes cleanly.
- **Weight Tying**: Shares memory between the input embedding layer and the final output projection layer.
- **Weight Decay Separation**: Applies weight decay strictly to 2D matmuls/embeddings, exempting 1D biases and LayerNorms.

## Pre-trained Model
The final trained weights are published in the modern `safetensors` format. You can download and run the model directly from Hugging Face:
🔗 **[codeby-hp/nanogpt-shakespeare](https://huggingface.co/codeby-hp/nanogpt-shakespeare)**