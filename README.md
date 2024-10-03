
# GPT-2 Training Script

This repository contains a PyTorch implementation of GPT-2 model training, fine-tuning, and evaluation using distributed data parallelism (DDP) with optional support for multi-GPU setups.

## Requirements

To run this script, you'll need the following libraries:

- Python 3.x
- PyTorch
- Hugging Face Transformers
- NumPy
- Tiktoken (for tokenization)
- HellaSwag (optional dataset for evaluation)

Install the dependencies using:

```bash
pip install torch transformers numpy tiktoken
```

## Key Features

- **GPT-2 Architecture**: The script implements the GPT-2 architecture, with support for different GPT-2 variants (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`).
- **Self-Attention**: Uses scaled dot-product attention for faster computations, especially on larger models.
- **Distributed Training**: Supports distributed training using PyTorch's DistributedDataParallel (DDP) with multi-GPU capabilities.
- **Gradient Accumulation**: Uses gradient accumulation to handle large batch sizes.
- **Learning Rate Scheduler**: Implements cosine decay learning rate schedule with warmup.
- **Validation and HellaSwag Evaluation**: Supports validation using cross-entropy loss and optional evaluation on the HellaSwag dataset.

## Model Architecture

The architecture is based on the GPT-2 model with self-attention blocks. It includes the following components:
- **Self-Attention Layers**: Multi-head attention mechanism.
- **MLP Layers**: Feedforward neural network layers with GELU activation.
- **Layer Normalization**: Applied after attention and MLP layers.
- **Positional and Token Embeddings**: Handles sequence inputs and outputs.

## Training

To train the model, run the script with the following command:

```bash
python train-gpt2.py
```

For distributed training with multiple GPUs, use the following command:

```bash
torchrun --standalone --nproc_per_node=<NUM_GPUS> train-gpt2.py
```

### Key Training Parameters:
- `block_size`: Maximum length of the input sequence (default: 1024).
- `n_emb`: Dimensionality of token embeddings (default: 768).
- `vocab_size`: Vocabulary size (default: 50,257).
- `n_head`: Number of attention heads (default: 12).
- `n_layers`: Number of Transformer layers (default: 12).
- `max_steps`: Number of training steps (default: 19,073).
- `warmup_step`: Steps for learning rate warmup (default: 715).
- `B`: Batch size (default: 4).
- `T`: Sequence length (default: 1024).
- `total_batch_size`: Total batch size (default: 32,768).
- `learning_rate`: Initial learning rate (default: 6e-4).

### Optimizer:
The script uses AdamW optimizer with the following configuration:
- Weight decay: 0.1
- Betas: (0.9, 0.95)
- Epsilon: 1e-8

## Evaluation

The script evaluates the model on the validation set every 250 steps and also supports evaluation on the HellaSwag dataset. To enable HellaSwag evaluation, download the dataset and update the paths in the script.

### Validation:
- Validation is performed using cross-entropy loss on the validation split of the dataset.
- Logs validation loss every 250 steps.

### HellaSwag Evaluation:
- The script computes accuracy on the HellaSwag dataset every 250 steps.
- Reports accuracy and logs the results.

## Inference

The script also supports text generation using the trained GPT-2 model. You can generate text by running the inference step in the script, which outputs generated sequences based on a given prompt.

To generate text:

1. Adjust the `tokens` variable in the generation block with your input prompt.
2. The script will generate multiple sequences and print them to the console.

## Checkpoints

Checkpoints are saved every 5,000 steps and at the end of training. The checkpoint includes:
- Model state dictionary
- Training step
- Validation loss

## Logging

The script logs training and validation losses to `log/log.txt`. It logs:
- Training loss every step
- Validation loss every 250 steps
- HellaSwag accuracy (if enabled)

## Distributed Training

### Multi-GPU Training

This script supports multi-GPU training using PyTorch's Distributed Data Parallel (DDP). Use the `torchrun` command to launch the script across multiple GPUs. The script automatically detects available GPUs and sets the device accordingly.

- Set `device` and `ddp` flags in the script to enable DDP.
- Use `torchrun` for launching the training process.

### Device Compatibility

The script is compatible with:
- **CUDA**: Multi-GPU support for distributed training.
- **MPS**: For Apple Silicon devices.

## License

This project is open-source under the MIT License.
