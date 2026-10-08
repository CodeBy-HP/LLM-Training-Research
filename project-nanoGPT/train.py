import os
import math
import urllib.request
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# Hardware and Training Variables
batch_size = 64                   # Micro-batch size per forward pass
gradient_accumulation_steps = 1   # Simulate larger batch sizes (e.g. 4 or 8) by accumulating gradients
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile_model = True if device == 'cuda' else False  # Accelerate training with PyTorch 2.0

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"Tokens per optimizer iteration: {tokens_per_iter:,}")

# Mixed Precision and Optimization Safeguards
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
use_amp = (device == 'cuda')  # Enable AMP on CUDA
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
grad_clip = 1.0               # Maximum gradient norm for clipping
ctx = torch.amp.autocast(device_type='cuda', dtype=dtype) if use_amp else nullcontext()

# Checkpoint and Google Drive Persistence Settings
init_from = 'scratch'      # 'scratch' or 'resume'
max_checkpoints = 3        # Rolling buffer: keep only the 3 most recent best checkpoints
mount_google_drive = True  # Automatically mount Google Drive if running in Google Colab

# Weights & Biases (WandB) Settings
wandb_log = True                   # Set to True to log to WandB, False to disable
wandb_project = 'nanogpt-training'
wandb_run_name = 'nanoGPT-run'

# Auto-detect Google Colab and set up checkpoint directory
try:
    import google.colab
    in_colab = True
except ImportError:
    in_colab = False

if in_colab and mount_google_drive:
    from google.colab import drive
    drive_mount_point = '/content/drive'
    if not os.path.ismount(drive_mount_point):
        print("Mounting Google Drive to /content/drive ...")
        drive.mount(drive_mount_point)
    out_dir = '/content/drive/MyDrive/nanogpt_checkpoints'
else:
    out_dir = 'checkpoints'

os.makedirs(out_dir, exist_ok=True)
print(f"Checkpoint directory: {out_dir}")

# Download the Dataset
data_path = 'input.txt'
if not os.path.exists(data_path):
    print("Downloading Tiny Shakespeare...")
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(url, data_path)

# Tokenizer Setup
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# RAM Allocation (1MB file fits easily in standard memory)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

weight_decay = 1e-1
betas = (0.9, 0.95)

# Build the Brain
config = GPTConfig(vocab_size=vocab_size, block_size=block_size)
model = GPT(config).to(device)
optimizer = model.configure_optimizers(weight_decay, learning_rate, betas, device_type='cuda' if 'cuda' in device else 'cpu')

# Initialize GradScaler (only needed for float16, no-op for bfloat16)
try:
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and dtype == torch.float16))
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and dtype == torch.float16))

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# Helper Functions for Rolling Checkpoints
def get_latest_checkpoint(checkpoint_dir):
    """Finds the most recent checkpoint file based on iteration number."""
    ckpts = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith("ckpt_iter_") and f.endswith(".pt")
    ]
    if not ckpts:
        return None
    ckpts.sort()  # Zero-padded step numbers ensure chronological order
    return ckpts[-1]

def save_rolling_checkpoint(checkpoint_dict, checkpoint_dir, iter_num, val_loss, max_keep=3):
    """Saves full checkpoint and maintains a rolling buffer of max_keep best checkpoints."""
    filename = f"ckpt_iter_{iter_num:05d}_val_{val_loss:.4f}.pt"
    ckpt_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint_dict, ckpt_path)
    print(f"Saved full checkpoint: {ckpt_path}")

    # Prune older checkpoints to preserve storage (e.g. on Google Drive)
    ckpts = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith("ckpt_iter_") and f.endswith(".pt")
    ]
    ckpts.sort()
    while len(ckpts) > max_keep:
        oldest = ckpts.pop(0)
        try:
            os.remove(oldest)
            print(f"Removed older checkpoint to preserve storage: {oldest}")
        except OSError as e:
            print(f"Warning: could not delete {oldest}: {e}")

start_iter = 0
best_val_loss = float('inf')
wandb_run_id = None

# Resume training if requested
if init_from == 'resume':
    ckpt_path = get_latest_checkpoint(out_dir)
    if ckpt_path is not None:
        print(f"Resuming training from checkpoint: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iter_num'] + 1
        best_val_loss = checkpoint['best_val_loss']
        wandb_run_id = checkpoint.get('wandb_run_id', None)
        if 'scaler' in checkpoint and checkpoint['scaler'] is not None and use_amp:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f"Successfully resumed at iteration {start_iter} with best_val_loss: {best_val_loss:.4f}")
    else:
        print(f"No checkpoint found in {out_dir}. Training starting from scratch.")

# Compile the model
if compile_model:
    print("Compiling the model... (this takes about 1-3 minutes)")
    model = torch.compile(model)

# Initialize Weights & Biases (WandB)
if wandb_log:
    import wandb
    if wandb_run_id is None:
        wandb_run_id = wandb.util.generate_id()
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        id=wandb_run_id,
        resume="allow",  # Resumes appending to the exact same experiment if wandb_run_id is preserved!
        config={
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "dropout": config.dropout,
            "bias": config.bias,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "tokens_per_iter": tokens_per_iter,
            "max_iters": max_iters,
        }
    )
    print(f"WandB run initialized with ID: {wandb_run_id} (resume='allow')")

# Execute Training
print(f"Starting training on {device} (steps {start_iter} to {max_iters})...")

for iter in range(start_iter, max_iters):
    # determine and set the learning rate for this iteration
    lr = get_lr(iter) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if wandb_log:
            wandb.log({
                "iter": iter,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            }, step=iter)

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if use_amp else None,
                'model_args': {
                    'vocab_size': config.vocab_size,
                    'block_size': config.block_size,
                    'n_layer': config.n_layer,
                    'n_head': config.n_head,
                    'n_embd': config.n_embd,
                    'dropout': config.dropout,
                    'bias': config.bias,
                },
                'iter_num': iter,
                'best_val_loss': best_val_loss,
                'wandb_run_id': wandb_run_id,
            }
            save_rolling_checkpoint(checkpoint, out_dir, iter, best_val_loss, max_keep=max_checkpoints)

    optimizer.zero_grad(set_to_none=True)
    
    # Forward-backward accumulation loop
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch('train')
        
        # Forward pass in float16 mixed precision
        with ctx:
            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps  # Scale down loss so sum of grads equals the true average
        
        # Backward pass with scaled loss
        scaler.scale(loss).backward()
    
    # Gradient clipping (unscale gradients before clipping)
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
    # Optimizer step via scaler
    scaler.step(optimizer)
    scaler.update()

# Prove Intelligence
print("\n--- Training Complete. Generating Sample ---")
raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
best_ckpt = get_latest_checkpoint(out_dir)
if best_ckpt is not None:
    print(f"Loading best weights from {best_ckpt} for text generation...")
    try:
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_ckpt, map_location=device)
    raw_model.load_state_dict(checkpoint['model'])

raw_model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = raw_model.generate(context, max_new_tokens=500, temperature=0.8, top_k=20)
print(decode(generated[0].tolist()))

if wandb_log:
    wandb.finish()