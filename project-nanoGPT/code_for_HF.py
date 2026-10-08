import os
import sys
import json
import shutil
import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo

repo_path = "/content/LLM-Training-Research/project-nanoGPT"
if os.path.exists(repo_path):
    os.chdir(repo_path)
    if repo_path not in sys.path:
        sys.path.append(repo_path)
else:
    print(f"Warning: {repo_path} not found. Ensure you cloned the repository!")
# -----------------------------------------------------------------------------------

# 1. Define your Hugging Face repo details
REPO_NAME = "nanogpt-shakespeare"  
HF_USERNAME = "codeby-hp"        
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

# 2. Path to your BEST checkpoint on Google Drive
ckpt_path = "/content/drive/MyDrive/nanogpt_checkpoints/ckpt_iter_01500_val_1.4646.pt"
print(f"Loading checkpoint from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# 3. Create a temporary export folder
export_dir = "/content/hf_export"
os.makedirs(export_dir, exist_ok=True)

# 4. Save clean weights securely as SAFETENSORS
from model import GPTConfig, GPT
from safetensors.torch import save_model

model_args = checkpoint["model_args"]
config = GPTConfig(**model_args)
model = GPT(config)
model.load_state_dict(checkpoint["model"])

model_weights_path = os.path.join(export_dir, "model.safetensors")
save_model(model, model_weights_path) 

# 5. Save the Architecture Config
config_path = os.path.join(export_dir, "config.json")
with open(config_path, "w") as f:
    json.dump(checkpoint["model_args"], f, indent=2)

# 6. Save the Vocabulary Mapping
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab = {
    "chars": chars,
    "stoi": {ch: i for i, ch in enumerate(chars)},
    "itos": {i: ch for i, ch in enumerate(chars)},
    "vocab_size": len(chars)
}
with open(os.path.join(export_dir, "vocab.json"), "w") as f:
    json.dump(vocab, f, indent=2)

# 7. Copy model.py so others have the code to load it
shutil.copy("model.py", os.path.join(export_dir, "model.py"))

# 8. Create a professional README.md
readme_content = f"""---
license: mit
tags:
- gpt
- nanogpt
- text-generation
- pytorch
- safetensors
---
# nanoGPT (Tiny Shakespeare)
A 10.7M parameter character-level GPT trained from scratch on the Tiny Shakespeare dataset.

## Training Details
- **Architecture**: 6 layers, 6 heads, 384 embedding dimensions
- **Context Length**: 256 tokens
- **Training Steps**: 5,000 iterations
- **Best Validation Loss**: 1.4646 (reached at step 1500)
- **Precision**: Mixed Precision (FP16) on Tesla T4 with PyTorch 2.0 compile

## How to Use
```python
import os
import sys
import json
import torch
from safetensors.torch import load_model 
from huggingface_hub import hf_hub_download

# 1. Download model files
repo_id = "{REPO_ID}"
weights_file = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
config_file = hf_hub_download(repo_id=repo_id, filename="config.json")
vocab_file = hf_hub_download(repo_id=repo_id, filename="vocab.json")

# Download model.py and add its folder to sys.path so we can import it!
model_code_file = hf_hub_download(repo_id=repo_id, filename="model.py")
sys.path.append(os.path.dirname(model_code_file))
from model import GPTConfig, GPT

# 2. Load config and vocabulary
with open(config_file, "r") as f:
    cfg_dict = json.load(f)
with open(vocab_file, "r") as f:
    vocab = json.load(f)

itos = {{int(k): v for k, v in vocab["itos"].items()}}
decode = lambda l: "".join([itos[i] for i in l])

# 3. Instantiate model
config = GPTConfig(**cfg_dict)
model = GPT(config)
load_model(model, weights_file) 
model.eval()

# 4. Generate text
context = torch.zeros((1, 1), dtype=torch.long)
out = model.generate(context, max_new_tokens=300, temperature=0.8, top_k=20)
print(decode(out[0].tolist()))
```
""" 

with open(os.path.join(export_dir, "README.md"), "w") as f: 
    f.write(readme_content)

# 9. Create Repo & Upload
api = HfApi()
print(f"Creating repo {REPO_ID} on Hugging Face...")
create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

print("Uploading files to Hugging Face...")
api.upload_folder(
    folder_path=export_dir,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Initial release: trained nanoGPT in safetensors format"
)

print(f"✅ Successfully published! View it here: https://huggingface.co/{REPO_ID}")
