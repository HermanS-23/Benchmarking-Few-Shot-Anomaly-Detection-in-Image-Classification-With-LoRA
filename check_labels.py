import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from tqdm.auto import tqdm
from peft import get_peft_model, LoraConfig
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F
from PIL import Image
import os
from datasets import load_dataset

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Define LoRA configuration
lora_config = LoraConfig(
    inference_mode=False,  # Since we're fine-tuning, set inference_mode to False
    r=16,  # LoRA rank, controls the number of trainable parameters
    lora_alpha=32,  # Scaling factor for LoRA
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=[
        "self_attn.q_proj", 
        "self_attn.k_proj", 
        "self_attn.v_proj", 
        "self_attn.out_proj"
    ],
)

# Apply LoRA to the CLIP model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

mvtec_data_path = "anomalib/datasets/MVTec/transistor/" 
tokenized_datasets = load_dataset("imagefolder", data_dir=mvtec_data_path)
category = mvtec_data_path.split('/')[-2]  # Get the category
print(category)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

print(tokenized_datasets['test']['labels'])