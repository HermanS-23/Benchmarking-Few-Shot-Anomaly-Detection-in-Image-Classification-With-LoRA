import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, Dinov2ForImageClassification
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
from setfit import sample_dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch.nn.functional as F
from torch import nn
from peft import get_peft_model, LoraConfig
from torch.optim.lr_scheduler import StepLR

def run_program(x):
    # Load dataset from local directory
    mvtec_data_path = "anomalib/datasets/MVTec/transistor/" # Specify the file path for the category of the testing dataset you wish to use.
    tokenized_datasets = load_dataset("imagefolder", data_dir=mvtec_data_path)

    # Extract the category from the path
    category = mvtec_data_path.split('/')[-2]  # Get the category (e.g., 'bottle')
    print(f"Category: {category}")

    # Rename the label column to 'labels' for compatibility
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # On varies samples
    train_val_split = tokenized_datasets["test"].train_test_split(test_size=0.50)
    train_dataset = sample_dataset(train_val_split['train'], num_samples=x, label_column="labels") # sample dataset
    val_dataset = train_val_split['test']
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # Ensure the labeling is correct
    def set_labels(example):
        if example['labels'] == 3: # Assuming 3 is "good"
            example['labels'] = 0
        else:
            example['labels'] = 1 # Anomaly (other classes can also be set to 1)
        return example

    train_dataset = train_dataset.map(set_labels)
    val_dataset = val_dataset.map(set_labels)
    print(train_dataset['labels'])
    print(val_dataset['labels'])

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the DINOv2 model and image processor from Hugging Face
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
    model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    # Define LoRA configuration
    lora_config = LoraConfig(
        inference_mode=False, 
        r=16,  # LoRA rank, controls the number of trainable parameters
        lora_alpha=32,  # Scaling factor for LoRA
        lora_dropout=0.1,  # Dropout for LoRA layers
        target_modules=[
            "query",  # Attention query projection
            "key",    # Attention key projection
            "value",  # Attention value projection
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.to(device)

    def data_collator(batch):
        images = [item['image'] for item in batch]
        labels = [item['labels'] for item in batch]
        processed_images = image_processor(images=images, return_tensors="pt")
        processed_images['labels'] = torch.tensor(labels)
        return processed_images

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    model.train()
    num_epochs = 15

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(train_dataloader):
    #        batch = {k: v.to(device) for k, v in batch.items()}  # Exclude labels from model input
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
    #        scheduler.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / len(train_dataloader)
        accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
    # Save the model
    model.save_pretrained("outputs_dinov2-Lora")
    image_processor.save_pretrained("outputs_dinov2-Lora")
    print("model saved")

    # Evaluation loop
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs).logits
            probs = F.softmax(outputs, dim=-1)
            print("probs:", probs)
            
            predicted = (probs[:, 1] > 0.7).int()  # threshold anomaly 0.7
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(all_preds)
    print(all_labels)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    return  roc_auc, f1

import openpyxl

# Create a new workbook and select the active sheet
workbook = openpyxl.Workbook()
sheet = workbook.active

# Optionally set the sheet name
sheet.title = "temp_Results"

# Starting row for appending results
start_row = 1

# Write headers
sheet['B1'] = "Variable 1"  # Header for Variable 1
sheet['C1'] = "Variable 2"  # Header for Variable 2

for j in range(4):  # For 4 sets of results
    col1 = chr(66 + (j * 2))  # B, D, F, H
    col2 = chr(67 + (j * 2))  # C, E, G, I
    x = pow(2, j)
    for i in range(5):
        var1, var2 = run_program(x) 
        sheet[f'{col1}{start_row + i}'] = round(var1 * 100, 2)  # Append var1 to the first column
        sheet[f'{col2}{start_row + i}'] = round(var2 * 100, 2)  # Append var2 to the second column

# Save the new workbook
workbook.save("data_collected_temp.xlsx")
print("Results saved to data_collected_temp.xlsx.")