import torch
from sklearn.metrics import roc_auc_score, f1_score
from transformers import CLIPProcessor, CLIPModel, TrainingArguments
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch.nn.functional as F
from setfit import sample_dataset

def run_program(x):
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    mvtec_data_path = "anomalib/datasets/MVTec/wood/" # Specify the file path for the category of the testing dataset you wish to use.
    tokenized_datasets = load_dataset("imagefolder", data_dir=mvtec_data_path)
    #tokenized_datasets = tokenized_datasets.remove_columns(["image"])
    category = mvtec_data_path.split('/')[-2]  # Get the category
    print(category)

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    # On varies samples
    train_val_split = tokenized_datasets["test"].train_test_split(test_size=0.50)
    train_dataset = sample_dataset(train_val_split['train'], num_samples=x, label_column="labels") # sample dataset
    val_dataset = train_val_split['test']
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    
    # Ensure the labeling is correct
    def set_labels(example):
        if example['labels'] == 2: # Assuming 2 is "good"
            example['labels'] = 0
        else:
            example['labels'] = 1 # Anomaly (other classes can also be set to 1)
        return example

    # Apply the label setting to include multiple anomaly labels
    train_dataset = train_dataset.map(set_labels)
    val_dataset = val_dataset.map(set_labels)
    print(train_dataset['labels'])
    print(val_dataset['labels'])


    def data_collator(batch):
        # Assuming batch is a list of tuples (image_tensor, label)
        images = [item['image'] for item in batch]  # Access the 'image' key
        labels = [item['labels'] for item in batch]  # Access the 'label' key
        texts = [f"normal {category}", f"anomalous {category}"]
        
        # Process images and texts using the CLIP processor
        data = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)

        # Adding labels into the returned data dictionary
        data['labels'] = torch.tensor(labels)

        return data

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn=data_collator)
    eval_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=data_collator)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=1e-7)

    # Custom training loop
    num_epochs = 15
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            labels = batch.pop('labels').to(device)  # Move labels to device  
            batch = {k: v.to(device) for k, v in batch.items()}  # Exclude labels from model input
            
            outputs = model(**batch)
            
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            # Compute contrastive loss in both directions
            loss_image_to_text = F.cross_entropy(logits_per_image, labels)
            loss_text_to_image = F.cross_entropy(logits_per_text.transpose(0, 1), labels)
            loss = (loss_image_to_text + loss_text_to_image) / 2
            
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
            progress_bar.update(1)

    model.save_pretrained('outputs_CLIP')
    processor.save_pretrained('outputs_CLIP')
    print("model saved")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Evaluation function
    def evaluate(model, eval_dataloader, device):
        model.eval()  # Set model to evaluation mode
        all_preds = []
        all_labels = []
        
        with torch.no_grad():  # Disable gradient computation
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                labels = batch.pop('labels').to(device)  # Move labels to device
                batch = {k: v.to(device) for k, v in batch.items()}  # Move the rest of the batch to the device

                # Perform forward pass
                outputs = model(**batch)

                logits_per_image = outputs.logits_per_image  # Get image logits
                probs = F.softmax(logits_per_image, dim=1)
                print("probs:", probs)

                anomalous_probs = probs[:, 1] 
                preds = (anomalous_probs > 0.7).long()  # Apply threshold
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute evaluation metrics
        f1 = f1_score(all_labels, all_preds, average='weighted')
        roc_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) == 2 else None

        print("preds:", all_preds)
        print("labels:", all_labels)
        return f1, roc_auc

    # Run evaluation
    f1, roc_auc = evaluate(model, eval_dataloader, device)

    # Output the results
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    else:
        print("ROC AUC is not applicable for multiclass classification.")
        
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
