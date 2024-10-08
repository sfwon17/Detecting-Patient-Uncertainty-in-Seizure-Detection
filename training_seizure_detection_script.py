import os
import pandas as pd
import numpy as np
from cnn import CNN
import torch.nn as nn
import torch.optim as optim
from utils.CustomDataset import CustomDataset

# Import your training and validation EEG in numpy format with 1 seconds segments. 
# training Data must be in the format of (samples, 1, time, channels), and labels (samples, )
train_label = np.load('EEG/train_EEG.npy')
train_label = np.load('EEG/train_label.npy')
val_data = np.load('EEG/val_EEG.npy')
val_label = np.load('EEG/val_label.npy') 

# Create data loaders  
train = CustomDataset(train_data, train_label)
val = CustomDataset(val_reshaped,val_label)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False) # make sure shuffle is set to false since we need it is a continuous EEG

# Define parameters
model = CNN().to("cuda:0") # A6000
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.90, nesterov=False)
epoch_count = 100

for epoch in range(epoch_count):
    epoch_loss = 0
    true_labels = []
    predicted_labels = []

    model.train()
    for data, label in train_loader:
        data, label = data.to('cuda:0'), label.to('cuda:0')
        
        # Forward pass
        output = model(data)
        loss = criterion(output, label)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Softmax and prediction
        output = F.softmax(output, dim=1)
        _, predictions = torch.max(output, 1)
        
        # Update running loss
        epoch_loss += loss.item() / len(train_loader)
        
        # Collect labels for evaluation
        true_labels.extend(label.cpu().numpy())
        predicted_labels.extend(predictions.cpu().numpy())

    # Calculate training metrics
    epoch_recall = recall_score(true_labels, predicted_labels, pos_label=1)
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Reset label collections for validation
    true_labels = []
    predicted_labels = []

    # Validation step
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for data, label in valid_loader:
            data, label = data.to('cuda:0'), label.to('cuda:0')

            # Forward pass for validation
            val_output = model(data)
            val_loss = criterion(val_output, label)
            epoch_val_loss += val_loss.item() / len(valid_loader)
            
            # Softmax and prediction for validation
            val_output = F.softmax(val_output, dim=1)
            _, val_predictions = torch.max(val_output, 1)
            
            # Collect validation labels for evaluation
            true_labels.extend(label.cpu().numpy())
            predicted_labels.extend(val_predictions.cpu().numpy())

    # Calculate validation metrics
    epoch_val_recall = recall_score(true_labels, predicted_labels, pos_label=1)
    val_accuracy = accuracy_score(true_labels, predicted_labels)

    # Print epoch summary
    print(
        f"Epoch: {epoch + 1} - "
        f"Loss: {epoch_loss:.4f} - Acc: {accuracy:.4f} - Recall: {epoch_recall:.4f} - "
        f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {val_accuracy:.4f} - Val Recall: {epoch_val_recall:.4f}"
    )
# save model
torch.save(model.state_dict(), 'seizure_detection_model.pth')
