import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from model.FusionNet import FusionNet
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        seg_softmax_path = self.dataframe.iloc[idx, 0]
        of_confidence_path = self.dataframe.iloc[idx, 1]
        target_path = self.dataframe.iloc[idx, 2]
        
        # Load arrays from .npy files
        seg_softmax_array = np.load(seg_softmax_path)
        of_confidence_array = np.load(of_confidence_path)
        target_array = np.load(target_path)
        
        # Convert arrays to tensors
        seg_softmax_tensor = torch.tensor(seg_softmax_array, dtype=torch.float32)
        of_confidence_tensor = torch.tensor(of_confidence_array, dtype=torch.float32)
        target_tensor = torch.tensor(target_array, dtype=torch.float32)

        
        return seg_softmax_tensor, of_confidence_tensor, target_tensor

# Load DataFrame with file paths
df = pd.read_csv('train_dataset.csv')

# Create Dataset and DataLoader
dataset = CustomDataset(dataframe=df)
train_loader = DataLoader(dataset, shuffle=True, batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes, height, width = 11, 480, 854
model = FusionNet(num_classes, height, width).to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer.zero_grad()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5) 

num_epochs = 5
best_loss = float('inf')

for epoch in range(num_epochs):  # Define num_epochs
    model.train()
    running_loss = 0.0
    # using tqdm to show a smart progress meter
    for seg_softmax, of_confidence, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        seg_softmax = seg_softmax.squeeze(1)
        of_confidence = of_confidence.squeeze(1)
        target = target.squeeze(1).squeeze(2)

        seg_softmax, of_confidence, target = seg_softmax.to(device), of_confidence.to(device), target.to(device)

        optimizer.zero_grad()
    
        outputs = model(seg_softmax, of_confidence)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    average_loss = running_loss / len(train_loader) 
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(model.state_dict(), 'best_Fusion_model.pth')  # Save best model
    scheduler.step()
    
