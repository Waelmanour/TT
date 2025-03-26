import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, duration=2)
        # Ensure audio length is consistent
        if len(audio) < sr * 2:
            audio = np.pad(audio, (0, sr * 2 - len(audio)))
        else:
            audio = audio[:sr * 2]
        # Extract mel spectrogram features with fixed parameters
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize with epsilon to avoid division by zero
        eps = 1e-6
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + eps)
        return torch.FloatTensor(mel_spec_db), self.labels[idx]

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        # Scale dot-product attention
        energy_scaled = energy / (C ** 0.5)
        attention = self.softmax(energy_scaled)
        value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Project and reshape for multi-head attention
        query = self.query(x).view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        key = self.key(x).view(batch_size, self.num_heads, self.head_dim, -1)
        value = self.value(x).view(batch_size, self.num_heads, self.head_dim, -1)
        
        # Scaled dot-product attention for each head
        energy = torch.matmul(query, key)
        energy_scaled = energy / (self.head_dim ** 0.5)
        attention = self.softmax(energy_scaled)
        
        # Apply attention weights
        out = torch.matmul(attention, value.permute(0, 1, 3, 2))
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, C, width, height)
        out = self.out_proj(out)
        
        # Residual connection
        out = self.gamma * out + x
        return out

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Fifth convolutional block (new)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # Attention mechanisms
        self.self_attention = SelfAttention(256)
        self.multi_head_attention = MultiHeadAttention(256, num_heads=8)
        
        # Adaptive pooling to ensure fixed size regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers with increased capacity
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 32)
        self.bn_fc3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        
        # Dropout for regularization (increased)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Convolutional blocks with batch normalization
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool5(torch.relu(self.bn5(self.conv5(x))))
        
        # Apply attention mechanisms
        x = self.self_attention(x)
        x = self.multi_head_attention(x)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, 256 * 4 * 4)
        
        # Fully connected layers with batch normalization
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Output layer with sigmoid activation
        x = torch.sigmoid(self.fc4(x))
        return x

def load_dataset():
    data_types = ['training', 'testing', 'validation']
    datasets = {dtype: {'paths': [], 'labels': []} for dtype in data_types}
    
    # Define all dataset folders
    dataset_folders = ['for-2seconds', 'for-norm', 'for-original', 'for-rerecorded']
    
    total_files = 0
    for dtype in data_types:
        for folder in dataset_folders:
            # Load real audio files
            real_dir = os.path.join('dataset', folder, dtype, 'real')
            if os.path.exists(real_dir):
                files = [f for f in os.listdir(real_dir) if f.endswith('.wav')]
                datasets[dtype]['paths'].extend([os.path.join(real_dir, f) for f in files])
                datasets[dtype]['labels'].extend([0] * len(files))  # 0 for real
                total_files += len(files)
                print(f'Loaded {len(files)} real files from {folder}/{dtype}')
            
            # Load fake audio files
            fake_dir = os.path.join('dataset', folder, dtype, 'fake')
            if os.path.exists(fake_dir):
                files = [f for f in os.listdir(fake_dir) if f.endswith('.wav')]
                datasets[dtype]['paths'].extend([os.path.join(fake_dir, f) for f in files])
                datasets[dtype]['labels'].extend([1] * len(files))  # 1 for fake
                total_files += len(files)
                print(f'Loaded {len(files)} fake files from {folder}/{dtype}')
    
    print(f'\nTotal files loaded: {total_files}')
    for dtype in data_types:
        print(f'{dtype} set size: {len(datasets[dtype]["paths"])}')
    
    return datasets

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, patience=5):
    model.train()
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # Data augmentation (random time shift and pitch shift)
            if np.random.random() > 0.5:
                # Apply random noise to input spectrograms
                noise = torch.randn_like(inputs) * 0.05
                inputs = inputs + noise
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}, Accuracy: {100*correct/total:.2f}%')
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%\n')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load the best model state if early stopping occurred
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model with validation loss: {best_val_loss:.4f}')
    
    return model

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)

def load_trained_model(path='model.pth'):
    model = AudioClassifier()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict(model, audio_path, device):
    model.eval()
    # Load audio with the same parameters as training
    audio, sr = librosa.load(audio_path, duration=2)
    
    # Ensure audio length is consistent
    if len(audio) < sr * 2:
        audio = np.pad(audio, (0, sr * 2 - len(audio)))
    else:
        audio = audio[:sr * 2]
    
    # Extract mel spectrogram features with the same parameters as training
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize with epsilon to avoid division by zero
    eps = 1e-6
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + eps)
    
    with torch.no_grad():
        input_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).to(device)
        output = model(input_tensor)
        prediction = output.item()
    
    # Return prediction confidence
    return prediction

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare datasets
    datasets = load_dataset()
    
    # Create data loaders for each set
    train_dataset = AudioDataset(datasets['training']['paths'], datasets['training']['labels'])
    test_dataset = AudioDataset(datasets['testing']['paths'], datasets['testing']['labels'])
    val_dataset = AudioDataset(datasets['validation']['paths'], datasets['validation']['labels'])
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and training components
    model = AudioClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print("Starting model training with enhanced architecture...")
    # Train model with validation data for early stopping
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30)
    
    # Evaluate model on test and validation sets
    model.eval()
    print("\nFinal model evaluation:")
    with torch.no_grad():
        for loader_name, loader in [('Test', test_loader), ('Validation', val_loader)]:
            correct = 0
            total = 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'{loader_name} Accuracy: {accuracy:.2f}%')
    
    # Save the trained model
    save_model(model)
    print("Enhanced model saved successfully.")

if __name__ == '__main__':
    main()