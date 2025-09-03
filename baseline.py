!pip install rasterio
import os
import glob
import rasterio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from google.colab import drive

# Mount Google Drive
mount_point = '/content/drive'
if not os.path.exists(os.path.join(mount_point, 'MyDrive')):
    drive.mount(mount_point)
    print(f"Drive mounted at {mount_point}")
else:
    print(f"Drive already mounted at {mount_point}")

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class WildfireGeoTIFFDataset(Dataset):
    """Dataset for wildfire prediction using GeoTIFF data"""
    def __init__(self, data_dir, seq_length=3, prediction_offset=1, transform=None,
                 fire_threshold=0.2, fire_band_idx=7):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.prediction_offset = prediction_offset
        self.transform = transform
        self.fire_threshold = fire_threshold
        self.fire_band_idx = fire_band_idx

        # Find all GeoTIFF files
        self.tif_files = sorted(glob.glob(os.path.join(data_dir, "HighRes_Data_*.tif")))
        if not self.tif_files:
            # Fall back to old pattern if no new files found
            self.tif_files = sorted(glob.glob(os.path.join(data_dir, "LSTM_Data_*.tif")))

        if not self.tif_files:
            raise ValueError(f"No GeoTIFF files found in {data_dir}")

        # Extract year and month from filenames for chronological ordering
        self.dates = [self._extract_date(os.path.basename(file)) for file in self.tif_files]

        # Sort files by date
        sorted_indices = np.argsort(self.dates)
        self.tif_files = [self.tif_files[i] for i in sorted_indices]
        self.dates = [self.dates[i] for i in sorted_indices]

        # Validate dataset
        if len(self.tif_files) < seq_length + prediction_offset:
            raise ValueError(f"Need at least {seq_length + prediction_offset} GeoTIFF files, but found only {len(self.tif_files)}")

        # Determine the standard number of bands by scanning all files
        self.num_bands = self._determine_standard_bands()

        # Get dimensions from first file
        with rasterio.open(self.tif_files[0]) as src:
            self.height = src.height
            self.width = src.width

        print(f"Dataset initialized with {len(self.tif_files)} GeoTIFF files")
        print(f"Standardized to {self.num_bands} bands with dimensions {self.height}x{self.width}")

        # Updated band order explanation for the new high-res data
        if "HighRes_Data" in self.tif_files[0]:
            print(f"New high-resolution data format detected")
            print(f"Channel meanings in high-resolution data:")
            print("  Channel 0: NDVI (Normalized Difference Vegetation Index)")
            print("  Channel 1: NBR (Normalized Burn Ratio)")
            print("  Channel 2: Temperature")
            print("  Channel 3: Dew point")
            print("  Channel 4: Wind speed (u component)")
            print("  Channel 5: Wind direction (v component)")
            print("  Channel 6: Elevation")
            print("  Channel 7: Fire intensity (FRP)")
        else:
            print(f"Standard band order: NDVI, Temperature, Dew Point, Wind U, Wind V, Fire Intensity")


        # Print date range of the dataset
        start_date = self._format_date(self.dates[0])
        end_date = self._format_date(self.dates[-1])
        print(f"Date range: {start_date} to {end_date}")

    def _determine_standard_bands(self):
        """Scan all files to determine a standard number of bands"""
        max_bands = 0
        for file in self.tif_files:
            with rasterio.open(file) as src:
                max_bands = max(max_bands, src.count)
        return max_bands

    def _extract_date(self, filename):
        """Extract date from filename format 'LSTM_Data_YYYY_MM.tif'"""
        parts = filename.replace('.tif', '').split('_')
        try:
            year = int(parts[-2])
            month = int(parts[-1])
            return year * 100 + month  # Convert to sortable format (YYYYMM)
        except (IndexError, ValueError):
            return 0

    def _format_date(self, date_code):
        """Format date code (YYYYMM) to human-readable string"""
        year = date_code // 100
        month = date_code % 100
        return f"{year}-{month:02d}"

    def __len__(self):
        """Return the number of sequences that can be formed"""
        return max(0, len(self.tif_files) - (self.seq_length + self.prediction_offset - 1))

    def __getitem__(self, idx):
        """Get a sequence of data and the corresponding target"""
        # Load sequence of images
        sequence = []
        for i in range(self.seq_length):
            file_idx = idx + i
            with rasterio.open(self.tif_files[file_idx]) as src:
                # Read all available bands
                available_bands = src.count
                data = src.read().astype(np.float32)

                # Pad if this file has fewer bands than standard
                if available_bands < self.num_bands:
                    padding = np.zeros((self.num_bands - available_bands, self.height, self.width), dtype=np.float32)
                    data = np.concatenate([data, padding], axis=0)

                # If there are more bands than standard, truncate
                elif available_bands > self.num_bands:
                    data = data[:self.num_bands]

                # Normalize each band
                for band in range(data.shape[0]):
                    band_min = np.nanmin(data[band])
                    band_max = np.nanmax(data[band])
                    if band_max > band_min:
                        data[band] = (data[band] - band_min) / (band_max - band_min)

                # Replace NaN values
                data = np.nan_to_num(data)
                sequence.append(data)

        # Stack sequence
        sequence_tensor = np.stack(sequence)

        # Load target
        target_idx = idx + self.seq_length + self.prediction_offset - 1
        with rasterio.open(self.tif_files[target_idx]) as src:
            # Ensure fire_band_idx doesn't exceed available bands
            available_bands = src.count
            fire_band = min(self.fire_band_idx, available_bands - 1)

            # Read the fire intensity band
            target = src.read(fire_band + 1).astype(np.float32)

            # Normalize target
            target_min = np.nanmin(target)
            target_max = np.nanmax(target)
            if target_max > target_min:
                target = (target - target_min) / (target_max - target_min)

            # Replace NaN values and binarize
            target = np.nan_to_num(target)
            target_binary = (target > self.fire_threshold).astype(np.float32)

        # Apply transformations if provided
        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)
            target_binary = self.transform(target_binary)

        # Convert to torch tensors
        x = torch.from_numpy(sequence_tensor)
        y = torch.from_numpy(target_binary).unsqueeze(0)  # Add channel dimension

        # Return the sample index along with the data for later reference
        return x, y, idx

# ULSTM model components (unchanged)
class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatio-temporal modeling"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Convert kernel_size to tuple if it's an integer
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.bias = bias

        # Gates: input, forget, output, cell
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Convolutional operation
        conv_output = self.conv(combined)

        # Split gates
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)

        # Apply activations
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate

        # Update states
        c_next = f * c_cur + i * g  # Cell state
        h_next = o * torch.tanh(c_next)  # Hidden state

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class EncoderBlock(nn.Module):
    """Encoder block with residual connections"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.3):
        super(EncoderBlock, self).__init__()

        # Ensure kernel_size is properly formatted
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x + residual  # Add residual connection
        x = self.dropout(x)

        return x

class DecoderBlock(nn.Module):
    """Decoder block with skip connections"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.3):
        super(DecoderBlock, self).__init__()

        # Ensure kernel_size is properly formatted
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        # Using in_channels directly for first conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x, skip=None):
        if skip is not None:
            # Make sure skip connection has same spatial dimensions
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.dropout(x)

        return x

class ULSTM(nn.Module):
    """
    U-Convolutional Long Short-Term Memory (ULSTM) model for wildfire prediction.
    """
    def __init__(self, input_channels=6, hidden_dim=32, lstm_kernel_size=3, time_steps=3):
        super(ULSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps

        # Encoder blocks
        self.encoder1 = EncoderBlock(input_channels, hidden_dim)
        self.encoder2 = EncoderBlock(hidden_dim, hidden_dim*2)
        self.encoder3 = EncoderBlock(hidden_dim*2, hidden_dim*4)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ConvLSTM layers
        self.conv_lstm1 = ConvLSTMCell(hidden_dim, hidden_dim, lstm_kernel_size)
        self.conv_lstm2 = ConvLSTMCell(hidden_dim*2, hidden_dim*2, lstm_kernel_size)
        self.conv_lstm3 = ConvLSTMCell(hidden_dim*4, hidden_dim*4, lstm_kernel_size)

        # Decoder blocks
        self.upconv3 = nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=2, stride=2)
        self.decoder3 = DecoderBlock(hidden_dim*4, hidden_dim*2)

        self.upconv2 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=2, stride=2)
        self.decoder2 = DecoderBlock(hidden_dim*2, hidden_dim)

        # Attention mechanism for better focus on relevant features
        self.attention = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        # Final layer
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()

        # Calculate pooled dimensions
        h1 = height // 2
        w1 = width // 2
        h2 = h1 // 2
        w2 = w1 // 2
        h3 = h2 // 2
        w3 = w2 // 2

        # Initialize hidden states for ConvLSTM layers
        h1, c1 = self.conv_lstm1.init_hidden(batch_size, (h1, w1))
        h2, c2 = self.conv_lstm2.init_hidden(batch_size, (h2, w2))
        h3, c3 = self.conv_lstm3.init_hidden(batch_size, (h3, w3))

        # Store encoder outputs for skip connections
        enc1_outputs = []
        enc2_outputs = []

        # Process each timestep through encoder
        for t in range(seq_len):
            x_t = x[:, t]

            # Encoder path
            enc1 = self.encoder1(x_t)
            enc1_outputs.append(enc1)

            x_t = self.pool(enc1)
            enc2 = self.encoder2(x_t)
            enc2_outputs.append(enc2)

            x_t = self.pool(enc2)
            enc3 = self.encoder3(x_t)
            x_t = self.pool(enc3)

            # Update ConvLSTM states
            h1, c1 = self.conv_lstm1(self.pool(enc1), (h1, c1))
            h2, c2 = self.conv_lstm2(self.pool(enc2), (h2, c2))
            h3, c3 = self.conv_lstm3(x_t, (h3, c3))

        # Start decoding from the last ConvLSTM state
        x = h3

        # Upsampling and skip connections from the last timestep
        x = self.upconv3(x)

        # Ensure size matches for skip connection
        if x.shape[2:] != enc2_outputs[-1].shape[2:]:
            x = F.interpolate(x, size=enc2_outputs[-1].shape[2:], mode='bilinear', align_corners=False)

        x = self.decoder3(x, enc2_outputs[-1])

        x = self.upconv2(x)

        # Ensure size matches for skip connection
        if x.shape[2:] != enc1_outputs[-1].shape[2:]:
            x = F.interpolate(x, size=enc1_outputs[-1].shape[2:], mode='bilinear', align_corners=False)

        x = self.decoder2(x, enc1_outputs[-1])

        # Apply attention mechanism
        attention_weights = torch.sigmoid(self.attention(x))
        x = x * attention_weights

        # Final convolution to get fire probability
        x = self.final_conv(x)

        # Ensure the final output matches the input spatial dimensions
        if x.shape[2:] != (height, width):
            x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)

        # Apply sigmoid to get probability in [0,1] range
        x = torch.sigmoid(x)

        return x

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def create_chronological_split(dataset, train_years=(2013, 2020), val_years=(2021, 2022)):
    """Split dataset chronologically based on years"""
    train_indices = []
    val_indices = []

    for i in range(len(dataset)):
        # Get sample index
        _, _, idx = dataset[i]

        # Get year from tif file name
        filename = os.path.basename(dataset.tif_files[idx])
        year = int(filename.split('_')[-2])

        if train_years[0] <= year <= train_years[1]:
            train_indices.append(i)
        elif val_years[0] <= year <= val_years[1]:
            val_indices.append(i)

    print(f"Chronological split - Training years: {train_years}, Validation years: {val_years}")
    print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001,
                weight_decay=1e-5, patience=10, save_path='best_model.pth',
                use_focal_loss=True):
    """Train the ULSTM model"""
    # Move model to device
    model = model.to(device)

    # Loss function and optimizer
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using Focal Loss for class imbalance")
    else:
        criterion = nn.BCELoss()
        print("Using standard BCE Loss")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_auc': [],
        'learning_rate': [],
        'pos_rate': []  # Track positive sample rate
    }

    # Early stopping variables
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        for batch in progress_bar:
            # Handle batch with or without index
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_probs = []  # Store raw probabilities for AUC calculation
        all_targets = []
        all_pos_pixels = 0
        all_total_pixels = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                # Handle batch with or without index
                if len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch

                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Update statistics
                val_loss += loss.item() * inputs.size(0)

                # Count positive pixels for class balance metrics
                all_pos_pixels += torch.sum(targets > 0.5).item()
                all_total_pixels += targets.numel()

                # Convert outputs to binary predictions and store raw probabilities
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(np.float32)
                targets_np = targets.cpu().numpy()

                # Store for metrics calculation
                all_probs.extend(probs.reshape(-1))
                all_preds.extend(preds.reshape(-1))
                all_targets.extend(targets_np.reshape(-1))

        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)

        # Calculate validation metrics
        val_precision = precision_score(all_targets, all_preds, zero_division=0)
        val_recall = recall_score(all_targets, all_preds, zero_division=0)
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        val_auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0

        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        # Calculate positive rate
        pos_rate = all_pos_pixels / all_total_pixels * 100
        history['pos_rate'].append(pos_rate)

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        scheduler.step(val_loss)

        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, AUC: {val_auc:.4f}')
        print(f'  Positive pixel rate: {pos_rate:.2f}%')
        print(f'  Learning Rate: {current_lr:.6f}')

        # Check for improvement for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history
            }, save_path)

            print(f'  Improved! Model saved to {save_path}')
        else:
            early_stop_counter += 1
            print(f'  No improvement for {early_stop_counter} epochs')

            if early_stop_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

    # Load best model weights
    try:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    except Exception as e:
        print(f"Could not load best model: {e}")

    # Plot training history
    plot_training_history(history, save_path.replace('.pth', '_history.png'))

    return model, history
def plot_training_history(history, save_path=None):
    """Plot training history metrics with enhanced visualization"""
    plt.figure(figsize=(15, 15))

    # Plot loss
    plt.subplot(3, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss During Training', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Plot F1 score
    plt.subplot(3, 2, 2)
    plt.plot(history['val_f1'], 'g-', label='F1 Score')
    plt.title('F1 Score (Validation)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim([0, 1.05])
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Plot precision and recall
    plt.subplot(3, 2, 3)
    plt.plot(history['val_precision'], 'c-', label='Precision')
    plt.plot(history['val_recall'], 'm-', label='Recall')
    plt.title('Precision and Recall (Validation)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim([0, 1.05])
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Plot AUC
    plt.subplot(3, 2, 4)
    plt.plot(history['val_auc'], 'y-', label='AUC')
    plt.title('ROC AUC (Validation)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim([0, 1.05])
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Plot learning rate
    plt.subplot(3, 2, 5)
    plt.plot(history['learning_rate'], 'k-', label='Learning Rate')
    plt.title('Learning Rate', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Plot positive sample rate
    plt.subplot(3, 2, 6)
    plt.plot(history['pos_rate'], 'r-', label='Positive Pixel %')
    plt.title('Positive Pixel Rate', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Training history plot saved to {save_path}')

    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Dataset parameters
    data_dir = '/content/drive/My Drive/wildfire_project/LSTM_Data_HighRes'  # Updated path
    seq_length = 3  # Use 3 months of data to predict the next month
    prediction_offset = 1  # Predict 1 month ahead
    batch_size = 4
    fire_threshold = 0.2  # Adjust based on your data

    # Create dataset with updated fire_band_idx
    dataset = WildfireGeoTIFFDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        prediction_offset=prediction_offset,
        fire_threshold=fire_threshold,
        fire_band_idx=7  # Updated to 7 for fire intensity in new format
    )

    # Split dataset chronologically - updated years based on your data range
    train_dataset, val_dataset = create_chronological_split(
        dataset,
        train_years=(2015, 2022),  # Train on 2015-2022 data
        val_years=(2023, 2024)     # Validate on 2023-2024 data
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f'Training set: {len(train_dataset)} samples')
    print(f'Validation set: {len(val_dataset)} samples')

    # Create model with updated input channels
    model = ULSTM(
        input_channels=dataset.num_bands,  # This will now be 8 for the new data
        hidden_dim=32,
        lstm_kernel_size=3,
        time_steps=seq_length
    )

    # Train model with focal loss to handle class imbalance
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30,
        learning_rate=0.001,
        weight_decay=1e-4,
        patience=10,
        save_path='high_res_wildfire_model.pth',  # Updated name
        use_focal_loss=True
    )

    print("Training complete! You can now evaluate the model on test data.")

if __name__ == '__main__':
    main()
