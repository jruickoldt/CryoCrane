import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
#test


def load_score(txt_path):
    """Reads the score from a .txt file."""
    with open(txt_path, 'r') as f:
        return float(f.read().strip())

class ImageScoreDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        # Traverse subdirectories
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith(".png"):
                        img_path = os.path.join(subdir_path, file)
                        txt_path = img_path.replace(".png", ".txt")
                        if os.path.exists(txt_path):
                            score = load_score(txt_path)
                            self.data.append((img_path, score))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, score = self.data[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        image = np.array(image, dtype=np.float32)  # Convert to NumPy array
    
        # Normalize to [0, 255]
        min_val, max_val = image.min(), image.max()
        image = 255 * (image - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
        
        image = Image.fromarray(image)  # Convert back to PIL
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(score, dtype=torch.float32)

    


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, with_r=False, bias=True):
        super().__init__()
        self.with_r = with_r
        extra_channels = 2 + int(with_r)  # x, y, and optional radius
        self.conv = nn.Conv2d(in_channels + extra_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        device = x.device
        xx_channel = torch.arange(width, dtype=torch.float32, device=device).repeat(1, height, 1)
        yy_channel = torch.arange(height, dtype=torch.float32, device=device).repeat(1, width, 1).transpose(1, 2)
        xx_channel = xx_channel / (width - 1)
        yy_channel = yy_channel / (height - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.expand(batch_size, -1, -1, -1)
        yy_channel = yy_channel.expand(batch_size, -1, -1, -1)

        coords = [xx_channel, yy_channel]

        if self.with_r:
            rr = torch.sqrt(xx_channel**2 + yy_channel**2)
            coords.append(rr)

        coord_tensor = torch.cat(coords, dim=1)
        x = torch.cat([x, coord_tensor], dim=1)
        return self.conv(x)



# Define Basic Residual Block
class ResidualBlock_small(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        return self.relu(out)



class CustomResNet(nn.Module):
    def __init__(self, block, layers, dropout_rate=0.5, num_classes=1, in_channels=1):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def ResNet34(dropout_rate, **kwargs):
    return CustomResNet(BasicBlock, [3, 4, 6, 3], dropout_rate=dropout_rate, **kwargs)

def ResNet50(dropout_rate, **kwargs):
    return CustomResNet(Bottleneck, [3, 4, 6, 3], dropout_rate=dropout_rate, **kwargs)

def ResNet152(dropout_rate, **kwargs):
    return CustomResNet(Bottleneck, [3, 8, 36, 3], dropout_rate=dropout_rate, **kwargs)
        
class CoordNet8(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.coordconv = CoordConv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = ResidualBlock(16, 32, stride=2)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.coordconv(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
# Define CoordConv ResNet-8 Model with radius attention
class RCoordNet8(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.coordconv = CoordConv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False, with_r=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = ResidualBlock(16, 32, stride=2)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.coordconv(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class ResNet4(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = ResidualBlock(16, 32, stride=2)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)

        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Sigmoid for regression output
        return x

    # Define ResNet-6 Model

class ResNet6(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = ResidualBlock(16, 32, stride=2)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)

        
        # Global Average Pooling instead of a fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(64, 1, kernel_size=1)  # 1x1 conv instead of dense layer
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.global_pool(x)
        x = self.fc(x)
        x = self.sigmoid(x)  # Sigmoid for regression output
        return x
    

# Define ResNet-8 Model
class ResNet8(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = ResidualBlock(16, 32, stride=2)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Sigmoid for regression output
        return x

# Define ResNet-8 Model
class ResNet10(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = ResidualBlock(16, 32, stride=2)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.layer4 = ResidualBlock(128, 256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Sigmoid for regression output
        return x
# Define ResNet-12 Model
class ResNet12(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = ResidualBlock(16, 32, stride=2)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.layer4 = ResidualBlock(128, 256, stride=2)
        self.layer5 = ResidualBlock(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Sigmoid for regression output
        return x

# Instantiate model with adjustable dropout

def predict(model, image_tensor, device):
    """
    Run inference on a single image.
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        score = model(image_tensor)  # Forward pass
    return score.item()

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help = "directory containing the training data set")
    parser.add_argument("-n", "--name", help = "name of the output file")
    parser.add_argument("-w", "--width", help = "dimension of the images",  type = int)
    #parser.add_argument("-b", "--batch_size", help = "batch size for training", type = int)
    parser.add_argument("-e", "--epochs", help = "number of epochs", type = int)
    parser.add_argument("-d", "--dropout", help= "Specifiy dropout in the last layer, number between 0 and 1", type = float)
    parser.add_argument("-a", "--augmentation", help = "use this flag if you want data augmentation" , action='store_true')
    parser.add_argument("-m", "--model", help = "Valid options are ResNet8,ResNet10, ResNet12, ResNet16, ResNet34, ResNet50, ResNet101 and ResNet152." )
    parser.add_argument("-l", "--learning_rate", help = "Default is 1e-4", type = float )
    args = parser.parse_args()
    size = args.width
    name = args.name
    Data_path = args.input
    epochs = args.epochs
    learning_rate = args.learning_rate
    dropout_rate = args.dropout
    model_name = args.model
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip (50% chance)
        transforms.RandomVerticalFlip(p=0.5),    # Random vertical flip (50% chance)
        transforms.RandomRotation(degrees=10),   # Random rotation between -10 and +10 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    print(f"_______________________________________\nStarting training - {name} for {epochs} epochs\nModel: {model_name}\ndropout: {dropout_rate:.2f}\nlearning rate: {learning_rate}\nimage size: {size}\n_______________________________________")
    # Load dataset
    dataset = ImageScoreDataset(Data_path, transform=transform)
    
    # Split dataset (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    test_scores = []
    for i in range(test_size):
        test_images,test_score = test_dataset[i]
        test_scores.append(test_score)
    
    

    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}, mean score: {np.mean(test_scores):.4f}, standard deviation: {np.std(test_scores):.4f}")
    
    
    if model_name == "ResNet8":
        model = ResNet8(dropout_rate=dropout_rate)
    elif model_name == "ResNet10":
        model = ResNet10(dropout_rate=dropout_rate)
    elif model_name == "ResNet12":
        model = ResNet12(dropout_rate=dropout_rate)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #device = torch.device("cpu")
    model.to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    def train_model(model, train_loader, criterion, optimizer, criterion_val = nn.L1Loss(), epochs=30):
        best_val_loss = 1000000  # Initialize best loss as infinite
        model.train()
        val_loss_history = []
        train_loss_history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for images, scores in train_loader:
                optimizer.zero_grad()
                outputs = model(images).squeeze()
                loss = criterion(outputs, scores)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss/len(train_loader)
            val_loss = 0.0
            for images, scores in test_loader:
                outputs = model(images).squeeze()
                loss = criterion(outputs, scores)
                val_loss += loss.item()
            validation_loss = val_loss/len(test_loader)
            val_loss_history.append(validation_loss)
            train_loss_history.append(train_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Validation MAE: {val_loss/len(test_loader):.4f}")
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss 
                torch.save(model.state_dict(), name+"_"+model_name+f"_{int(size)}_"+"model_weights.pth")
                print(f"saved model to {name}_{model_name}_{int(size)}_model_weights.pth with a validation loss of {best_val_loss:.4f}")
        return val_loss_history, train_loss_history, epochs


    
    validation_loss, training_loss, epochs = train_model(model, train_loader, criterion, optimizer, epochs=epochs)

    #Load the best model
    model.load_state_dict(torch.load(name+"_"+model_name+f"_{int(size)}_"+"model_weights.pth", map_location=device))
    model.to(device)
    model.eval()
    


    plot_training(model, training_loss, validation_loss, epochs, name, test_dataset=test_dataset, device = device)
    
