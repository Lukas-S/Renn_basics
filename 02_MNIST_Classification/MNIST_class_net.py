import torch
import torch.nn as nn

class SimpleMNISTNet(nn.Module):
    """Simple CNN for MNIST classification"""
    def __init__(self):
        super(SimpleMNISTNet, self).__init__()
        # Define model layers in sequential container
        self.model = nn.Sequential(
            # First conv layer: 1 input channel, 32 output channels, 3x3 kernel
            nn.Conv2d(1, 32, 3),  # 28x28 -> 26x26
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 26x26 -> 13x13
            
            # Second conv layer: 32 input channels, 64 output channels, 3x3 kernel
            nn.Conv2d(32, 64, 3),  # 13x13 -> 11x11
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # 11x11 -> 5x5
            
            # Flatten and fully connected layers
            nn.Flatten(),          # 64 * 5 * 5
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.model(x)
    
class MNIST(nn.Module):
    """Simple fully-connected neural network for MNIST classification"""
    def __init__(self):
        super(MNIST, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # Flatten 28x28 image to 784 vector
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        # x shape: [batch_size, 1, 28, 28]
        return self.model(x)
    
class EmbedMNISTNet(nn.Module):
    """Simple CNN for MNIST classification with embedding layer"""
    def __init__(self, embedding_dim=64):
        super(EmbedMNISTNet, self).__init__()
        
        # Convolutional feature extraction
        self.features = nn.Sequential(
            # First conv layer: 1 input channel, 32 output channels, 3x3 kernel
            nn.Conv2d(1, 32, 3),  # 28x28 -> 26x26
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 26x26 -> 13x13
            
            # Second conv layer: 32 input channels, 64 output channels, 3x3 kernel
            nn.Conv2d(32, 64, 3),  # 13x13 -> 11x11
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # 11x11 -> 5x5
            
            nn.Flatten()           # 64 * 5 * 5
        )
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(64 * 5 * 5, embedding_dim),
            nn.ReLU()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        # Extract features
        x = self.features(x)
        
        # Generate embedding
        embedding = self.embedding(x)
        
        # Classify
        output = self.classifier(embedding)
        
        return output
    
    def get_embedding(self, x):
        """Get the embedding representation for input x"""
        x = self.features(x)
        embedding = self.embedding(x)
        return embedding
    
