import torch
import torch.nn as nn
from torchvision import models

class HybridModel(nn.Module):
    def __init__(self, num_classes=1):
        super(HybridModel, self).__init__()
        
        # CNN Backbone (ResNet50)
        resnet = models.resnet50(pretrained=True)
        # Remove the fully connected layer
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        
        # Transformer Encoder
        # ResNet50 output: (Batch, 2048, 7, 7) -> flatten spatial dimensions -> (Batch, 2048, 49)
        self.feature_dim = 2048
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool1d(1) # Pool over sequence length
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 3, 224, 224)
        features = self.cnn(x) 
        # features shape: (Batch, 2048, 7, 7)
        
        batch_size, channels, h, w = features.size()
        
        # Reshape for transformer: (Batch, Sequence Length, Features)
        # Sequence Length = h * w = 49
        features = features.view(batch_size, channels, -1).permute(0, 2, 1)
        # features shape: (Batch, 49, 2048)
        
        transformer_out = self.transformer(features)
        # transformer_out shape: (Batch, 49, 2048)
        
        # Global Average Pooling over the sequence dimension
        # Permute back to (Batch, Features, Sequence Length) for pooling
        transformer_out = transformer_out.permute(0, 2, 1)
        pooled = self.avgpool(transformer_out).squeeze(-1)
        # pooled shape: (Batch, 2048)
        
        out = self.fc(pooled)
        return out
