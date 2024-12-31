import torch
import torch.nn.functional as F
import os

# Layer Norm instead of Batch norm for language applications.
class ResNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, features):
        super().__init__()
        self.cnn1 = torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = torch.nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.layer_norm1 = torch.nn.LayerNorm(features)
        self.layer_norm2 = torch.nn.LayerNorm(features)
       
    def forward(self, x):
        res = x  
        x = (self.layer_norm1(x.transpose(2,3))).transpose(2,3)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = (self.layer_norm2(x.transpose(2,3))).transpose(2,3)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += res
        return x 

class ASRModel(torch.nn.Module):
    def __init__(self, n_resnet_layers, n_rnn_layers, rnn_dim, features, n_class, dropout=0.15):
        super().__init__()
        features = features // 2
        # If it does not train properly change this 
        # to extract local features slowlier.
        self.cnn = torch.nn.Conv2d(1, 32, 3, stride=2, padding=1) # Extract local features
        self.rescnn_layers = torch.nn.Sequential(*[
            ResNet(32, 32, kernel=3, stride=1, dropout=dropout, features=features) 
            for _ in range(n_resnet_layers)
        ])

        self.fully_connected = torch.nn.Linear(features * 32, rnn_dim)

        self.birnn_layers = torch.nn.GRU(
            input_size=rnn_dim, hidden_size=rnn_dim,
            num_layers=n_rnn_layers, batch_first=True, bidirectional=True, dropout=dropout)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(rnn_dim * 2, rnn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(rnn_dim, n_class)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], -1) 
        x = x.transpose(1, 2) # put time up front
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x[0])
        return x


def save_model(model, path="models/asr_model.pth"):
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="models/asr_model.pth"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        print(f"No model found at {path}. Training from scratch.")


if __name__ == '__main__':
    # Usage example
    model = ASRModel(5, 5, 512, 40, 28)
    print(model(torch.randn((1, 1, 40, 2451))).shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The total number of params is : {total_params}')