import torch
import torch.nn as nn

class Connector(nn.Module):
    """Projection layer mapping student feature space to CLIP embedding space."""
    def __init__(self, input_dim=2048, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def get_connector(input_dim=2048, output_dim=512, device='cuda'):
    connector = Connector(input_dim=input_dim, output_dim=output_dim)
    return connector.to(device)