import torch

class AudiomentationsWrapper(torch.nn.Module):
    def __init__(self, transform, sample_rate):
        super().__init__()
        self.transform = transform
        self.sample_rate = sample_rate

    def forward(self, x):        
        augmented_x = self.transform(x.numpy(), self.sample_rate)
        return torch.from_numpy(augmented_x)