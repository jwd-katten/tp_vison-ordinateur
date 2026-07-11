import torch
import torch.nn as nn
import torch.nn.functional as F

# Définition du réseau CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Couches de convolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Couches de pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Couches fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Dropout pour régularisation
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Bloc de convolution 1
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        # Bloc de convolution 2
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        # Bloc de convolution 3
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4

        # Aplatir pour les couches fully connected
        x = x.view(-1, 128 * 4 * 4)

        # Couches fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# Test du modèle
if __name__ == "__main__":
    model = CNN()
    print(model)

    # Vérifier le nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nNombre total de paramètres: {total_params:,}")

    # Test avec un batch fictif
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"\nShape de sortie: {output.shape}")
