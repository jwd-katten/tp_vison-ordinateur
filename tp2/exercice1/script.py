import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Transformation des données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 2. Chargement du dataset CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# 3. Création des DataLoaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# 4. Classes du dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 5. Afficher quelques images
def imshow(img):
    img = img / 2 + 0.5  # Dénormaliser
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# Récupérer un batch d'images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Afficher les images
plt.figure(figsize=(12, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    imshow(images[i])
    plt.title(classes[labels[i]])
plt.tight_layout()
plt.savefig("exemples_cifar10.png")
plt.show()

print("Shape des images:", images.shape)
print("Nombre d'images d'entraînement:", len(trainset))
print("Nombre d'images de test:", len(testset))
