import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from exercice2 import CNN

# 1. Chargement des données de test
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Charger le modèle entraîné
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

checkpoint = torch.load("model_cnn.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. Évaluation sur le jeu de test
all_preds = []
all_labels = []

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 4. Rapport de classification
print("Rapport de classification:")
print(classification_report(all_labels, all_preds, target_names=classes))

# 5. Matrice de confusion
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Matrice de Confusion')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig("matrice_confusion.png")
plt.show()

# 6. Visualiser des prédictions
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

plt.figure(figsize=(12, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    img = images[i].cpu() / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    color = 'green' if predicted[i] == labels[i] else 'red'
    plt.title(f"Prédit: {classes[predicted[i]]}\nRéel: {classes[labels[i]]}",
              color=color, fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.savefig("predictions.png")
plt.show()

# 7. Accuracy globale
correct = sum(p == l for p, l in zip(all_preds, all_labels))
total = len(all_labels)
print(f"\nAccuracy globale: {correct}/{total} = {correct/total:.4f}")
