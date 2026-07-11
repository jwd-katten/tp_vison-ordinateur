import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from exercice2 import CNN

# 1. Préparation des données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# 2. Initialisation du modèle, de la loss et de l'optimiseur
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device utilisé: {device}")

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Boucle d'entraînement
nb_epoch = 10
loss_list = []
accuracy_list = []

for epoch in range(nb_epoch):
    model.train()
    running_loss = 0.0
    nb_batches = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Remettre les gradients à zéro
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        nb_batches += 1

    # Calcul de la loss moyenne
    train_loss = running_loss / nb_batches
    loss_list.append(train_loss)

    # Évaluation sur le jeu de test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    accuracy_list.append(accuracy)

    print(f"Epoch [{epoch+1}/{nb_epoch}] - Loss: {train_loss:.4f} - Accuracy: {accuracy:.4f}")

# 4. Sauvegarde du modèle
torch.save({
    'epoch': nb_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss_list': loss_list,
    'accuracy_list': accuracy_list,
}, "model_cnn.pth")

# 5. Courbes d'apprentissage
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_list, label='Loss entraînement')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Courbe de Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(accuracy_list, label='Accuracy test', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Courbe d\'Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("apprentissage_cnn.png")
plt.show()

print("Entraînement terminé !")
