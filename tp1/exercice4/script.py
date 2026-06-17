import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Charger une image couleur
img = cv2.imread('image_300x200.jpg')

if img is None:
    print("Erreur : Impossible de charger l'image")
    exit()

# 2. Convertir en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Détecter les contours
edges = cv2.Canny(gray, 100, 200) # 100 = seuil faible, 200 = seuil fort
plt.imshow(edges, cmap='gray')
plt.title("Contours détectés")
plt.axis('off')
plt.savefig("contours.png")
plt.show()

# 4. histogramme
plt.hist(gray.ravel(), 256, [0, 256])
plt.title("Histogramme")
plt.xlabel("Valeur de pixel")
plt.ylabel("Fréquence")
plt.savefig("histogramme.png")
plt.show()

# to be continued...