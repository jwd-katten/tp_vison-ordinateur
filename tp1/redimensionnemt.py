import cv2

# 1. Charger l'image depuis ton dossier
img = cv2.imread("maroc.jpg")

# 2. Vérifier si l'image est bien chargée
if img is None:
    print("Erreur : maroc.jpg introuvable ou illisible")
    exit()

# 3. Redimensionner en 300x200
resized = cv2.resize(img, (300, 200))

# 4. Sauvegarder
cv2.imwrite("image_300x200.jpg", resized)

print("Image créée : image_300x200.jpg")