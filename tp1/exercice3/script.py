import cv2

# Charger image
img = cv2.imread('image_300x200.jpg')
if img is None:
 print("Erreur")
else:
    # conversion en niveau de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Filtrage médian (réduction du bruit)
    median = cv2.medianBlur(gray, 5)
    #redimensionnement à 320x240
    resized = cv2.resize(median, (320, 240))
    # sauvegarde
    cv2.imwrite('resized.jpg', resized)
    # Affichage
    cv2.imshow('Original', img)
    cv2.imshow('Gris', gray)
    cv2.imshow('Median', median)
    cv2.imshow('Redimensionne', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()