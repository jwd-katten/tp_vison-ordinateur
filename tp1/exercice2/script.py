import cv2
import matplotlib.pyplot as plt

# Charger image
img = cv2.imread('image_300x200.jpg')
if img is None:
 print("Erreur")
else:
    # Niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Filtrage
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Contours
    edges = cv2.Canny(blur, 100, 200)
    # Histogramme
    plt.hist(gray.ravel(), 256, [0,256])
    plt.title("Histogramme")
    plt.savefig("histogramme.png")
    
    # Affichage
    cv2.imshow('Original', img)
    cv2.imshow('Gris', gray)
    cv2.imshow('Flou', blur)
    cv2.imshow('Contours', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()