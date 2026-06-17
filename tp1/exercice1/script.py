import cv2
img = cv2.imread('image_300x200.jpg')
if img is None:
 print("Erreur : image non trouvée")
else:
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 cv2.imshow('Image originale', img)
 cv2.imshow('Image en gris', gray)
 cv2.imwrite('gray.jpg', gray)
 cv2.waitKey(0)
 cv2.destroyAllWindows()