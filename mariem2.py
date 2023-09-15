import cv2
import numpy as np

# Charger l'image du rouge à lèvres (assurez-vous qu'elle n'a pas de canal alpha)
lipstick = cv2.imread('C:\\Users\\asus\\Desktop\\ss.png')

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturer une image depuis la webcam
    ret, frame = cap.read()

    # Assurez-vous que l'image du rouge à lèvres est de la même taille que le cadre de la webcam
    lipstick_resized = cv2.resize(lipstick, (frame.shape[1], frame.shape[0]))

    # Superposer le rouge à lèvres sur l'image de la webcam
    combined = cv2.addWeighted(frame, 1, lipstick_resized, 0.1, 0.2)

    # Afficher l'image résultante
    cv2.imshow('Webcam avec Rouge à Lèvres', combined)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la webcam et détruire la fenêtre OpenCV
cap.release()
cv2.destroyAllWindows()
