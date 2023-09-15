import cv2

# Chargement du modèle de détection de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialisation de la capture vidéo à partir de la caméra
cap = cv2.VideoCapture(0)  # Utilisez 0 pour la caméra par défaut, ou spécifiez l'index de la caméra souhaitée.

while True:
    # Capture une image depuis la caméra
    ret, frame = cap.read()

    # Convertit l'image en niveaux de gris pour la détection de visage
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecte les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Applique un filtre de flou à l'image
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

    # Applique un filtre de détection de contours à l'image
    edges_frame = cv2.Canny(frame, 100, 200)

    # Convertit l'image en niveaux de gris pour le filtre de seuillage
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applique un filtre de seuillage à l'image (noir et blanc)
    _, thresholded_frame = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Dessine un rectangle autour de chaque visage détecté
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Affiche les images avec les filtres appliqués et les visages encadrés
    cv2.imshow('Original', frame)
    cv2.imshow('Blurred', blurred_frame)
    cv2.imshow('Edges', edges_frame)
    cv2.imshow('Thresholded', thresholded_frame)

    # Sortie de la boucle si la touche 'q' est enfoncée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libère la ressource de la caméra et ferme les fenêtres d'affichage
cap.release()
cv2.destroyAllWindows()