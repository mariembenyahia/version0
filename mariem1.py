import cv2

# Chargement du modèle de détection de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialisation de la capture vidéo à partir de la caméra
cap = cv2.VideoCapture(0)  # Utilisez 0 pour la caméra par défaut, ou spécifiez l'index de la caméra souhaitée.


# Variables pour activer/désactiver les filtres
show_original = True
show_blurred = False
show_thresholded = False

while True:
    # Capture une image depuis la caméra
    ret, frame = cap.read()


    # Convertit l'image en niveaux de gris pour la détection de visage
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Gestion des touches pressées
    x = cv2.waitKey(1) & 0xFF
    if x == ord('1'):
        show_original = True
        show_blurred = False
        show_thresholded = False
    elif x == ord('2'):
        show_original = True
        show_blurred = True
        show_thresholded = False
    
    elif x == ord('3'):
        show_original = True
        show_blurred = False
        show_thresholded = True
        

# Sortie de la boucle si la touche 'q' est enfoncée
    elif x == ord('q'):
        break
     # Applique un filtre de flou à l'image si show_blurred est True
    if show_blurred:
        frame = cv2.GaussianBlur(frame, (15, 15), 0)

    # Applique un filtre de seuillage à l'image (noir et blanc) si show_thresholded est True
    if show_thresholded:
        _, frame = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)


    # Affiche l'image en fonction des filtres activés/désactivés
    if show_original:
        cv2.imshow('Filtre en temps réel', frame)
    else:
        cv2.destroyWindow('Filtre en temps réel')
    

# Libère la ressource de la caméra et ferme les fenêtres d'affichage
cap.release()
cv2.destroyAllWindows()
