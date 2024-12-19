import cv2
import numpy as np
import pickle

def load_and_extract_contours(gesture_image_path):
    # Carregar a imagem do gesto (certifique-se de que a imagem é clara e bem definida)
    gesture_image = cv2.imread(gesture_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Aplicar o threshold para binarizar a imagem (transforma em preto e branco)
    _, thresh = cv2.threshold(gesture_image, 127, 255, cv2.THRESH_BINARY)
    
    # Caso necessário, aplique um desfoque para reduzir ruído
    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)

    # Encontrar os contornos da imagem
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

# Exemplo de uso
gesture_contours = load_and_extract_contours('assets/images/tiger_mudra.png')

# Visualização (opcional, apenas para depuração)
gesture_image = cv2.imread('assets/images/tiger_mudra.png')
with open('gesture_contours.pkl', 'wb') as f:
    pickle.dump(gesture_contours, f)
cv2.drawContours(gesture_image, gesture_contours, -1, (0, 255, 0), 3)
cv2.imshow('Contornos do Gestos', gesture_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
