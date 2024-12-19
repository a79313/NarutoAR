import cv2
import mediapipe
import numpy as np
import pickle

class NarutoAR:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # MediaPipe
        self.mp_hands = mediapipe.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mediapipe.solutions.drawing_utils
        self.mp_selfie_segmentation = mediapipe.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # Images
        self.background_images = ['assets/images/background.png']
        self.background_image = None

    def compare_contours(self, contours):
        """Compara os contornos detectados com os contornos salvos de gestos específicos"""
        with open('gesture_contours.pkl', 'rb') as f:
            gesture_contours = pickle.load(f)
        
        for contour in contours:
            for gesture_contour in gesture_contours:
                # Ajuste no limiar de comparação para reduzir falsos positivos
                score = cv2.matchShapes(contour, gesture_contour, cv2.CONTOURS_MATCH_I1, 0)
                print("Score de comparação de contornos:", score)  # Imprime o valor de comparação para depuração
                if score < 1:  # Limite ajustado para melhorar a precisão
                    return True
        return False

    def detect_hand_contours(self, frame):
        """Detecta os contornos das mãos a partir dos pontos de referência do MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        contours = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Cria uma imagem em preto e branco para detectar os contornos
                blank_image = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

                # Extrai as coordenadas de cada landmark e desenha a mão
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(blank_image, (x, y), 5, 255, -1)

                # Encontra os contornos na imagem binarizada
                contours_found, _ = cv2.findContours(blank_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filtra contornos com base na área (a área mínima dos contornos pode ser ajustada)
                contours_found = [cnt for cnt in contours_found if cv2.contourArea(cnt) > 500]

                # Adiciona os contornos encontrados à lista
                contours.extend(contours_found)

        return contours

    def replace_background(self, frame):
        """Substitui o fundo da imagem com a segmentação da pessoa"""
        if self.background_image is None:
            self.background_image = cv2.imread(self.background_images[0])

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(frame_rgb)

        # Cria a máscara binária (pessoa vs. fundo)
        mask = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
        mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)

        background_resized = cv2.resize(self.background_image, (frame.shape[1], frame.shape[0]))

        # Aplica a máscara no frame
        output_image = np.uint8(mask * frame + (1 - mask) * background_resized)

        return output_image

if __name__ == '__main__':
    naruto_ar = NarutoAR()

    while True:
        ret, frame = naruto_ar.cap.read()
        if not ret:
            break

        # Detecta os contornos das mãos
        hand_contours = naruto_ar.detect_hand_contours(frame)
        
        # Verifica se algum gesto corresponde ao gesto salvo
        if naruto_ar.compare_contours(hand_contours):
            # Se o gesto for detectado, substitui o fundo
            frame = naruto_ar.replace_background(frame)

        # Exibe a imagem com ou sem o fundo substituído
        cv2.imshow('Naruto AR', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    naruto_ar.cap.release()
    cv2.destroyAllWindows()
