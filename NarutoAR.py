import cv2
import mediapipe as mp
import numpy as np

class NarutoAR:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # Images
        self.background_images = ['assets/images/background.png']
        self.background_image = None

    def find_hands_area(self, frame):
        x_min, y_min, x_max, y_max = frame.shape[1], frame.shape[0], 0, 0
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                x_coords = [landmark.x * frame_width for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * frame_height for landmark in hand_landmarks.landmark]

                x_min = min(x_min, int(min(x_coords)))
                y_min = min(y_min, int(min(y_coords)))
                x_max = max(x_max, int(max(x_coords)))
                y_max = max(y_max, int(max(y_coords)))

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return frame
    
    def detect_hand_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame, results
    
    def detect_jutsu(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        
        if landmarks:
            # Coordenadas normalizadas (x, y, z) para cada ponto
            thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
            
            # Exemplos de regras simples para identificar gestos:
            # 1. Tigre: Todos os dedos retos e juntos
            if (abs(index_tip.x - middle_tip.x) < 0.02 and 
                abs(middle_tip.x - ring_tip.x) < 0.02 and 
                abs(ring_tip.x - pinky_tip.x) < 0.02 and 
                thumb_tip.y > index_tip.y):
                return "Tigre (Tiger)"
            
            # 2. Cobra: Todos os dedos retos, mas polegar afastado
            if (abs(index_tip.x - middle_tip.x) < 0.02 and 
                abs(middle_tip.x - ring_tip.x) < 0.02 and 
                abs(ring_tip.x - pinky_tip.x) < 0.02 and 
                thumb_tip.y < index_tip.y):
                return "Cobra (Snake)"
            
            # 3. Rato: Indicador dobrado, outros dedos retos
            if (index_tip.y > middle_tip.y and 
                middle_tip.y < ring_tip.y and 
                ring_tip.y < pinky_tip.y):
                return "Rato (Rat)"
            
            # 4. Pássaro: Dedo médio cruzado sobre o indicador
            if (middle_tip.x < index_tip.x and 
                abs(ring_tip.x - pinky_tip.x) < 0.02):
                return "Pássaro (Bird)"
            
            # 5. Dragão: Formar triângulo com polegar e indicador
            if (abs(thumb_tip.y - index_tip.y) < 0.05 and 
                abs(middle_tip.x - ring_tip.x) > 0.1 and 
                abs(ring_tip.x - pinky_tip.x) > 0.1):
                return "Dragão (Dragon)"
        
        return "Nenhum jutsu reconhecido"

    def replace_background(self, frame):
        if self.background_image is None:
            self.background_image = cv2.imread(self.background_images[0])
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(frame_rgb)

        # Cria a máscara binária (pessoa vs. fundo)
        mask = results.segmentation_mask > 0.5
        background_resized = cv2.resize(self.background_image, (frame.shape[1], frame.shape[0]))

        # Aplica a máscara no frame
        output_image = np.where(mask[..., None], frame, background_resized)

        return output_image

if __name__ == '__main__':
    naruto_ar = NarutoAR()

    while True:
        ret, frame = naruto_ar.cap.read()
        if not ret:
            break

        frame, results = naruto_ar.detect_hand_landmarks(frame)
        if results and results.multi_hand_landmarks:
            jutsu = naruto_ar.detect_jutsu(results.multi_hand_landmarks[0])
            cv2.putText(frame, jutsu, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Naruto AR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    naruto_ar.cap.release()
    cv2.destroyAllWindows()
