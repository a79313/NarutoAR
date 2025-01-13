import cv2
import mediapipe as mp
import numpy as np
import json
from PIL import Image, ImageSequence

class NarutoAR:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # Carregar o GIF do Rasengan
        self.rasengan_gif = Image.open('assets/videos/rasengan-gif.gif')  # Substitua pelo caminho do seu GIF
        self.frames = [frame.copy() for frame in ImageSequence.Iterator(self.rasengan_gif)]  # Armazenando as frames do GIF
        self.frame_idx = 0  # Índice da frame do GIF a ser mostrado
        
        # Background
        self.background_images = ['assets/images/background.png']
        self.background_image = None

        # Carregar o GIF da Fireball
        self.fireball_gif = Image.open('assets/videos/fireball-gif.gif')  # Substitua pelo caminho do seu GIF
        self.fireball_frames = [frame.copy() for frame in ImageSequence.Iterator(self.fireball_gif)]  # Armazenando as frames do GIF
        self.fireball_frame_idx = 0  # Índice da frame do GIF a ser mostrado
        self.fireball_active = False  # Flag para verificar se a Fireball está ativa
        
        # Carregar os gestos salvos
        self.snapshots = self.load_snapshots_from_file()

        # Flag para verificar se o Rasengan está ativo
        self.rasengan_active = False
        self.rasengan_position = None  # Posição do Rasengan na tela
        self.rasengan_direction = None  # Direção do Rasengan
        self.rasengan_speed = 15  # Velocidade do movimento do Rasengan

    def load_background_image(self, background_path, frame_shape):
        background = cv2.imread(background_path)
        if background is None:
            print("Erro: Não foi possível carregar a imagem de fundo. Verifique o caminho.")
            return None
        background_resized = cv2.resize(background, (frame_shape[1], frame_shape[0]))
        return background_resized

    def load_snapshots_from_file(self, file_path="snapshots.json"):
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Ficheiro '{file_path}' não encontrado. Nenhum gesto carregado.")
            return []

    def normalize_landmarks(self, hand_landmarks):
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        x_values = [lm[0] for lm in landmarks]
        y_values = [lm[1] for lm in landmarks]
        z_values = [lm[2] for lm in landmarks]

        hand_size = max(max(x_values) - min(x_values), max(y_values) - min(y_values))
        hand_size = max(hand_size, 1e-6)

        normalized_landmarks = [(x / hand_size, y / hand_size, z / hand_size) for x, y, z in landmarks]
        return normalized_landmarks

    def compare_with_snapshots(self, hand_landmarks, threshold=0.34):
        current_landmarks = self.normalize_landmarks(hand_landmarks)

        for snapshot in self.snapshots:
            match = True
            for i, (x, y, z) in enumerate(current_landmarks):
                snapshot_x, snapshot_y, snapshot_z = snapshot["landmarks"][i]
                dist = np.linalg.norm(np.array([x - snapshot_x, y - snapshot_y, z - snapshot_z]))
                if dist > threshold:
                    match = False
                    break
            if match:
                return snapshot["name"]

        return None

    def detect_hand_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame, results
    
    def detect_jutsu(self, hand_landmarks):
        gesture_name = self.compare_with_snapshots(hand_landmarks)
        if gesture_name:
            return f"Gesto Reconhecido: {gesture_name}"
        else:
            return "Nenhum gesto reconhecido"

    def overlay_rasengan(self, frame, hand_landmarks):
        palm_x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * frame.shape[1])
        palm_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * frame.shape[0])

        rasengan = self.frames[self.frame_idx]
        rasengan = rasengan.convert("RGBA")
        rasengan_resized = rasengan.resize((100, 100))

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_pil.paste(rasengan_resized, (palm_x - 50, palm_y - 50), rasengan_resized)

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        self.frame_idx = (self.frame_idx + 1) % len(self.frames)
        
        return frame

    def overlay_fireball(self, frame, hand_landmarks):
        palm_x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * frame.shape[1])
        palm_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * frame.shape[0])

        fireball = self.fireball_frames[self.fireball_frame_idx]
        fireball = fireball.convert("RGBA")
        fireball_resized = fireball.resize((100, 100))

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_pil.paste(fireball_resized, (palm_x - 50, palm_y - 50), fireball_resized)

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        self.fireball_frame_idx = (self.fireball_frame_idx + 1) % len(self.fireball_frames)
        return frame

    def replace_background(self, frame):
        if self.background_image is None:
            self.background_image = cv2.imread(self.background_images[0])
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(frame_rgb)

        mask = results.segmentation_mask > 0.5
        background_resized = cv2.resize(self.background_image, (frame.shape[1], frame.shape[0]))

        output_image = np.where(mask[..., None], frame, background_resized)

        return output_image

    def create_clones(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(frame_rgb)

        mask = results.segmentation_mask
        mask = mask > 0.5

        common_background = self.load_background_image(self.background_images[0], frame.shape)

        if common_background is None:
            print('Erro ao carregar o fundo comum.')
            return frame

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        person_region = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))
        
        clone_width = frame.shape[1] // 3
        clone_height = frame.shape[0]

        common_background_sections = [
            common_background[:, 0:clone_width],
            common_background[:, clone_width:2*clone_width],
            common_background[:, 2*clone_width:frame.shape[1]]
        ]

        final_frame = np.zeros_like(frame)

        for i in range(3):
            x_start = i * clone_width
            x_end = (i + 1) * clone_width if i < 2 else frame.shape[1]

            clone_section = cv2.resize(person_region, (x_end - x_start, clone_height))

            blended_section = np.where(
                clone_section > 0,
                clone_section,
                common_background_sections[i]
            )

            final_frame[:, x_start:x_end] = blended_section

        return final_frame

if __name__ == '__main__':
    naruto_ar = NarutoAR()

    if not naruto_ar.cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        exit()

    while True:
        ret, frame = naruto_ar.cap.read()
        if not ret or frame is None:
            print("Erro: Não foi possível capturar o frame da câmera.")
            break

        original_frame = frame.copy()
        frame, results = naruto_ar.detect_hand_landmarks(frame)
        
        if results and results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            jutsu = naruto_ar.detect_jutsu(hand_landmarks)
            cv2.putText(frame, jutsu, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if 'clones' in jutsu:
                frame = naruto_ar.create_clones(frame)

            if "rasengan" in jutsu:
                naruto_ar.rasengan_active = True

            if 'circle' in jutsu:
                naruto_ar.fireball_active = True

        if naruto_ar.rasengan_active and results and results.multi_hand_landmarks:
            frame = naruto_ar.overlay_rasengan(frame, results.multi_hand_landmarks[0])
        if naruto_ar.fireball_active and results and results.multi_hand_landmarks:
            frame = naruto_ar.overlay_fireball(frame, results.multi_hand_landmarks[0])

        try:
            frame = naruto_ar.replace_background(frame)
            cv2.imshow('Naruto AR - Efeitos e Jutsus', frame)
            cv2.imshow('Naruto AR - Feed Original', original_frame)
        except cv2.error as e:
            print(f"Erro ao exibir o frame: {e}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    naruto_ar.cap.release()
    cv2.destroyAllWindows()
