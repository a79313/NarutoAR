import cv2
import mediapipe as mp
import numpy as np
import math
import json
from PIL import ImageSequence, Image
import time




class NarutoAR:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.overlay_images = {
            "gaara": cv2.imread("assets/images/gaara-pose.png", cv2.IMREAD_UNCHANGED),
            "lee": cv2.imread("assets/images/lee-pose.png", cv2.IMREAD_UNCHANGED),
            "guy": cv2.imread("assets/images/guy.pose.png", cv2.IMREAD_UNCHANGED),
            "naruto": cv2.imread("assets/images/naruto-pose.png", cv2.IMREAD_UNCHANGED),
            "chidori": cv2.imread("assets/images/chidori-pose.png", cv2.IMREAD_UNCHANGED)
        }

        self.gifs = [Image.open('assets/videos/rasengan-gif.gif'), Image.open('assets/videos/fireball-gif.gif'), Image.open('assets/videos/sharigan-gif.gif'), Image.open('assets/videos/chidori.gif')]
        self.frames = [
            [frame.copy() for frame in ImageSequence.Iterator(self.gifs[0])],
            [frame.copy() for frame in ImageSequence.Iterator(self.gifs[1])],
            [frame.copy() for frame in ImageSequence.Iterator(self.gifs[2])],
            [frame.copy() for frame in ImageSequence.Iterator(self.gifs[3])]
        ]
        self.frame_idx = 0

        
        

    
    def normalize_landmarks(self, landmarks):
        """Normaliza landmarks em relação ao tamanho da mão ou corpo."""
        x_values = [lm[0] for lm in landmarks]
        y_values = [lm[1] for lm in landmarks]

        # Calcula o tamanho como a diferença máxima entre os pontos
        size = max(max(x_values) - min(x_values), max(y_values) - min(y_values))
        size = max(size, 1e-6)  # Evitar divisão por zero

        # Normaliza os landmarks
        normalized_landmarks = [(x / size, y / size, z / size) for x, y, z in landmarks]
        return normalized_landmarks
    def load_hand_snapshots(self, filename):
        try:
            with open(filename, 'r') as f:
                return {snap['name']: snap['landmarks'] for snap in json.load(f)}
        except FileNotFoundError:
            print('Hand snapshots file not found')
            return {}

    def load_poses(self, filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print('Poses file not found')
            return []

    def compare_with_snapshots(self, landmarks, snapshots, threshold=0.5):
        """Compara landmarks normalizados com os snapshots carregados."""
        normalized_landmarks = self.normalize_landmarks([(lm.x, lm.y, lm.z) for lm in landmarks])

        for snapshot in snapshots:
            match = True
            for (x, y, z), (sx, sy, sz) in zip(normalized_landmarks, snapshot["landmarks"]):
                dist = np.linalg.norm(np.array([x - sx, y - sy, z - sz]))
                if dist > threshold:
                    match = False
                    break
            if match:
                print(f"Correspondência encontrada: {snapshot['name']}")
                return snapshot["name"]
        return None

    def detect_jutsu(self, hand_landmarks):
        """Detecta o jutsu realizado com base nos landmarks normalizados."""
        if hand_landmarks:
            normalized_landmarks = self.normalize_landmarks(hand_landmarks)
            return self.compare_hand_signs(normalized_landmarks)
        return None

    def detect_hand_landmarks(self, frame):
        """Detecta landmarks das mãos e desenha sobre o frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame, results

    def detect_pose_landmarks(self, frame):
        """Detecta landmarks do corpo e desenha sobre o frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame, results

    

    def compare_pose(self, landmarks):
        normalized_landmarks = self.normalize_pose_landmarks(landmarks)

        for pose in self.poses:
            if 'name' not in pose or 'landmarks' not in pose:
                continue  # Ignora poses malformadas

            saved_landmarks = np.array(pose['landmarks'])
            if len(normalized_landmarks) != len(saved_landmarks):
                continue

            # Soma das distâncias quadradas
            distance = np.sum(
                [
                    np.linalg.norm(np.array(normalized_landmarks[i]) - saved_landmarks[i]) ** 2
                    for i in range(len(normalized_landmarks))
                ]
            )

            # Threshold reduzido
            if distance < 0.3:  # Ajuste o valor conforme necessário
                return pose['name']

        return None
    
    

# Inside your main loop

    def overlay_chidori(self, frame, coordinates):
        """Sobrepor a animação do Chidori na posição fornecida."""
        if not self.gifs[3]:
            print("Erro: Animação de Chidori não carregada.")
            return frame

        # Obter o quadro atual da animação
        chidori_frame = self.frames[3][self.frame_idx]

        # Converter o quadro para o formato necessário (se for PIL Image)
        chidori_frame = np.array(chidori_frame)
        chidori_frame = cv2.cvtColor(chidori_frame, cv2.COLOR_RGBA2BGRA)

        # Coordenadas de sobreposição
        x, y = coordinates
        h, w = chidori_frame.shape[:2]

        # Garantir que as coordenadas estejam dentro dos limites do frame principal
        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            print("Erro: Coordenadas da sobreposição estão fora dos limites do quadro principal.")
            return frame

        # Combinar o Chidori com o frame usando transparência
        for i in range(h):
            for j in range(w):
                if chidori_frame[i, j, 3] != 0:  # Verificar canal alfa
                    frame[y + i, x + j] = chidori_frame[i, j, :3]

        # Atualizar o índice do quadro para animação
        self.frame_idx = (self.frame_idx + 1) % len(self.frames[3])

        return frame

    def create_aura_mask(self, image):
        """Cria uma máscara onde somente as partes coloridas do PNG são consideradas."""
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]

        # Criar uma máscara a partir do canal alfa
        _, binary_mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

        # Encontrar contornos na máscara
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Criar uma imagem para desenhar os contornos da aura
        aura = np.zeros_like(bgr)

        # Desenhar contornos com uma cor azul brilhante
        cv2.drawContours(aura, contours, -1, (255, 0, 0), thickness=10)

        # Aplicar desfoque para suavizar a aura
        aura = cv2.GaussianBlur(aura, (25, 25), 0)

        # Combinar a aura com a imagem original
        aura_with_mask = cv2.addWeighted(bgr, 1, aura, 0.5, 0)

        return aura_with_mask

    def load_snapshots_from_file(self, filename):
        """Carrega os snapshots de um arquivo JSON."""
        try:
            with open(filename, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Arquivo '{filename}' não encontrado. Nenhum snapshot carregado.")
            return []
    def compare_with_snapshots(self, landmarks, snapshots, threshold=0.4):
        """Compara landmarks normalizados com os snapshots carregados."""
        normalized_landmarks = self.normalize_landmarks(landmarks)

        for snapshot in snapshots:
            # Verifica se o snapshot contém o campo 'name' e 'landmarks'
            if 'name' not in snapshot or 'landmarks' not in snapshot:
                continue  # Ignora entradas malformadas

            match = True
            for (x, y, z), (sx, sy, sz) in zip(normalized_landmarks, snapshot["landmarks"]):
                dist = np.linalg.norm(np.array([x - sx, y - sy, z - sz]))
                if dist > threshold:
                    match = False
                    break
            if match:
                detected_name = snapshot['name']
                print(f"Correspondência encontrada: {detected_name}")
                return detected_name
        return None
    def detect_and_compare_pose(self, pose_results, pose_snapshots, threshold=0.4):
        """
        Detecta e compara landmarks da pose com snapshots.
        """
        if pose_results and pose_results.pose_landmarks:
            # Extrai landmarks da pose
            pose_landmarks = [(lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks.landmark]
            # Compara landmarks com os snapshots
            pose_name = self.compare_with_snapshots(pose_landmarks, pose_snapshots, threshold)
            if pose_name:
                print(f"Pose detected: {pose_name}")
            return pose_name
        return None
    
def resize_image_to_fit(image, window_width, window_height):
    """Resize the image to fit within the given window dimensions while maintaining aspect ratio."""
    h, w = image.shape[:2]
    scale = min(window_width / w, window_height / h)  # Scale based on the smaller dimension
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

if __name__ == '__main__':
    naruto_ar = NarutoAR()
    
    window_width, window_height = 500, 500  # Set window dimensions
    cv2.namedWindow('Characters', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Characters', window_width, window_height)
    hand_snapshots = naruto_ar.load_snapshots_from_file('hand_snapshots.json')
    pose_snapshots = naruto_ar.load_snapshots_from_file('pose-landmarks.json')
    characters_frame = None
    while True:
        ret, frame = naruto_ar.cap.read()
        if not ret:
            break

        # Detecta os landmarks da pose
        cv2.imshow('Naruto AR', frame)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            frame, pose_results = naruto_ar.detect_pose_landmarks(frame)
        # Verifica se resultados existem e passa para o método
            if pose_results and pose_results.pose_landmarks:
                pose_name = naruto_ar.detect_and_compare_pose(pose_results, pose_snapshots)
                pose_name = "guy"
            
                overlay_image = naruto_ar.overlay_images[pose_name]
                if overlay_image is not None:
                    overlay_image = resize_image_to_fit(overlay_image, window_width, window_height)
                    characters_frame = overlay_image
                else:
                    print(f"Erro: Imagem para '{pose_name}' não carregada.")
            
        if cv2.waitKey(1) & 0xFF == ord('o'):
            
        # Mostra o frame
        if characters_frame is not None:
            cv2.imshow('Characters', characters_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    naruto_ar.cap.release()
    cv2.destroyAllWindows()


