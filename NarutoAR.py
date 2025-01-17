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

        self.poses = self.load_poses('pose-landmarks.json')
        self.hand_snapshots = self.load_hand_snapshots('hand_snapshots.json')

        self.pose_is_locked = False

        # Images
        self.overlay_images = {
            "gaara": cv2.imread("assets/images/gaara-pose.png", cv2.IMREAD_UNCHANGED),
            "lee": cv2.imread("assets/images/lee-pose.png", cv2.IMREAD_UNCHANGED),
            "guy": cv2.imread("assets/images/guy.pose.png", cv2.IMREAD_UNCHANGED),
            "naruto": cv2.imread("assets/images/naruto-pose.png", cv2.IMREAD_UNCHANGED),
            "chidori": cv2.imread("assets/images/chidori-pose.png", cv2.IMREAD_UNCHANGED)
        }

        self.gifs = [Image.open('assets/videos/rasengan-gif.gif'), Image.open('assets/videos/fireball-gif.gif'), Image.open('assets/videos/sharigan-gif.gif')]
        self.frames = [[frame.copy() for frame in ImageSequence.Iterator(self.gifs[0])],
                       [frame.copy() for frame in ImageSequence.Iterator(self.gifs[1])], [frame.copy() for frame in ImageSequence.Iterator(self.gifs[2])]]
        self.frame_idx = 0

        try:
            with open("hand_snapshots.json", "r") as file:
                self.hand_snapshots = {snap['name']: snap['landmarks'] for snap in json.load(file)}
        except FileNotFoundError:
            print("File 'hand_snapshots.json' not found. No gestures loaded.")

        # Previous hand position
        self.prev_position = None

        # Timer to prevent pose switching
        self.last_detected_pose = None
        self.last_pose_time = 0

    def normalize_hand_landmarks(self, hand_landmarks):
        """Normaliza os pontos de referência da mão."""
        if not hand_landmarks:
            return []
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        x_values = [lm[0] for lm in landmarks]
        y_values = [lm[1] for lm in landmarks]
        hand_size = max(max(x_values) - min(x_values), max(y_values) - min(y_values))
        hand_size = max(hand_size, 1e-6)
        normalized_landmarks = [(x / hand_size, y / hand_size, z / hand_size) for x, y, z in landmarks]
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

    def compare_hand_signs(self, normalized_landmarks, threshold=0.35):
        """Compara os pontos de referência da mão com os gestos salvos."""
        for name, snapshot_landmarks in self.hand_snapshots.items():
            match = True
            for i, (x, y, z) in enumerate(normalized_landmarks):
                snapshot_x, snapshot_y, snapshot_z = snapshot_landmarks[i]
                dist = np.linalg.norm(np.array([x - snapshot_x, y - snapshot_y, z - snapshot_z]))
                if dist > threshold:
                    match = False
                    break
            if match:
                return name
        return None

    def detect_jutsu(self, hand_landmarks):
        """Detecta o jutsu realizado com base nos landmarks normalizados."""
        if hand_landmarks:
            normalized_landmarks = self.normalize_hand_landmarks(hand_landmarks)
            return self.compare_hand_signs(normalized_landmarks)
        return None

    def detect_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # Get detected landmarks
            pose_landmarks = [
                (lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark
            ]

            # Compare with saved poses
            pose_name = self.compare_pose(pose_landmarks)

            
            
            return frame, pose_name

        return frame, self.last_detected_pose

    def normalize_pose_landmarks(self, landmarks):
        root = np.array([landmarks[0][0], landmarks[0][1], landmarks[0][2]])
        normalized_landmarks = [np.array([lm[0], lm[1], lm[2]]) - root for lm in landmarks]
        max_distance = max(np.linalg.norm(lm) for lm in normalized_landmarks if np.linalg.norm(lm) > 0)
        max_distance = max_distance if max_distance > 0 else 1e-6
        normalized_landmarks = [lm / max_distance for lm in normalized_landmarks]

        return normalized_landmarks

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
            if distance < 0.:  # Ajuste o valor conforme necessário
                return pose['name']

        return None
    
    def detect_hand_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            return results.multi_hand_landmarks  # Returns list of hand landmarks
        return []

# Inside your main loop



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

if __name__ == '__main__':
    naruto_ar = NarutoAR()
    aura_image = None
    while True:
        ret, frame = naruto_ar.cap.read()
        if not ret:
            break

        # Process hand landmarks
        frame, pose_name = naruto_ar.detect_pose(frame)
        if pose_name is not None:
            naruto_ar.pose_is_locked = True
            if naruto_ar.pose_is_locked is True:
                hand_landmarks_list = naruto_ar.detect_hand_landmarks(frame)
                for hand_landmarks in hand_landmarks_list:
                    jutsu = naruto_ar.detect_jutsu(hand_landmarks)
                    if jutsu == 'clones':
                        aura_image = naruto_ar.create_aura_mask(naruto_ar.overlay_images[pose_name])
                        time.sleep(3)
                        naruto_ar.pose_is_locked = False
                        naruto_ar.last_detected_pose = None
                        naruto_ar.last_pose_time = 0
                        pose_name = None
                
            
            cv2.namedWindow('Characters', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Characters', 500, 500)
            if aura_image is not None:
                cv2.imshow('Characters', aura_image)
            else:
                cv2.imshow('Characters', naruto_ar.overlay_images[pose_name])

        # Show the frame
        cv2.imshow('Naruto AR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    naruto_ar.cap.release()
    cv2.destroyAllWindows()
