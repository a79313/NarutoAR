import cv2
import mediapipe 
import numpy as np
import math
import json
from PIL import ImageSequence, Image

class NarutoAR:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        self.cap = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        


        self.gifs = [Image.open('assets/videos/rasengan-gif.gif'), Image.open('assets/videos/fireball-gif.gif'), Image.open('assets/videos/sharigan-gif.gif')]
        self.frames = [[frame.copy() for frame in ImageSequence.Iterator(self.gifs[0])],
                        [frame.copy() for frame in ImageSequence.Iterator(self.gifs[1])], [frame.copy() for frame in ImageSequence.Iterator(self.gifs[2])]]
        self.frame_idx = 0


        try:
            with open("snapshots.json", "r") as file:
                self.snapshots = json.load(file)
        except FileNotFoundError:
            print("Ficheiro 'snapshots.json' não encontrado. Nenhum gesto carregado.")

        # Posição anterior da mão
        self.prev_position = None


    def normalize_landmarks(self, hand_landmarks):
        """Normaliza os pontos de referência da mão."""
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        x_values = [lm[0] for lm in landmarks]
        y_values = [lm[1] for lm in landmarks]
        hand_size = max(max(x_values) - min(x_values), max(y_values) - min(y_values))
        hand_size = max(hand_size, 1e-6)
        normalized_landmarks = [(x / hand_size, y / hand_size, z / hand_size) for x, y, z in landmarks]
        return normalized_landmarks
        # MediaPipe Pose
        self.mp_pose = mediapipe.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mediapipe.solutions.drawing_utils  # Para desenhar as landmarks
        self.poses = self.load_poses('landmarks.json')


        #Imagens
        self.overlay_images = {
            "gaara" : cv2.imread("assets/images/gaara-pose.png", cv2.IMREAD_UNCHANGED),
            "lee" : cv2.imread("assets/images/lee-pose.png", cv2.IMREAD_UNCHANGED),
            "guy": cv2.imread("assets/images/guy.pose.png", cv2.IMREAD_UNCHANGED),
            "naruto": cv2.imread("assets/images/naruto-pose.png", cv2.IMREAD_UNCHANGED),
            "chidori": cv2.imread("assets/images/chidori-pose.png", cv2.IMREAD_UNCHANGED)


        }

    
    
    def load_poses(self, filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print('Poses file not found')
            return []
    def normalize_landmarks(self, landmarks):
        root = np.array([landmarks[0][0], landmarks[0][1], landmarks[0][2]])
        normalized_landmarks = [np.array([lm[0], lm[1], lm[2]]) - root for lm in landmarks]
        max_distance = max(np.linalg.norm(lm) for lm in normalized_landmarks if np.linalg.norm(lm) > 0)
        max_distance = max_distance if max_distance > 0 else 1e-6
        normalized_landmarks = [lm / max_distance for lm in normalized_landmarks]

        return normalized_landmarks

    def compare_pose(self, landmarks):
        normalized_landmarks = self.normalize_landmarks(landmarks)

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
            if distance < 0.25:  # Ajuste o valor conforme necessário
                return pose['name']

        
        return None


    

    def detect_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            # Desenhar landmarks
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # Obter landmarks detectados
            pose_landmarks = [
                (lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark
            ]

            # Comparar com poses salvas
            pose_name = self.compare_pose(pose_landmarks)

            
        return frame,pose_name


    

if __name__ == '__main__':
    naruto_ar = NarutoAR()

    while True:
        ret, frame = naruto_ar.cap.read()
        if not ret:
            break

        frame,pose_name = naruto_ar.detect_pose(frame)
        if pose_name:
            cv2.namedWindow('Characters', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Characters', 700, 700)
            cv2.imshow('Characters', naruto_ar.overlay_images[pose_name])
        

        cv2.imshow('Naruto AR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    naruto_ar.cap.release()
    cv2.destroyAllWindows()
