import cv2
import mediapipe 
import numpy as np
import math

class NarutoAR:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        # MediaPipe Pose
        self.mp_pose = mediapipe.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mediapipe.solutions.drawing_utils  # Para desenhar as landmarks

    def calculate_distance(self, p1, p2):
        """
        Calcula a distância euclidiana entre dois pontos (p1, p2).
        """
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def is_naruto_pose(self, pose_landmarks):
        """
        Detecta a pose do Naruto com base nas posições relativas dos ombros, cotovelos, pulsos
        e a posição do torso.
        """
        # Obter as coordenadas dos pontos chave
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        # Obter as coordenadas do torso
        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Cálculo das distâncias para garantir que a pose está correta
        left_shoulder_elbow_distance = self.calculate_distance(
            (left_shoulder.x, left_shoulder.y),
            (left_elbow.x, left_elbow.y)
        )
        
        left_elbow_wrist_distance = self.calculate_distance(
            (left_elbow.x, left_elbow.y),
            (left_wrist.x, left_wrist.y)
        )
        
        right_shoulder_elbow_distance = self.calculate_distance(
            (right_shoulder.x, right_shoulder.y),
            (right_elbow.x, right_elbow.y)
        )
        
        right_elbow_wrist_distance = self.calculate_distance(
            (right_elbow.x, right_elbow.y),
            (right_wrist.x, right_wrist.y)
        )

        # Cálculo da distância entre os quadris
        torso_distance = self.calculate_distance(
            (left_hip.x, left_hip.y),
            (right_hip.x, right_hip.y)
        )

        # Critério de detecção da pose do Naruto
        # O braço esquerdo deve estar esticado (ombro -> cotovelo -> pulso)
        # E o braço direito deve estar dobrado, ou seja, cotovelo mais próximo do corpo
        # O torso deve estar reto, e a distância entre os quadris deve ser significativa
        if (left_shoulder_elbow_distance > left_elbow_wrist_distance and
            right_shoulder_elbow_distance < right_elbow_wrist_distance and
            torso_distance > 0.2):  # Ajuste o valor conforme necessário
            return True
        
        return False


    def is_crossed_arms_pose(self, pose_landmarks):
        """
        Detecta a pose de braços cruzados do Gaara.
        """
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]

        # Calcular distâncias para verificar braços cruzados
        left_wrist_right_elbow_distance = self.calculate_distance(
            (left_wrist.x, left_wrist.y),
            (right_elbow.x, right_elbow.y)
        )
        
        right_wrist_left_elbow_distance = self.calculate_distance(
            (right_wrist.x, right_wrist.y),
            (left_elbow.x, left_elbow.y)
        )

        # Limite de distância para braços cruzados
        distance_threshold = 0.15  
        if left_wrist_right_elbow_distance < distance_threshold and right_wrist_left_elbow_distance < distance_threshold:
            return True
        
        return False

    def is_lee_pose(self, pose_landmarks):
        """
        Detecta a pose do Lee com as mãos sobre os ombros e o joelho levantado.
        """
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        left_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]

        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]

        # Verificar se as mãos estão acima dos ombros e joelhos levantados
        hands_above_shoulders = (left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y)
        left_knee_raised = left_knee.y < left_hip.y
        right_knee_raised = right_knee.y < right_hip.y

        if hands_above_shoulders and (left_knee_raised or right_knee_raised):
            return True
        
        return False

    def is_chidori_pose(self, pose_landmarks):
        """
        Detecta a pose do Chidori, onde a mão esquerda segura o pulso direito.
        """
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calcular a distância entre a mão esquerda e o pulso direito
        left_wrist_right_pulse_distance = self.calculate_distance(
            (left_wrist.x, left_wrist.y),
            (right_wrist.x, right_wrist.y)
        )

        chidori_threshold = 0.2  # Limite para a detecção da pose
        if left_wrist_right_pulse_distance < chidori_threshold:
            return True
        
        return False

    def is_guy_sensei_thumbs_up(self, pose_landmarks):
        """
        Detecta a pose de 'thumbs up' de Guy Sensei.
        """
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        right_thumb = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB]
        
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        
        right_wrist_elbow_distance = self.calculate_distance(
            (right_wrist.x, right_wrist.y),
            (right_elbow.x, right_elbow.y)
        )

        thumb_angle = math.degrees(math.atan2(right_thumb.y - right_wrist.y, right_thumb.x - right_wrist.x))
        
        nose_wrist_distance = self.calculate_distance(
            (nose.x, nose.y),
            (right_wrist.x, right_wrist.y)
        )
        
        if thumb_angle < 45 and nose_wrist_distance < 0.3 and right_wrist_elbow_distance < 0.3:
            return True
        
        return False

    def detect_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)  # Processar o quadro com o Pose Detector

        if results.pose_landmarks:
            # Desenhar as landmarks da pose
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Priorizar a detecção das poses
            if self.is_naruto_pose(results.pose_landmarks):
                cv2.putText(frame, 'Pose Naruto Detectada!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif self.is_crossed_arms_pose(results.pose_landmarks):
                cv2.putText(frame, 'Pose Gaara Detectada!', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif self.is_chidori_pose(results.pose_landmarks):
                cv2.putText(frame, 'Pose Chidori Detectada!', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif self.is_guy_sensei_thumbs_up(results.pose_landmarks):
                cv2.putText(frame, 'Pose Guy Sensei (Thumbs Up) Detectada!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            
            elif self.is_lee_pose(results.pose_landmarks):
                cv2.putText(frame, 'Pose Lee (Karate Kid) Detectada!', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        
        return frame

    def process_frame(self, frame):
        frame = self.detect_pose(frame)
        return frame

if __name__ == '__main__':
    naruto_ar = NarutoAR()

    while True:
        ret, frame = naruto_ar.cap.read()
        if not ret:
            break

        frame = naruto_ar.process_frame(frame)

        cv2.imshow('Naruto AR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    naruto_ar.cap.release()
    cv2.destroyAllWindows()
