import cv2
import mediapipe
import numpy as np
import math
from ultralytics import YOLO
import time

class NarutoAR:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)


        
        # MediaPipe Pose
        self.mp_pose = mediapipe.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mediapipe.solutions.drawing_utils  # Para desenhar as landmarks

        # Modelo YOLO
        self.model = YOLO('models/yolo11s.pt')

        # Imagem da pose
        self.pose_images = ['assets/images/naruto-pose.png', 'assets/images/gaara-pose.png','assets/images/lee-pose.png','assets/images/guy-pose.png', 'assets/images/chidori-pose.png']
        self.pose_image = cv2.imread(self.pose_images[0], cv2.IMREAD_UNCHANGED)

        #Videos para overlay
        self.videos = [cv2.VideoCapture('assets/videos/naruto-gif.gif'), cv2.VideoCapture('assets/videos/gaara-gif.gif'), cv2.VideoCapture('assets/videos/lee-gif.gif'), cv2.VideoCapture('assets/videos/guy-gif.gif'), cv2.VideoCapture('assets/videos/chidori-gif.gif')]
        
        self.video = self.videos[0]

        #Video Normalizing
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1 / self.fps


    def calculate_distance(self, p1, p2):
        """
        Calcula a distância euclidiana entre dois pontos (p1, p2).
        """
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

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
            cords = self.detect_person_bounding_box(frame)
            # Desenhar as landmarks da pose
            #self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            if self.is_naruto_pose(results.pose_landmarks):
                self.video = self.videos[0]
                if not self.video.isOpened():
                    print("Erro: Arquivo de vídeo não aberto.")
                    return frame

                # Resetar para o primeiro quadro do GIF, se necessário
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

                start_time = time.time()  # Marcar o tempo inicial

                while True:
                    ret_vid, frame_vid = self.video.read()
                    if ret_vid:
                        # Redimensionar o quadro do GIF para corresponder ao tamanho do quadro da câmera
                        frame_vid = cv2.resize(frame_vid, (frame.shape[1], frame.shape[0]))

                        # Mesclar o quadro do GIF com o da câmera
                        alpha = 0.6  # Ajuste de transparência
                        blended_frame = cv2.addWeighted(frame_vid, alpha, frame, 1 - alpha, 0)

                        # Exibir o quadro combinado
                        cv2.imshow('Naruto AR', blended_frame)

                        # Aguardar o tempo necessário para sincronizar com o FPS do GIF
                        elapsed_time = time.time() - start_time
                        time_to_wait = max(0, self.frame_time - elapsed_time)
                        time.sleep(time_to_wait)

                        start_time = time.time()  # Atualizar o tempo inicial

                        # Interromper o loop ao pressionar 'q'
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("Aviso: Não foi possível ler o quadro do GIF. Reiniciando o vídeo.")
                        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break

                cv2.destroyAllWindows()
            elif self.is_guy_sensei_thumbs_up(results.pose_landmarks):
                self.video = self.videos[3]
                # Verificar se o vídeo foi carregado corretamente
                if not self.video.isOpened():
                    print("Erro: Arquivo de vídeo não aberto.")
                    return frame

                # Resetar para o primeiro quadro do GIF, se necessário
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

                start_time = time.time()  # Marcar o tempo inicial

                while True:
                    ret_vid, frame_vid = self.video.read()
                    if ret_vid:
                        # Redimensionar o quadro do GIF para corresponder ao tamanho do quadro da câmera
                        frame_vid = cv2.resize(frame_vid, (frame.shape[1], frame.shape[0]))

                        # Mesclar o quadro do GIF com o da câmera
                        alpha = 0.6  # Ajuste de transparência
                        blended_frame = cv2.addWeighted(frame_vid, alpha, frame, 1 - alpha, 0)

                        # Exibir o quadro combinado
                        cv2.imshow('Naruto AR', blended_frame)

                        # Aguardar o tempo necessário para sincronizar com o FPS do GIF
                        elapsed_time = time.time() - start_time
                        time_to_wait = max(0, self.frame_time - elapsed_time)
                        time.sleep(time_to_wait)

                        start_time = time.time()  # Atualizar o tempo inicial

                        # Interromper o loop ao pressionar 'q'
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("Aviso: Não foi possível ler o quadro do GIF. Reiniciando o vídeo.")
                        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break

                cv2.destroyAllWindows()
            elif self.is_crossed_arms_pose(results.pose_landmarks):
                self.video = self.videos[1]
                if not self.video.isOpened():
                    print("Erro: Arquivo de vídeo não aberto.")
                    return frame

                # Resetar para o primeiro quadro do GIF, se necessário
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

                start_time = time.time()  # Marcar o tempo inicial

                while True:
                    ret_vid, frame_vid = self.video.read()
                    if ret_vid:
                        # Redimensionar o quadro do GIF para corresponder ao tamanho do quadro da câmera
                        frame_vid = cv2.resize(frame_vid, (frame.shape[1], frame.shape[0]))

                        # Mesclar o quadro do GIF com o da câmera
                        alpha = 0.6  # Ajuste de transparência
                        blended_frame = cv2.addWeighted(frame_vid, alpha, frame, 1 - alpha, 0)

                        # Exibir o quadro combinado
                        cv2.imshow('Naruto AR', blended_frame)

                        # Aguardar o tempo necessário para sincronizar com o FPS do GIF
                        elapsed_time = time.time() - start_time
                        time_to_wait = max(0, self.frame_time - elapsed_time)
                        time.sleep(time_to_wait)

                        start_time = time.time()  # Atualizar o tempo inicial

                        # Interromper o loop ao pressionar 'q'
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("Aviso: Não foi possível ler o quadro do GIF. Reiniciando o vídeo.")
                        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break

                cv2.destroyAllWindows()

            

            elif self.is_chidori_pose(results.pose_landmarks):
            
               # Verificar se o vídeo foi carregado corretamente
                self.video = self.videos[4]
                if not self.video.isOpened():
                    print("Erro: Arquivo de vídeo não aberto.")
                    return frame

                # Resetar para o primeiro quadro do GIF, se necessário
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

                start_time = time.time()  # Marcar o tempo inicial

                while True:
                    ret_vid, frame_vid = self.video.read()
                    if ret_vid:
                        # Redimensionar o quadro do GIF para corresponder ao tamanho do quadro da câmera
                        frame_vid = cv2.resize(frame_vid, (frame.shape[1], frame.shape[0]))

                        # Mesclar o quadro do GIF com o da câmera
                        alpha = 0.8  # Ajuste de transparência
                        blended_frame = cv2.addWeighted(frame_vid, alpha, frame, 1 - alpha, 0)

                        # Exibir o quadro combinado
                        cv2.imshow('Naruto AR', blended_frame)

                        # Aguardar o tempo necessário para sincronizar com o FPS do GIF
                        elapsed_time = time.time() - start_time
                        time_to_wait = max(0, self.frame_time - elapsed_time)
                        time.sleep(time_to_wait)

                        start_time = time.time()  # Atualizar o tempo inicial

                        # Interromper o loop ao pressionar 'q'
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        
                        break

                cv2.destroyAllWindows()
            elif self.is_lee_pose(results.pose_landmarks):
                self.video = self.videos[2]
                if not self.video.isOpened():
                    print("Erro: Arquivo de vídeo não aberto.")
                    return frame

                # Resetar para o primeiro quadro do GIF, se necessário
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

                start_time = time.time()  # Marcar o tempo inicial

                while True:
                    ret_vid, frame_vid = self.video.read()
                    if ret_vid:
                        # Redimensionar o quadro do GIF para corresponder ao tamanho do quadro da câmera
                        frame_vid = cv2.resize(frame_vid, (frame.shape[1], frame.shape[0]))

                        # Mesclar o quadro do GIF com o da câmera
                        alpha = 0.6  # Ajuste de transparência
                        blended_frame = cv2.addWeighted(frame_vid, alpha, frame, 1 - alpha, 0)

                        # Exibir o quadro combinado
                        cv2.imshow('Naruto AR', blended_frame)

                        # Aguardar o tempo necessário para sincronizar com o FPS do GIF
                        elapsed_time = time.time() - start_time
                        time_to_wait = max(0, self.frame_time - elapsed_time)
                        time.sleep(time_to_wait)

                        start_time = time.time()  # Atualizar o tempo inicial

                        # Interromper o loop ao pressionar 'q'
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("Aviso: Não foi possível ler o quadro do GIF. Reiniciando o vídeo.")
                        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break

                cv2.destroyAllWindows()
        return frame

    def detect_person_bounding_box(self, frame):
        results = self.model.predict(frame)
        best_box = None
        best_confidence = 0

        for result in results:
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    if cls == 0 and conf > best_confidence:  # Apenas pessoas com confiança alta
                        best_confidence = conf
                        best_box = (int(x1), int(y1), int(x2), int(y2))

        if best_box:
            x1, y1, x2, y2 = best_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return x1, y1, x2, y2
        return None, None, None, None

    def resize_image(self, image, cords):
        x1, y1, x2, y2 = cords
        height, width = int(y2 - y1), int(x2 - x1)
        image_height, image_width = image.shape[:2]

        scale = min(width / image_width, height / image_height)
        new_width = int(image_width * scale)
        new_height = int(image_height * scale)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image

    def overlay_pose_image(self, frame, pose_image, cords):
        x1, y1, x2, y2 = cords
        overlay = cv2.resize(pose_image, (x2 - x1, y2 - y1))

        # Separar os canais de cor e alfa (se disponível)
        if overlay.shape[2] == 4:  # Verificar se há canal alfa
            b, g, r, alpha = cv2.split(overlay)
            alpha = alpha / 255.0

            for c in range(0, 3):  # Mesclar os canais BGR
                frame[y1:y2, x1:x2, c] = frame[y1:y2, x1:x2, c] * (1 - alpha) + overlay[:, :, c] * alpha
        else:
            frame[y1:y2, x1:x2] = overlay

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
