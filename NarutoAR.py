import cv2
import mediapipe 
import numpy as np
from ultralytics import YOLO
from pprint import pprint
import os


class NarutoAR:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

                                                            #MEDIAPIPE
        ###################################################MediaPipe Hands############################################################
        self.mp_hands = mediapipe.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mediapipe.solutions.drawing_utils
        
        ############################################### MediaPipe Selfie Segmentation ##############################################
        self.mp_selfie_segmentation = mediapipe.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        ######################################### MediaPipe Face Detection ###################################################################
        self.mp_face_detection = mediapipe.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)
                                                            #YOLO
        self.model = YOLO('models/yolo11s.pt')
        # Images
        self.background_images = ['assets/images/background.png']
        self.background_image = None
        self.object_images = {
            76: 'assets/images/shuriken.png',
            45: 'assets/images/bowl.png',
            16: 'assets/images/dog.png',
            24: 'assets/images/backpack.png',
            43: 'assets/images/kunai.png',

        }
        self.loaded_images = {cls: cv2.imread(path, cv2.IMREAD_UNCHANGED) for cls, path in self.object_images.items()}


    
    def remove_alpha_channel_from_image(self, image):
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def detect_hand_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame, results
    
    def detect_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                self.mp_drawing.draw_detection(frame, detection)
        return frame


    def resize_object(self, object_img, x1, y1 ,x2 ,y2):
        width, height = x2 - x1, y2 - y1
        if object_img.shape[2] == 4:
            alpha = object_img[:, :, 3] / 255.0
            cords = cv2.findNonZero(alpha)
            if cords is not None:
                x, y, w, h = cv2.boundingRect(cords)
                object_img = object_img[y:y + h, x:x + w]  # Crop the object to remove transparency
        resized_object = cv2.resize(object_img, (width, height))
        return resized_object
    def apply_object_in_frame(self, roi, object_img):
        # Verifique se o tamanho da imagem do objeto e a ROI são compatíveis
        object_resized = cv2.resize(object_img, (roi.shape[1], roi.shape[0]))

        # Aplicar a imagem do objeto na ROI
        for i in range(3):  # Trabalhando com os 3 canais de cor (R, G, B)
            roi[:, :, i] = np.where(object_resized[:, :, 0] == 0, roi[:, :, i], object_resized[:, :, i])

        return roi

    def detect_objects(self, frame):
        # Realizar a detecção com YOLO
        results = self.model(frame)

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Coordenadas da caixa delimitadora
            conf = result.conf[0]  # Confiança da detecção
            cls = int(result.cls[0])  # Classe detectada

            # Verificar se a classe está no dicionário
            if cls in self.object_images:
                roi = frame[y1:y2, x1:x2]
                object_img = self.loaded_images.get(cls)

                # Verificar se a imagem foi carregada corretamente
                if object_img is not None and object_img.shape[2] == 4:
                    # Redimensionar e aplicar o objeto na ROI
                    resized_object = self.resize_object(object_img, x1, y1, x2, y2)
                    roi = self.apply_object_in_frame(roi, resized_object)

                    # Atualizar o frame com o ROI processado
                    frame[y1:y2, x1:x2] = roi

                # Adicionar rótulo e caixa delimitadora
                label = self.model.names[cls]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame



    
    
    def replace_background(self, frame):
        if self.background_image is None:
            self.background_image = cv2.imread(self.background_images[0])
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(frame_rgb)

        # Create binary mask (person vs. background)
        mask = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
        mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
        
        background_resized = cv2.resize(self.background_image, (frame.shape[1], frame.shape[0]))

        # Apply mask on frame
        output_image = np.uint8(mask * frame + (1 - mask) * background_resized)

        return output_image

if __name__ == '__main__':
    naruto_ar = NarutoAR()

    while True:
        ret, frame = naruto_ar.cap.read()
        if not ret:
            break

      

        frame = naruto_ar.detect_objects(frame)

        cv2.imshow('Naruto AR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
         
    naruto_ar.cap.release()
    cv2.destroyAllWindows()
