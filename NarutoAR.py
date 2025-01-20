import cv2
import mediapipe as mp
import numpy as np
import math
import json
from PIL import ImageSequence, Image
import time
from ultralytics import YOLO




class NarutoAR:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
                                                            #MEDIAPIPE
        ###################################################MediaPipe Hands############################################################
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        ############################################### MediaPipe Selfie Segmentation ##############################################
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        ######################################### MediaPipe Face Detection ###################################################################
        self.mp_face_detection = mp.solutions.face_detection
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
            42: 'assets/images/kunai.png',
        }
        self.loaded_images = {cls: cv2.imread(path, cv2.IMREAD_UNCHANGED) for cls, path in self.object_images.items()}
        self.overlay_images = {
            "gaara": cv2.imread("assets/images/gaara-pose.png", cv2.IMREAD_UNCHANGED),
            "lee": cv2.imread("assets/images/lee-pose.png", cv2.IMREAD_UNCHANGED),
            "guy": cv2.imread("assets/images/guy.pose.png", cv2.IMREAD_UNCHANGED),
            "naruto": cv2.imread("assets/images/naruto-pose.png", cv2.IMREAD_UNCHANGED),
            "chidori": cv2.imread("assets/images/chidori-pose.png", cv2.IMREAD_UNCHANGED),
            "clones": cv2.imread("assets/images/naruto-clones-pose.png", cv2.IMREAD_UNCHANGED)
        }
        self.overlay_clones_images = {
            "gaara": cv2.imread("assets/images/gaara-pose-clones.png", cv2.IMREAD_UNCHANGED),
            "lee": cv2.imread("assets/images/lee-pose-clones.png", cv2.IMREAD_UNCHANGED),
            "guy": cv2.imread("assets/images/guy.pose-clones.png", cv2.IMREAD_UNCHANGED),
            "naruto": cv2.imread("assets/images/naruto-clones-pose.png", cv2.IMREAD_UNCHANGED),
            "chidori": cv2.imread("assets/images/chidori-pose-clones.png", cv2.IMREAD_UNCHANGED)
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
            # Converter landmarks para listas de [x, y, z]
            formatted_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            normalized_landmarks = self.normalize_landmarks(formatted_landmarks)
            return normalized_landmarks
        return None

    
    def remove_alpha_channel_from_image(self, image):
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

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
    def create_aura_mask(self, image, color):
        """Cria uma máscara onde somente as bordas do PNG são consideradas."""
        if image.shape[2] != 4:
            print("Erro: A imagem precisa de um canal alfa.")
            return image

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_smooth = cv2.GaussianBlur(image_gray, (5, 5), 0)
        edges = cv2.Canny(image_smooth, 50, 100)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, contours, -1, color, thickness=10)
        aura = cv2.GaussianBlur(mask, (25, 25), 0)

        return aura

    def apply_aura(self, characters_frame, hand_sign):
        """Aplica a aura azul ao redor do personagem."""
        if characters_frame is not None:
            if characters_frame.shape[2] != 4:
                characters_frame = cv2.cvtColor(characters_frame, cv2.COLOR_BGR2BGRA)
            if hand_sign == "fireball" or hand_sign == "circle":
                aura = self.create_aura_mask(characters_frame, (0, 0, 255))
            else:

                aura = self.create_aura_mask(characters_frame, (255, 0, 0))
            combined = cv2.addWeighted(characters_frame, 1, aura, 0.5, 0)
            return combined
        return characters_frame  
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
    current_mode = 'poses'
    current_pose_name = None

    while True:
        ret, frame = naruto_ar.cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        # Alternar entre modos
        if key == ord('o'):
            current_mode = "objects" if current_mode != "objects" else "none"
            print(f"Modo atual: {'Deteção de Objetos' if current_mode == 'objects' else 'Nenhum'}")
        elif key == ord('p'):
            current_mode = "poses" if current_mode != "poses" else "none"
            print(f"Modo atual: {'Deteção de Poses' if current_mode == 'poses' else 'Nenhum'}")
        elif key == ord('h'):
            current_mode = "hands" if current_mode != "hands" else "none"
            print(f"Modo atual: {'Deteção de Mãos' if current_mode == 'hands' else 'Nenhum'}")
        # Processar o modo atual
        if current_mode == "objects":
            results = naruto_ar.model(frame)

            if results:
                for result in results[0].boxes:
                    x1, y1, x2, y2 = map(int, result.xyxy[0])  # Coordenadas da caixa delimitadora
                    cls = int(result.cls[0])  # Classe detectada
                    conf = result.conf[0]  # Confiança da detecção

                    print(f"Objeto detectado: Classe {cls}, Confiança: {conf:.2f}")

                    if cls in naruto_ar.object_images:
                        object_img = naruto_ar.loaded_images.get(cls)
                        if object_img is not None:
                            print(f"Classe {cls} encontrada no dicionário. Carregando imagem...")
                            if object_img.shape[2] == 4:
                                object_img = cv2.cvtColor(object_img, cv2.COLOR_BGRA2BGR)

                            characters_frame = resize_image_to_fit(object_img, window_width, window_height)
                        else:
                            print(f"Erro: Imagem do objeto para classe {cls} não carregada.")
                    else:
                        print(f"Classe {cls} não encontrada no dicionário de imagens.")

        elif current_mode == "poses":
            frame, pose_results = naruto_ar.detect_pose_landmarks(frame)
            if pose_results and pose_results.pose_landmarks:
                pose_name = naruto_ar.detect_and_compare_pose(pose_results, pose_snapshots)
                #pose_name = "guy"
                if pose_name in naruto_ar.overlay_images:
                    overlay_image = naruto_ar.overlay_images[pose_name]
                    current_pose_name = pose_name
                    if overlay_image is not None:
                        # Redimensionar para a janela Characters
                        characters_frame = resize_image_to_fit(overlay_image, window_width, window_height)
                    else:
                        print(f"Erro: Imagem para '{pose_name}' não carregada.")
        elif current_mode == "hands":
            previous_hand_sign = None
            frame, hands_results = naruto_ar.detect_hand_landmarks(frame)
            if hands_results and hands_results.multi_hand_landmarks:
                hand_landmarks = hands_results.multi_hand_landmarks[0]
                hand_landmarks = naruto_ar.detect_jutsu(hand_landmarks.landmark)
                hand_sign = None  # Inicie com None para evitar valores antigos
                if hand_landmarks:
                    hand_sign = naruto_ar.compare_with_snapshots(hand_landmarks, hand_snapshots)

                
                if hand_sign == "clones":
                    previous_hand_sign = "clones"
                    overlay_image = naruto_ar.overlay_clones_images[pose_name]
                    if overlay_image is not None:
                        # Redimensionar para a janela Characters
                        characters_frame = resize_image_to_fit(overlay_image, window_width, window_height)
                    else:
                        print(f"Erro: Imagem para '{hand_sign}' não carregada.")
                elif hand_sign == "fist" and previous_hand_sign != "fist":
                    characters_frame = resize_image_to_fit(overlay_image, window_width, window_height)
                    previous_hand_sign = "fist"
                    characters_frame = naruto_ar.apply_aura(characters_frame, hand_sign)
                elif hand_sign == "fireball" or hand_sign == "circle" and previous_hand_sign != "fireballandcircle":
                    characters_frame = resize_image_to_fit(overlay_image, window_width, window_height)

                    previous_hand_sign = "fireballandcircle"
                    characters_frame = naruto_ar.apply_aura(characters_frame, hand_sign)

                    

                    
        # Exibir na janela Characters
        if characters_frame is not None:
            cv2.imshow('Characters', characters_frame)

        # Exibir o feed principal
        cv2.imshow('Naruto AR', frame)

        # Fechar o programa ao pressionar 'Q'
        if key == ord('q'):
            break

    naruto_ar.cap.release()
    cv2.destroyAllWindows()
