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
        self.previous_wrist_position = None  # Posição anterior do
        
    def load_background_image(self, background_path, frame_shape):
        # Carregar a imagem do fundo
        background = cv2.imread(background_path)
        
        if background is None:
            print("Erro: Não foi possível carregar a imagem de fundo. Verifique o caminho.")
            return None
        
        # Redimensionar o fundo para cobrir toda a tela (respeitando a altura e largura do frame)
        background_resized = cv2.resize(background, (frame_shape[1], frame_shape[0]))
        
        return background_resized
    
    def load_snapshots_from_file(self, file_path="snapshots.json"):
        """Carrega os gestos salvos no ficheiro JSON."""
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Ficheiro '{file_path}' não encontrado. Nenhum gesto carregado.")
            return []

    def normalize_landmarks(self, hand_landmarks):
        """Normaliza os landmarks em relação ao tamanho da mão."""
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        x_values = [lm[0] for lm in landmarks]
        y_values = [lm[1] for lm in landmarks]
        z_values = [lm[2] for lm in landmarks]

        # Calcula o tamanho da mão como a diferença máxima entre os pontos
        hand_size = max(max(x_values) - min(x_values), max(y_values) - min(y_values))
        hand_size = max(hand_size, 1e-6)  # Evitar divisão por zero

        # Normaliza os landmarks
        normalized_landmarks = [(x / hand_size, y / hand_size, z / hand_size) for x, y, z in landmarks]
        return normalized_landmarks

    def compare_with_snapshots(self, hand_landmarks, threshold=0.34):
        """Compara a posição atual dos landmarks com os gestos armazenados no arquivo JSON."""
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
                return snapshot["name"]  # Retorna o nome do gesto correspondente

        return None  # Nenhuma correspondência encontrada

    def detect_hand_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame, results
    
    def detect_jutsu(self, hand_landmarks):
        """Compara o movimento atual com os gestos salvos no arquivo .json"""
        gesture_name = self.compare_with_snapshots(hand_landmarks)
        if gesture_name:
            return f"Gesto Reconhecido: {gesture_name}"
        else:
            return "Nenhum gesto reconhecido"

    def overlay_rasengan(self, frame, hand_landmarks):
        """Sobrepõe o GIF do Rasengan na palma da mão."""
        # Aqui estamos pegando o primeiro landmark da mão, que é uma boa referência para a posição da palma
        palm_x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * frame.shape[1])
        palm_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * frame.shape[0])

        # Redimensiona o GIF do Rasengan para que caiba na palma da mão
        rasengan = self.frames[self.frame_idx]
        rasengan = rasengan.convert("RGBA")
        rasengan_resized = rasengan.resize((100, 100))  # Ajuste o tamanho conforme necessário

        # Converte o frame para formato RGBA
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_pil.paste(rasengan_resized, (palm_x - 50, palm_y - 50), rasengan_resized)  # Desloca para o centro da palma

        # Converte de volta para o formato OpenCV
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        # Atualiza o índice para a próxima frame do GIF
        self.frame_idx = (self.frame_idx + 1) % len(self.frames)
        
        return frame
    def overlay_fireball(self, frame, hand_landmarks):
        """Sobrepõe o GIF da Fireball na palma da mão."""
        # Aqui estamos pegando o primeiro landmark da mão, que é uma boa referência para a posição da palma
        palm_x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * frame.shape[1])
        palm_y = int(hand_landmarks.landmark[self.mp_hands
        .HandLandmark.WRIST].y * frame.shape[0])

        # Redimensiona o GIF da Fireball para que caiba na palma da mão
        fireball = self.fireball_frames[self.fireball_frame_idx]
        fireball = fireball.convert("RGBA")
        fireball_resized = fireball.resize((100, 100))  # Ajuste o tamanho conforme necessário

        # Converte o frame para formato RGBA
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_pil.paste(fireball_resized, (palm_x - 50, palm_y - 50), fireball_resized)  # Desloca para o centro da palma

        # Converte de volta para o formato OpenCV
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Atualiza o índice para a próxima frame do GIF
        self.fireball_frame_idx = (self.fireball_frame_idx + 1) % len(self.fireball_frames)

        return frame
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

    def extract_person_from_frame(self, frame, contour_coords):
        # Criando uma máscara binária da pessoa
        mask_person = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
        
        for contour in contour_coords:
            # Criando uma máscara para o contorno
            cv2.drawContours(mask_person, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Usando a máscara para copiar a pessoa da imagem original
        person_region = cv2.bitwise_and(frame, frame, mask=mask_person)
        return person_region, mask_person

    def create_clones(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processando a segmentação de fundo
        results = self.segmenter.process(frame_rgb)
        
        # Máscara binária da segmentação (pessoa vs fundo)
        mask = results.segmentation_mask
        
        # Criando a máscara binária com um limiar de 0.5 (só a pessoa)
        mask = mask > 0.5  # 1 para a pessoa e 0 para o fundo
        
        # Carregando e redimensionando o fundo comum
        common_background = self.load_background_image(self.background_images[0], frame.shape)

        # Verificar se o fundo foi carregado corretamente
        if common_background is None:
            print('Erro ao carregar o fundo comum.')
            return frame

        # Encontrar os contornos da máscara binária (onde há branco, ou seja, a pessoa)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Inicializando uma lista para armazenar as coordenadas dos contornos
        contour_coords = []

        # Para cada contorno encontrado, armazenamos as coordenadas
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filtra contornos pequenos
                contour_coords.append(contour)  # Guardando as coordenadas dos contornos

        # Extraindo a pessoa do frame (máscara da pessoa)
        person_region, person_mask = self.extract_person_from_frame(frame, contour_coords)

        # Dividindo a largura do frame em 3 partes
        clone_width = frame.shape[1] // 3
        clone_height = frame.shape[0]

        # Garantindo que a largura total seja coberta (evitando diferenças de pixels)
        common_background_sections = [
            common_background[:, 0:clone_width],                # Primeira seção do fundo
            common_background[:, clone_width:2*clone_width],   # Segunda seção do fundo
            common_background[:, 2*clone_width:frame.shape[1]] # Terceira seção do fundo (cobre o resto)
        ]

        # Criando as três partes da tela com a pessoa clonada sobre os fundos
        final_frame = np.zeros_like(frame)

        for i in range(3):
            # Calculando onde colocar a ROI (pessoa) na tela dividida
            x_start = i * clone_width
            x_end = (i + 1) * clone_width if i < 2 else frame.shape[1]  # Última seção cobre o restante

            # Redimensionando a pessoa para caber na largura de cada parte da tela
            clone_section = cv2.resize(person_region, (x_end - x_start, clone_height))

            # Misturando a pessoa com a respectiva seção do fundo comum
            blended_section = np.where(
                clone_section > 0,  # Onde há pessoa
                clone_section,      # Mostra a pessoa
                common_background_sections[i]  # Mostra o fundo
            )

            # Colocando a seção mesclada no frame final
            final_frame[:, x_start:x_end] = blended_section

        return final_frame


if __name__ == '__main__':
    naruto_ar = NarutoAR()

    # Verifique se a câmera foi aberta corretamente
    if not naruto_ar.cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        exit()

    while True:
        ret, frame = naruto_ar.cap.read()
        if not ret or frame is None:
            print("Erro: Não foi possível capturar o frame da câmera.")
            break

        # Detecta os landmarks
        frame, results = naruto_ar.detect_hand_landmarks(frame)
        if results and results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Detecta o gesto e mostra o nome
            jutsu = naruto_ar.detect_jutsu(hand_landmarks)
            cv2.putText(frame, jutsu, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if 'clones' in jutsu:
                frame = naruto_ar.create_clones(frame)

            # Se o gesto for "rasengan", ativa o efeito
            if "rasengan" in jutsu:
                print('Rasengan detectado')
                naruto_ar.rasengan_active = True

            if 'circle' in jutsu:
                print('Fireball detectado')
                naruto_ar.fireball_active = True

        # Se o Rasengan estiver ativo, sobrepõe o GIF
        if naruto_ar.rasengan_active and results and results.multi_hand_landmarks:
            frame = naruto_ar.overlay_rasengan(frame, results.multi_hand_landmarks[0])
        if naruto_ar.fireball_active and results and results.multi_hand_landmarks:
            frame = naruto_ar.overlay_fireball(frame, results.multi_hand_landmarks[0])


        # Exibe o frame
        try:
            cv2.imshow('Naruto AR', frame)
        except cv2.error as e:
            print(f"Erro ao exibir o frame: {e}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    naruto_ar.cap.release()
    cv2.destroyAllWindows()
