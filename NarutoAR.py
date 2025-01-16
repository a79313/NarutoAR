import cv2
import mediapipe as mp
import numpy as np
import json
from PIL import Image, ImageSequence
from pprint import pprint

class NarutoARv2:
  def __init__(self):
      self.cap = cv2.VideoCapture(0)

      # Inicialização do MediaPipe
      self.mp_hands = mp.solutions.hands
      self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
      self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
      self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
      self.mp_face_mesh = mp.solutions.face_mesh
      self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)
      
      self.iris_indices = {
            'left': [468, 469, 470, 471],  # Índices da íris esquerda
            'right': [473, 474, 475, 476]  # Índices da íris direita
        }

  
      # Carregamento dos GIFs
      self.gifs = [Image.open('assets/videos/rasengan-gif.gif'), Image.open('assets/videos/fireball-gif.gif'), Image.open('assets/videos/sharigan-gif.gif')]
      self.frames = [[frame.copy() for frame in ImageSequence.Iterator(self.gifs[0])],
                      [frame.copy() for frame in ImageSequence.Iterator(self.gifs[1])], [frame.copy() for frame in ImageSequence.Iterator(self.gifs[2])]]
      self.frame_idx = 0

      # Imagem de fundo
      self.background_images = ['assets/images/background.png']
      self.background_image = None

      # Carregamento dos gestos
      try:
          with open("snapshots.json", "r") as file:
              self.snapshots = json.load(file)
      except FileNotFoundError:
          print("Ficheiro 'snapshots.json' não encontrado. Nenhum gesto carregado.")

      # Posição anterior da mão
      self.prev_position = None

  

  def get_eye_landmarks(self, face_landmarks, frame):
    """Obtém as coordenadas dos olhos a partir dos marcos faciais."""
    h, w = frame.shape[:2]
    left_eye_indices = [33, 130]  # Índices aproximados para o olho esquerdo
    right_eye_indices = [362, 260]  # Índices aproximados para o olho direito

    left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]
    right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]

    return left_eye, right_eye

  def overlay_sharingan(self, frame, eye_position):
      """Sobrepõe o frame atual do GIF do Sharingan na posição do olho."""
      eye_x, eye_y = eye_position
      sharingan_frame = self.frames[2][self.frame_idx]
      self.frame_idx = (self.frame_idx + 1) % len(self.frames[2])

      # Redimensiona o frame do Sharingan para se ajustar ao tamanho do olho
      sharingan_resized = sharingan_frame.resize((20, 20))  # Ajusta o tamanho conforme necessário

      # Converte o frame atual para RGBA
      frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')

      # Calcula a posição para sobrepor o Sharingan
      position = (eye_x - sharingan_resized.width // 2 + 13, eye_y - sharingan_resized.height // 2) 
      sharingan_resized = sharingan_resized.convert('RGBA')
      frame_pil.paste(sharingan_resized, position, sharingan_resized)

      return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)

  def process_face_mesh(self, frame):
      
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      return self.face_mesh.process(rgb_frame)
     





  
  def load_background_image(self, background_image_path, frame_shape):
      """Carrega e redimensiona a imagem de fundo."""
      background = cv2.imread(background_image_path)
      if background is None:
          print("Erro: Não foi possível carregar a imagem de fundo. Verifique o caminho.")
          return None
      background_resized = cv2.resize(background, (frame_shape[1], frame_shape[0]))
      return background_resized

  def normalize_landmarks(self, hand_landmarks):
      """Normaliza os pontos de referência da mão."""
      landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
      x_values = [lm[0] for lm in landmarks]
      y_values = [lm[1] for lm in landmarks]
      hand_size = max(max(x_values) - min(x_values), max(y_values) - min(y_values))
      hand_size = max(hand_size, 1e-6)
      normalized_landmarks = [(x / hand_size, y / hand_size, z / hand_size) for x, y, z in landmarks]
      return normalized_landmarks

  def compare_with_snapshots(self, hand_landmarks, threshold=0.35):
      """Compara os pontos de referência da mão com os gestos salvos."""
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
      """Detecta os pontos de referência da mão no frame."""
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = self.hands.process(rgb_frame)
      return frame, results

  def detect_jutsu(self, hand_landmarks):
      """Detecta o gesto realizado pelos pontos de referência da mão."""
      gesture_name = self.compare_with_snapshots(hand_landmarks)
      if gesture_name:
          return f"Gesto Reconhecido: {gesture_name}"
      else:
          return "Nenhum gesto reconhecido"

  def calcular_velocidade_mao(self, hand_landmarks, frame_shape):
      """Calcula a velocidade da mão com base na posição atual e anterior."""
      current_position = (
          int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * frame_shape[1]),
          int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * frame_shape[0])
      )
      if self.prev_position is None:
          self.prev_position = current_position
          return 0, current_position
      velocidade = np.linalg.norm(np.array(current_position) - np.array(self.prev_position))
      self.prev_position = current_position
      return velocidade, current_position

  def overlay_rasengan(self, frame, hand_landmarks):
      """Sobrepõe o Rasengan na mão com brilho dinâmico."""
      velocidade, current_position = self.calcular_velocidade_mao(hand_landmarks, frame.shape)
      brilho = min(1.0, velocidade * 5)  # Ajuste o fator conforme necessário

      rasengan = self.frames[0][self.frame_idx].convert("RGBA")
      rasengan_resized = rasengan.resize((100, 100))
      alpha = rasengan_resized.split()[3].point(lambda p: p * brilho)
      rasengan_resized.putalpha(alpha)

      frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      frame_pil.paste(rasengan_resized, (current_position[0] - 50, current_position[1] - 50), rasengan_resized)
      frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

      self.frame_idx = (self.frame_idx + 1) % len(self.frames[0])
      return frame

  def overlay_fireball(self, frame, hand_landmarks):
      """Sobrepõe a Bola de Fogo na mão."""
      palm_x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * frame.shape[1])
      palm_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * frame.shape[0])

      fireball = self.frames[1][self.frame_idx].convert("RGBA")
      fireball_resized = fireball.resize((100, 100))

      frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      frame_pil.paste(fireball_resized, (palm_x - 50, palm_y - 50), fireball_resized)
      frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

      self.frame_idx = (self.frame_idx + 1) % len(self.frames[1])
      return frame

  def replace_background(self, frame):
    """Substitui o fundo do frame."""
    if self.background_image is None:
        self.background_image = self.load_background_image(self.background_images[0], frame.shape)

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
    
  def overlay_chakra_aura(self, frame, mask, frame_idx):
    """Cria um efeito de aura semelhante ao chakra do Naruto."""
    aura_color = (255, 0, 0)  # Cor azul em formato BGR
    aura_thickness = 15 + int(5 * np.sin(frame_idx * 0.1))  # Espessura pulsante
    blur_size = 25  # Tamanho do desfoque para suavizar o contorno
    glowing_intensity = 0.6 + 0.4 * np.sin(frame_idx * 0.1)  # Intensidade variável

    # Converte a máscara para uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Detecta contornos na máscara
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Cria uma camada transparente para desenhar a aura
    aura_layer = np.zeros_like(frame)

    # Desenha os contornos dilatados para criar o efeito de aura
    for contour in contours:
        cv2.drawContours(aura_layer, [contour], -1, aura_color, thickness=aura_thickness)

    # Aplica um desfoque para suavizar os contornos da aura
    aura_layer = cv2.GaussianBlur(aura_layer, (blur_size, blur_size), 0)

    # Combina o frame original com a camada de aura usando intensidade variável
    combined_frame = cv2.addWeighted(frame, 1, aura_layer, glowing_intensity, 0)

    return combined_frame


if __name__ == '__main__':
  naruto_arv2 = NarutoARv2()

  while True:
    ret , frame = naruto_arv2.cap.read()
    copied_frame = frame.copy()
    if not ret:
      print("Erro ao capturar o frame.")
      break
    
    
    frame, results = naruto_arv2.detect_hand_landmarks(frame)
    if results.multi_hand_landmarks:
      hand_landmarks = results.multi_hand_landmarks[0]
      jutsu = naruto_arv2.detect_jutsu(hand_landmarks)
      if 'clones' in jutsu:
        copied_frame = naruto_arv2.create_clones(frame)
      if 'fireball' in jutsu:
        copied_frame = naruto_arv2.overlay_fireball(frame, hand_landmarks)
      if 'rasengan' in jutsu:
        copied_frame = naruto_arv2.overlay_rasengan(frame, hand_landmarks)
      if 'fist' in jutsu:
        face_landmarks = naruto_arv2.process_face_mesh(frame)
        if face_landmarks:
           for face_landmark in face_landmarks.multi_face_landmarks:
              left_eye, right_eye = naruto_arv2.get_eye_landmarks(face_landmark, frame)
              if left_eye:
                 copied_frame = naruto_arv2.overlay_sharingan(copied_frame, left_eye[0])
              if right_eye:
                  copied_frame = naruto_arv2.overlay_sharingan(copied_frame, right_eye[0])

      if '2finger' in jutsu:
        # Segmentação da pessoa para criar a máscara
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = naruto_arv2.segmenter.process(frame_rgb)
        if results.segmentation_mask is not None:
            mask = results.segmentation_mask > 0.5
            copied_frame = naruto_arv2.overlay_chakra_aura(copied_frame, mask, naruto_arv2.frame_idx)




    copied_frame = naruto_arv2.replace_background(copied_frame)
    cv2.imshow("Naruto AR", frame)
    cv2.imshow("Naruto ARv2", copied_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  naruto_arv2.cap.release()
  cv2.destroyAllWindows()
      
