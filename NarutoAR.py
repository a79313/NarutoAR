import cv2
import mediapipe as mp
import numpy as np
import json
from PIL import Image, ImageSequence

class NarutoAR:
  def __init__(self):
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
      

  def overlay_rasengan(self, coordinates, frame):
    rasengan_frame = self.frames[0][self.frame_idx]
    self.frame_idx = (self.frame_idx + 1) % len(self.frames[0])

    # Redimensionar o Rasengan
    rasengan_resized = rasengan_frame.resize((100, 100))  # Ajuste o tamanho conforme necessário

    # Converte o frame para PIL e adiciona o Rasengan
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    position = (coordinates[0] - rasengan_resized.width // 2, coordinates[1] - rasengan_resized.height // 2)
    frame_pil.paste(rasengan_resized, position, rasengan_resized)

    # Converter de volta para o formato OpenCV
    frame_with_rasengan = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)
    return frame_with_rasengan     
  
  def overlay_fireball(self, coordinates, frame):
    fireball_frame = self.frames[1][self.frame_idx]
    self.frame_idx = (self.frame_idx + 1) % len(self.frames[1])

    # Redimensionar o Rasengan
    fireball_resized = fireball_frame.resize((100, 100))  # Ajuste o tamanho conforme necessário

    # Converte o frame para PIL e adiciona o Rasengan
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    position = (coordinates[0] - fireball_resized.width // 2, coordinates[1] - fireball_resized.height // 2)
    frame_pil.paste(fireball_resized, position, fireball_resized)

    # Converter de volta para o formato OpenCV
    frame_with_fireball = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)
    return frame_with_fireball
      

