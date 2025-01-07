import cv2
import mediapipe as mp
import numpy as np
import json

class NarutoAR:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Variável para armazenar snapshots
        self.snapshots = []

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

    def capture_snapshot(self, hand_landmarks, name):
        """Captura as posições de landmarks normalizados e armazena no snapshot com um nome."""
        snapshot = {
            "name": name,
            "landmarks": self.normalize_landmarks(hand_landmarks)
        }
        self.snapshots.append(snapshot)  # Adiciona o snapshot à lista

    def save_snapshots_to_file(self, filename="snapshots.json"):
        """Salva os snapshots no ficheiro JSON."""
        try:
            with open(filename, "w") as file:
                json.dump(self.snapshots, file, indent=4)
            print(f"Snapshots salvos em '{filename}'.")
        except Exception as e:
            print(f"Erro ao salvar os snapshots: {e}")

    def load_snapshots_from_file(self, file_path="snapshots.json"):
        """Carrega os snapshots do ficheiro JSON."""
        try:
            with open(file_path, "r") as file:
                self.snapshots = json.load(file)
            print(f"Snapshots carregados de '{file_path}'.")
        except FileNotFoundError:
            print(f"Ficheiro '{file_path}' não encontrado. Nenhum snapshot carregado.")
            self.snapshots = []  # Inicializa como vazio

    def compare_with_snapshot(self, hand_landmarks, threshold=0.5):
        """Compara a posição atual dos landmarks normalizados com os snapshots armazenados."""
        if not self.snapshots:
            return None  # Não há snapshots armazenados para comparação

        # Normaliza os landmarks atuais
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
                print(f"Gesto correspondente: {snapshot['name']}")
                return snapshot["name"]  # Retorna o nome do gesto correspondente
        return None  # Nenhuma correspondência encontrada

    def detect_hand_landmarks(self, frame):
        """Detecta landmarks e desenha sobre a mão."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame, results


if __name__ == '__main__':
    naruto_ar = NarutoAR()

    # Carregar snapshots salvos
    naruto_ar.load_snapshots_from_file()

    while True:
        ret, frame = naruto_ar.cap.read()
        if not ret:
            break

        frame, results = naruto_ar.detect_hand_landmarks(frame)
        if results and results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Capturar snapshot ao pressionar a tecla 'c'
            if cv2.waitKey(1) & 0xFF == ord('c'):
                name = input("Digite o nome do gesto capturado: ")  # Solicita o nome do gesto
                naruto_ar.capture_snapshot(hand_landmarks, name)  # Salva a posição dos landmarks
                cv2.putText(frame, f"Gesto '{name}' capturado", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Comparar os movimentos com os snapshots ao pressionar 'm'
            if cv2.waitKey(1) & 0xFF == ord('m'):
                gesture_name = naruto_ar.compare_with_snapshot(hand_landmarks)
                if gesture_name:
                    cv2.putText(frame, f"Gesto Reconhecido: {gesture_name}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Nenhum gesto correspondente", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Salvar snapshots no ficheiro ao pressionar 's'
            if cv2.waitKey(1) & 0xFF == ord('s'):
                naruto_ar.save_snapshots_to_file()

        # Exibição
        cv2.imshow('Naruto AR', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Salvar snapshots no final do programa (opcional)
    naruto_ar.save_snapshots_to_file()
    naruto_ar.cap.release()
    cv2.destroyAllWindows()
