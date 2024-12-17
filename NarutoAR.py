import cv2
import mediapipe 
import numpy as np

class NarutoAR:
  def __init__(self):
    self.cap = cv2.VideoCapture(0)
    #MediaPipe
    self.mp_hands = mediapipe.solutions.hands
    self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    self.mp_drawing = mediapipe.solutions.drawing_utils
    self.mp_selfie_segmentation = mediapipe.solutions.selfie_segmentation
    self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    #Images
    self.background_images = ['assets/images/background.png']
    self.background_image = None

   
  

  def replace_background(self, frame):
    if self.background_image is None:
      self.background_image = cv2.imread(self.background_images[0])
    
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.segmenter.process(frame_rgb)

    # Cria a máscara binária (pessoa vs. fundo)
    mask = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
    mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
    
    background_resized = cv2.resize(self.background_image, (frame.shape[1], frame.shape[0]))

    # Aplica a máscara no frame
    output_image = np.uint8(mask * frame + (1 - mask) * background_resized)

    return output_image

    
if __name__ == '__main__':

  naruto_ar = NarutoAR()

  while True:
    ret, frame = naruto_ar.cap.read()
    if not ret:
      break

    frame = naruto_ar.replace_background(frame)

    cv2.imshow('Naruto AR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
      
  naruto_ar.cap.release()
  cv2.destroyAllWindows()
