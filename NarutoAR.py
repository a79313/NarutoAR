import cv2
import mediapipe 

class NarutoAR:
  def __init__(self):
    self.mp_hands = mediapipe.solutions.hands
    self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    self.mp_drawing = mediapipe.solutions.drawing_utils
    self.cap = cv2.VideoCapture(0)
    
  



if __name__ == '__main__':

  naruto_ar = NarutoAR()

  while True:
    ret, frame = naruto_ar.cap.read()
    if not ret:
      break
        
    cv2.imshow('Naruto AR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
      
  naruto_ar.cap.release()
  cv2.destroyAllWindows()
