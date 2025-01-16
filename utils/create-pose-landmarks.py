import mediapipe as mp
import cv2
import json
import numpy as np

cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
landmarks_list = []
def extract_landmarks(results):
  if not results.pose_landmarks:
    return None
  return[(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

def normalize_landmarks(landmarks):
    """Normalize landmarks by subtracting the root landmark and scaling by the size of the pose."""
    root = np.array([landmarks[0][0], landmarks[0][1], landmarks[0][2]])
    normalized_landmarks = [np.array([lm[0], lm[1], lm[2]]) - root for lm in landmarks]

    # Calculate the size of the pose based on landmark distances
    max_distance = max(np.linalg.norm(lm) for lm in normalized_landmarks if np.linalg.norm(lm) > 0)
    max_distance = max_distance if max_distance > 0 else 1e-6
    normalized_landmarks = [lm / max_distance for lm in normalized_landmarks]

    return normalized_landmarks

def save_landmarks(landmarks, filename):
  normalized_landmarks = normalize_landmarks(landmarks)
  pose_name = input('Enter the pose name: ')
  normalized_landmarks_list = [lm.tolist() for lm in normalized_landmarks]
  data = {
    'pose': pose_name,
    'landmarks': normalized_landmarks_list
  }
  with open(filename, 'w') as f:
    json.dump(data, f)


while True:
  ret, frame = cap.read()
  if not ret:
    break

  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = pose.process(rgb_frame)

  landmarks = extract_landmarks(results)

  if results.pose_landmarks:
    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
  cv2.imshow('Frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('c'):
    landmarks_list.append(save_landmarks(landmarks, 'landmarks.json'))
    continue
  if cv2.waitKey(1) & 0xFF == ord('q'):
    save_landmarks(landmarks_list, 'landmarks.json')
    break