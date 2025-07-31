import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import sys
import os

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Load and convert image
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    return img_np

# Extract 468 3D landmarks
def get_landmarks_np(image_np):
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(img_bgr)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        points = np.array([[lm.x, lm.y] for lm in landmarks])
        return points
    return None

# Symmetry score: lower distance between left-right = better symmetry
def compute_symmetry_score(landmarks):
    mid = 0.5  # image normalized width midpoint
    left = landmarks[landmarks[:, 0] < mid]
    right = landmarks[landmarks[:, 0] > mid]

    # Flip right x-axis to compare with left
    right_flipped = right.copy()
    right_flipped[:, 0] = 1.0 - right_flipped[:, 0]

    # Match points (smallest set)
    count = min(len(left), len(right_flipped))
    diffs = np.abs(left[:count] - right_flipped[:count])
    symmetry_score = 1.0 - np.mean(diffs)  # higher = more symmetrical
    return round(float(np.clip(symmetry_score, 0, 1)), 4)

# Visualization
def draw_landmarks(image_np, landmarks, save_path):
    h, w = image_np.shape[:2]
    for lm in landmarks:
        x, y = int(lm[0] * w), int(lm[1] * h)
        cv2.circle(image_np, (x, y), 1, (255, 255, 0), -1)
    cv2.imwrite(save_path, image_np)
    print(f"[✔] Saved landmark overlay: {save_path}")

# Main runner
def analyze(image_path):
    print(f"\n[INFO] Analyzing landmark symmetry: {image_path}")
    image_np = load_image(image_path)
    landmarks = get_landmarks_np(image_np)
    if landmarks is None:
        print(" No face or landmarks detected.")
        return

    score = compute_symmetry_score(landmarks)
    print(f" Landmark Symmetry Score: {score:.4f} (1.0 = perfect symmetry)")

    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", f"landmarks_{os.path.basename(image_path)}")
    draw_landmarks(image_np.copy(), landmarks, output_path)

# CLI entry
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python landmark_symmetry.py <image_path>")
    else:
        analyze(sys.argv[1])
