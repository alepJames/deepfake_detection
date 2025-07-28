import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import sys

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def draw_landmarks(image_path, output_path='output_landmarks.jpg'):
    image = Image.open(image_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = face_mesh.process(img_bgr)
    if results.multi_face_landmarks:
        annotated_image = img_bgr.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        cv2.imwrite(output_path, annotated_image)
        print(f"✅ Saved annotated image with landmarks to: {output_path}")
    else:
        print("❌ No face landmarks detected")


# Run from terminal
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize.py path_to_image.jpg")
    else:
        draw_landmarks(sys.argv[1])
