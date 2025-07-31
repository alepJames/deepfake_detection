import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image

def fft_spectrum(image_path):
    print(f"\n[INFO] Running FFT spectrum analysis: {image_path}")
    img = Image.open(image_path).convert("L")  # grayscale
    img_np = np.array(img.resize((224, 224)))  # shape: (H, W)

    # Apply 2D FFT
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)  # log transform for visibility

    # Visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude, cmap='gray')
    plt.title("2D FFT Frequency Spectrum")
    plt.axis('off')

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", f"fft_spectrum_{os.path.basename(image_path)}")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print(f"[✔] Saved FFT spectrum: {out_path}")

# CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fft_analyze.py <image_path>")
    else:
        fft_spectrum(sys.argv[1])
