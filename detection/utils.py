import cv2
import torch
import numpy as np
from torchvision import transforms
from torch import nn
from .model_def import Model
import face_recognition  # always use face_recognition

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "detection/models/deepfake_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

model = Model(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("[INFO] Model loaded and set to eval mode.")

# -----------------------------
# Preprocessing
# -----------------------------
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

sm = nn.Softmax(dim=1)

# -----------------------------
# Frame Extraction (face_recognition only)
# -----------------------------
def extract_frames(video_path, sequence_length=40):
    print(f"[INFO] Extracting frames from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total frames in video: {total_frames}")
    step = max(total_frames // (sequence_length * 2), 1)  # sample more densely
    print(f"[INFO] Sampling every {step} frame(s), target: {sequence_length} frames")

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARN] Could not read frame {i}, skipping...")
            continue

        # -------- FIX: ensure valid format --------
        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)

        # If grayscale → RGB
        if len(frame.shape) == 2:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # If RGBA, drop alpha
        if rgb_frame.ndim == 3 and rgb_frame.shape[2] == 4:
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGBA2RGB)
        # ------------------------------------------

        try:
            faces = face_recognition.face_locations(rgb_frame, model="cnn")  
        except Exception as e:
            print(f"{i}: {e}")
            continue

        if faces:
            top, right, bottom, left = faces[0]
            face_img = rgb_frame[top:bottom, left:right, :]
            if face_img.size > 0:
                try:
                    face_img = cv2.resize(face_img, (im_size, im_size))
                    frames.append(transform(face_img))
                    print(f"[INFO] Collected {len(frames)}/{sequence_length} faces (frame {i})")
                except Exception as e:
                    print(f"[WARN] Failed to resize/crop face at frame {i}: {e}")

        if len(frames) >= sequence_length:
            print("[INFO] Reached required number of frames.")
            break

    cap.release()

    # -------- Fallback if no faces detected --------
    if len(frames) == 0:
        print("[WARN] No faces detected, using fallback center crop.")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise ValueError("Could not read video for fallback.")

        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)

        if len(frame.shape) == 2:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if rgb_frame.ndim == 3 and rgb_frame.shape[2] == 4:
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGBA2RGB)

        h, w, _ = rgb_frame.shape
        center_crop = rgb_frame[h//4:3*h//4, w//4:3*w//4]
        face_img = cv2.resize(center_crop, (im_size, im_size))
        frames = [transform(face_img)] * sequence_length
    # ----------------------------------------------

    print(f"[INFO] Total frames prepared for model: {len(frames)}")
    return torch.stack(frames).unsqueeze(0)

# -----------------------------
# Prediction
# -----------------------------
def predict_video(video_path):
    print("[INFO] Starting prediction...")
    frames = extract_frames(video_path).to(device)
    with torch.no_grad():
        fmap, logits = model(frames)
        probs = sm(logits)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item() * 100

    label = "REAL" if pred_class == 1 else "FAKE"
    print(f"[RESULT] Prediction: {label} ({confidence:.2f}% confidence)")
    return label, confidence
