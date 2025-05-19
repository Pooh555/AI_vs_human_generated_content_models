# inference.py

import argparse
import os
import sys
import torch
import cv2
import numpy as np
from torchvision import transforms
from model import create_model

def extract_frames(video_path, num_frames=7):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise IOError(f"Video {video_path} contains no frames")
    indices = [int(i * (total / num_frames)) for i in range(num_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames extracted from video")
    return frames

def preprocess_frame(frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std =[0.229,0.224,0.225]
        )
    ])
    return preprocess(img)

def predict_video(model, video_path, num_frames, device):
    model.eval()
    frames = extract_frames(video_path, num_frames)
    batch = torch.stack([preprocess_frame(f) for f in frames], dim=0).to(device)
    with torch.no_grad():
        logits = model(batch).squeeze(1)               # shape [N]
        probs  = torch.sigmoid(logits).cpu().numpy()   # shape [N]
    avg_fake_conf = float(np.mean(probs))
    if avg_fake_conf > 0.5:
        label = "FAKE"
        confidence = avg_fake_conf
    else:
        label = "REAL"
        confidence = 1.0 - avg_fake_conf
    return label, confidence

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Detect whether a video is AI-generated or real."
    )
    p.add_argument("video", help="Path to the input video file")
    p.add_argument(
        "--frames", "-f", type=int, default=7,
        help="Number of frames to sample (default: 7)"
    )
    p.add_argument(
        "--model", "-m", default="best_model.pth",
        help="Path to the trained model weights"
    )
    args = p.parse_args()

    # sanity checks
    if not os.path.isfile(args.video):
        print(f"ERROR: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load model
    print("Loading model...", end="", flush=True)
    model = create_model().to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    print(" done.")

    # run inference
    print(f"Analyzing {args.video} ({args.frames} frames)...")
    try:
        label, confidence = predict_video(
            model, args.video, args.frames, device
        )
    except Exception as e:
        print(f"ERROR during inference: {e}", file=sys.stderr)
        sys.exit(1)

    # output
    print(f"\n🔍 Prediction: {label}")
    print(f"🔢 Confidence ({label}): {confidence * 100:.2f}%")
