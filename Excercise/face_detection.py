"""
Face Detection with Deep Neural Network (SSD + ResNet10)
=========================================================
This script uses OpenCV's DNN module to run a Single Shot Detector
(SSD) with a ResNet-10 backbone — the same architecture family used
in lightweight YOLO-style detectors.  The model is pre-trained on
faces and produces bounding boxes + confidence scores just like YOLO.

For full YOLOv8 usage (requires internet to download weights once):
  from ultralytics import YOLO
  model = YOLO("yolov8n.pt")
  results = model("image.jpg")

File layout
-----------
  face_detection.py               ← this file
  deploy.prototxt                  ← SSD network architecture
  res10_300x300_ssd_iter_140000.caffemodel  ← pre-trained weights

Usage
-----
  python face_detection.py --image  path/to/photo.jpg
  python face_detection.py --webcam          # live webcam
  python face_detection.py --demo            # built-in test
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ─── paths (relative to this script) ────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
PROTOTXT     = BASE_DIR / "deploy.prototxt"
CAFFEMODEL   = BASE_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# ─── helpers ─────────────────────────────────────────────────────────────────

def load_model() -> cv2.dnn.Net:
    if not PROTOTXT.exists() or not CAFFEMODEL.exists():
        raise FileNotFoundError(
            "Model files missing.  Run:\n"
            "  python face_detection.py --download"
        )
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))
    print(f"[FaceDetector] Loaded SSD-ResNet10 face detector from {BASE_DIR}")
    return net


def detect_faces(
    net: cv2.dnn.Net,
    frame: np.ndarray,
    conf_threshold: float = 0.5,
) -> list[dict]:
    """
    Run the SSD face detector on *frame* (BGR numpy array).

    Returns
    -------
    list of {"bbox": (x1,y1,x2,y2), "confidence": float}
    """
    h, w = frame.shape[:2]

    # Pre-process: resize to 300×300, subtract mean pixel values
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()   # shape: (1, 1, N, 7)

    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue
        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w, int(detections[0, 0, i, 5] * w))
        y2 = min(h, int(detections[0, 0, i, 6] * h))
        faces.append({"bbox": (x1, y1, x2, y2), "confidence": confidence})

    return faces


def draw_detections(frame: np.ndarray, faces: list[dict]) -> np.ndarray:
    out = frame.copy()
    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        conf = f["confidence"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 230, 0), 2)
        label = f"Face  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 230, 0), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"Detected: {len(faces)} face(s)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2, cv2.LINE_AA)
    return out


def download_model_files():
    """Download the two model files from the OpenCV repository."""
    import urllib.request

    files = {
        "deploy.prototxt": (
            "https://raw.githubusercontent.com/opencv/opencv/master/"
            "samples/dnn/face_detector/deploy.prototxt"
        ),
        "res10_300x300_ssd_iter_140000.caffemodel": (
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
            "dnn_samples_face_detector_20170830/"
            "res10_300x300_ssd_iter_140000.caffemodel"
        ),
    }
    for fname, url in files.items():
        dest = BASE_DIR / fname
        if dest.exists():
            print(f"  {fname} already present, skipping.")
            continue
        print(f"  Downloading {fname} …")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved → {dest}")
    print("Download complete.")


# ─── demo image generator ────────────────────────────────────────────────────

def make_demo_image() -> np.ndarray:
    """Create a realistic-ish synthetic portrait for the demo."""
    img = np.full((480, 640, 3), (60, 60, 60), dtype=np.uint8)

    # Background gradient
    for y in range(480):
        v = int(30 + y * 80 / 480)
        img[y] = (v, v + 10, v + 20)

    # Three face-like circles
    for cx, cy, r, skin in [(160,220,70,(180,130,100)),
                             (320,180,80,(200,150,120)),
                             (500,240,65,(160,110,90))]:
        cv2.ellipse(img, (cx, cy), (r, int(r*1.2)), 0, 0, 360, skin, -1)
        # eyes
        cv2.circle(img, (cx-r//3, cy-r//5), r//8, (40,40,40), -1)
        cv2.circle(img, (cx+r//3, cy-r//5), r//8, (40,40,40), -1)
        # mouth
        cv2.ellipse(img, (cx, cy+r//3), (r//3, r//6), 0, 0, 180, (60,30,30), 2)
        # hair
        cv2.ellipse(img, (cx, cy-r//2), (r, r//2), 0, 180, 360, (40,20,10), -1)

    cv2.putText(img, "Demo — synthetic faces", (10, 462),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    return img


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SSD-ResNet10 Face Detector")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--image",    type=str, help="Detect faces in an image file")
    grp.add_argument("--webcam",   action="store_true", help="Live webcam detection")
    grp.add_argument("--demo",     action="store_true", help="Built-in synthetic demo")
    grp.add_argument("--download", action="store_true", help="Download model weights")
    ap.add_argument("--conf",   type=float, default=0.50, help="Confidence threshold")
    ap.add_argument("--output", type=str,   default=None, help="Save result image here")
    args = ap.parse_args()

    if args.download:
        download_model_files()
        return

    net = load_model()

    # ── demo ──────────────────────────────────────────────────────────────
    if args.demo or (not args.image and not args.webcam):
        print("\n[Demo] Running on synthetic image …")
        frame = make_demo_image()
        faces = detect_faces(net, frame, args.conf)
        print(f"[Demo] Detected {len(faces)} face(s):")
        for i, f in enumerate(faces, 1):
            print(f"  [{i}] bbox={f['bbox']}  confidence={f['confidence']:.3f}")
        annotated = draw_detections(frame, faces)
        out_path = args.output or "/mnt/user-data/outputs/demo_result.jpg"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, annotated)
        print(f"[Demo] Annotated image saved → {out_path}")
        return

    # ── image ─────────────────────────────────────────────────────────────
    if args.image:
        p = Path(args.image)
        if not p.exists():
            sys.exit(f"File not found: {p}")
        frame = cv2.imread(str(p))
        if frame is None:
            sys.exit(f"Cannot read image: {p}")
        faces = detect_faces(net, frame, args.conf)
        print(f"Detected {len(faces)} face(s):")
        for i, f in enumerate(faces, 1):
            print(f"  [{i}] bbox={f['bbox']}  confidence={f['confidence']:.3f}")
        annotated = draw_detections(frame, faces)
        out = args.output or str(p.with_stem(p.stem + "_detected"))
        cv2.imwrite(out, annotated)
        print(f"Saved → {out}")
        return

    # ── webcam ────────────────────────────────────────────────────────────
    if args.webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("Cannot open webcam.")
        print("Webcam running — press 'q' to quit, 's' to save frame.")
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = detect_faces(net, frame, args.conf)
            cv2.imshow("Face Detection", draw_detections(frame, faces))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                fn = f"capture_{idx:04d}.jpg"
                cv2.imwrite(fn, draw_detections(frame, faces))
                print(f"Saved {fn}")
                idx += 1
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
