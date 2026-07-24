"""
Test the best.pt cap detection model on all test images
Model classes: {0: 'cap', 1: 'no_cap'}
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"d:/webbased projects/Image Quality & Compliance Engine/models/best.pt"
TEST_DIR = r"d:/webbased projects/Image Quality & Compliance Engine/test images"


def get_all_images(base_dir):
    """Get all image files recursively"""
    images = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                rel_dir = os.path.relpath(root, base_dir)
                images.append({
                    'path': os.path.join(root, f),
                    'name': f,
                    'folder': rel_dir if rel_dir != '.' else 'root'
                })
    return images


def test_yolo_cap_model(model, img_bgr):
    """Test best.pt model on image"""
    results = model(img_bgr, conf=0.01, verbose=False)
    r = results[0]
    boxes = r.boxes

    if boxes is not None and len(boxes) > 0:
        all_dets = []
        cap_confs = []
        no_cap_confs = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = r.names.get(cls_id, "?")
            xyxy = box.xyxy[0].tolist()
            det = f"{cls_name}({conf:.3f})[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}]"
            all_dets.append(det)
            if cls_id == 0:
                cap_confs.append(conf)
            else:
                no_cap_confs.append(conf)

        best = boxes[0]
        best_cls = int(best.cls[0])
        best_conf = float(best.conf[0])
        best_name = r.names.get(best_cls, "?")

        cap_conf = max(cap_confs) if cap_confs else 0
        no_cap_conf = max(no_cap_confs) if no_cap_confs else 0

        return {
            'has_detection': True,
            'best_class': best_name,
            'best_conf': best_conf,
            'num_detections': len(boxes),
            'cap_detected': cap_conf > no_cap_conf,
            'cap_confidence': cap_conf,
            'no_cap_confidence': no_cap_conf,
            'all_detections': ', '.join(all_dets),
        }
    else:
        return {
            'has_detection': False,
            'best_class': 'NONE',
            'best_conf': 0,
            'num_detections': 0,
            'cap_detected': False,
            'cap_confidence': 0,
            'no_cap_confidence': 0,
            'all_detections': '',
        }


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    all_images = get_all_images(TEST_DIR)
    print(f"Found {len(all_images)} images\n")

    # Load the best.pt model
    print("Loading best.pt cap detection model...")
    model = YOLO(MODEL_PATH)
    print(f"Model task: {model.task}")
    print(f"Model classes: {model.names}\n")

    # Print header
    print("=" * 200)
    header = f"{'FOLDER':<10} | {'FILENAME':<70} | {'Best Pred':<12} | {'Conf':<8} | {'Cap?':<8} | {'Cap Conf':<10} | {'NoCap Conf':<12} | {'All Detections'}"
    print(header)
    print("=" * 200)

    for img_info in all_images:
        img = cv2.imread(img_info['path'])
        if img is None:
            print(f"{img_info['folder']:<10} | {img_info['name']:<70} | {'ERROR':<12} | {'':<8} | {'':<8} | {'':<10} | {'':<12} | Could not load")
            continue

        # Run YOLO best.pt model
        result = test_yolo_cap_model(model, img)

        cap_str = "YES" if result['cap_detected'] else "NO"
        all_dets_str = result['all_detections']
        print(f"{img_info['folder']:<10} | {img_info['name']:<70} | {result['best_class']:<12} | {result['best_conf']:.3f}  | {cap_str:<8} | {result['cap_confidence']:.3f}    | {result['no_cap_confidence']:.3f}      | {all_dets_str}")

    print("=" * 200)

    # Detailed analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS PER IMAGE")
    print("=" * 80)

    for img_info in all_images:
        img = cv2.imread(img_info['path'])
        if img is None:
            continue
        result = test_yolo_cap_model(model, img)
        print(f"\n[{img_info['folder']}] {img_info['name']}")
        print(f"  best.pt best prediction: {result['best_class']} (conf: {result['best_conf']:.3f})")
        print(f"  Cap detected: {result['cap_detected']} (cap_conf: {result['cap_confidence']:.3f}, no_cap_conf: {result['no_cap_confidence']:.3f})")
        print(f"  All detections ({result['num_detections']}): {result['all_detections']}")

    print("\n" + "=" * 80)
    print("VERDICT ON best.pt MODEL")
    print("=" * 80)
    print(f"Classes: {model.names}")
    print(f"  class 0 = 'cap'    - person is wearing a cap")
    print(f"  class 1 = 'no_cap' - person is NOT wearing a cap")
    print("")
    print("Capabilities:")
    print("  ✅ Can detect cap presence/absence (binary classification)")
    print("  ❌ Does NOT detect cap color (no color classes in model)")
    print("")
    print("For color detection, the existing cap_detector.py uses:")
    print("  - MediaPipe face detection to find cap region")
    print("  - Edge analysis + color heuristics to determine cap/color")
    print("  - Config rules.yaml defines allowed cap colors")
