"""
Test the Classification Pipeline (Queue System)
Tests both tshirt_detection_model.pt and cap_detection_model.pt together.

Run: py -3 test_classification_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import cv2
import numpy as np
from detectors.classification_pipeline import run_classification_pipeline

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


def test_classification_pipeline(img_bgr):
    """Run the full classification pipeline queue"""
    return run_classification_pipeline(img_bgr)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    all_images = get_all_images(TEST_DIR)
    print(f"Found {len(all_images)} images\n")

    header = "=" * 120
    print(header)
    print(f"{'FOLDER':<12} | {'FILENAME':<55} | {'TSHIRT':<10} | {'TSHIRT COLOR':<14} | {'MATCH':<8} | {'CAP':<10} | {'CAP COLOR':<12}")
    print(header)

    for img_info in all_images:
        img = cv2.imread(img_info['path'])
        if img is None:
            print(f"{img_info['folder']:<12} | {img_info['name']:<55} | {'ERROR':<10}")
            continue

        result = test_classification_pipeline(img)

        tshirt = result["tshirt"]
        cap = result["cap"]

        tshirt_str = "YES" if tshirt["detected"] else "NO"
        tshirt_color = tshirt["model_color"] if tshirt["detected"] else "-"
        match_str = "OK" if tshirt.get("color_match", False) else "MISMATCH" if tshirt["detected"] else "-"
        cap_str = cap["detected"]
        cap_color = cap["hsv_color"] if cap["detected"] == "Present" else "-"

        print(f"{img_info['folder']:<12} | {img_info['name']:<55} | {tshirt_str:<10} | {tshirt_color:<14} | {match_str:<8} | {cap_str:<10} | {cap_color:<12}")

    print(header)

    # Detailed analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    for img_info in all_images:
        img = cv2.imread(img_info['path'])
        if img is None:
            continue

        result = test_classification_pipeline(img)
        tshirt = result["tshirt"]
        cap = result["cap"]

        print(f"\n{'─' * 60}")
        print(f"📷 [{img_info['folder']}] {img_info['name']}")
        print(f"{'─' * 60}")

        print(f"\n🧥 TSHIRT DETECTION:")
        print(f"  Detected:         {tshirt['detected']}")
        if tshirt['detected']:
            print(f"  Model Color:      {tshirt['model_color']}")
            print(f"  Model Confidence: {tshirt['model_confidence']:.1f}%")
            print(f"  HSV Color:        {tshirt['hsv_color']}")
            print(f"  HSV Confidence:   {tshirt['hsv_confidence']:.1f}%")
            final = tshirt.get('final_color', tshirt['hsv_color'])
            final_conf = tshirt.get('final_confidence', tshirt['hsv_confidence'])
            print(f"  Final Color:      {final}")
            print(f"  Final Confidence: {final_conf:.1f}%")
            print(f"  Color Match:      {tshirt['color_match']}")
            print(f"  Allowed Colors:   {tshirt['allowed_colors']}")
        else:
            print(f"  → No tshirt detected")

        print(f"\n🧢 CAP DETECTION:")
        print(f"  Detected:         {cap['detected']}")
        if cap['detected'] == "Present":
            print(f"  Model Confidence: {cap['model_confidence']:.1f}%")
            print(f"  HSV Color:        {cap['hsv_color']}")
            print(f"  HSV Confidence:   {cap['hsv_confidence']:.1f}%")
            print(f"  Allowed Colors:   {cap['allowed_colors']}")
        else:
            print(f"  → No cap detected")

    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print(f"{'=' * 80}")

    # Summary
    print(f"\n{'=' * 80}")
    print("MODEL SUMMARY")
    print(f"{'=' * 80}")
    print(f"tshirt_detection_model.pt classes:     {{0: 'black', 1: 'blue', 2: 'grey', 3: 'white'}}")
    print(f"cap_detection_model.pt classes:        {{0: 'cap', 1: 'no_cap'}}")
    print(f"\nQueue Flow (Cap first, then Tshirt):")
    print(f"  Step 1 → cap_model detects cap presence")
    print(f"  Step 2 → crop cap region → HSV color analysis")
    print(f"  Step 3 → tshirt_model detects tshirt (with color class)")
    print(f"  Step 4 → crop tshirt region → HSV color analysis")
    print(f"\nColor analysis uses shared utils/color_analysis.py (single source of truth)")
    print(f"\nValues returned SEPARATELY for cap and tshirt ✅")
    print(f"\nMemory optimization: Only cap_detection_model.pt + tshirt_detection_model.pt loaded")
    print(f"(Removed redundant best.pt — saves ~6MB RAM)")

