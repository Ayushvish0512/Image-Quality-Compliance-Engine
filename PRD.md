# Product Requirements Document (PRD)

# Project Name

**BBIPL Vision Compliance Engine (VCE)**

Version: 1.0

Status: Draft

Owner: Ayush Vishwakarma

---

# 1. Overview

The BBIPL Vision Compliance Engine (VCE) is a lightweight Computer Vision API that validates employee attendance selfies before they are accepted by the ERP system.

Unlike traditional attendance systems that only record an image, VCE evaluates the quality and compliance of the submitted selfie against configurable business rules and returns a detailed compliance report.

The system is intended to replace the first level of manual image verification performed by supervisors.

The API is designed to run entirely on **Render Free Tier** without requiring GPU acceleration.

---

# 2. Problem Statement

Supervisors currently verify attendance images manually because employees may submit photos that are:

- Blurry
- Too dark
- Multiple people in frame
- Wrong uniform
- Face hidden
- Looking away
- Standing too far
- Standing too close
- Screenshot instead of live photo
- Previously taken image
- Cropped image
- Missing mandatory accessories (Cap, Apron, etc.)

Manual verification is inconsistent, slow, and difficult to scale across multiple locations.

---

# 3. Goal

The system should automatically answer the following questions:

- Is there exactly one person?
- Is a face clearly visible?
- Is the image blurry?
- Is the employee wearing the required T-shirt?
- Is the employee wearing the required cap?
- Are other required accessories present?
- Is the face at the correct distance?
- Is lighting acceptable?
- Is the person facing the camera?
- Is the image likely genuine?
- Generate an overall compliance score.

---

# 4. Objectives

## Functional Objectives

- Validate attendance selfie automatically.
- Return detailed scores for every validation.
- Return reasons for failure.
- Support configurable dress code.
- Support configurable accessories.
- Support configurable scoring weights.
- Work completely on CPU.

---

## Business Objectives

- Reduce manual attendance verification.
- Increase attendance quality.
- Standardize image validation across all locations.
- Prevent fake attendance submissions.
- Provide consistent compliance scoring.

---

# 5. Scope

## Included

- Face Detection
- Face Visibility
- Blur Detection
- Brightness Detection
- Face Orientation
- Face Distance
- Multiple Person Detection
- Dress Detection
- Cap Detection
- Accessory Detection
- Screenshot Detection
- Compliance Score Generation

---

## Not Included

- Face Recognition
- Employee Identity Verification
- Liveness Detection
- Counter Validation
- Stall Cleanliness Detection
- Attendance Database
- Mobile Application

---

# 6. User Flow

Employee

↓

Open Attendance

↓

Upload Selfie

↓

Image Validation

↓

Generate Compliance Report

↓

Return Result

↓

ERP decides

PASS / REVIEW / REJECT

---

# 7. Functional Requirements

## FR-1 Image Upload

Input

Supported Formats

- JPG
- JPEG
- PNG
- WEBP

Maximum Size

10 MB

---

## FR-2 Person Detection

Requirement

Determine whether exactly one person exists.

Output

- No Person
- One Person
- Multiple Persons

Fail Conditions

- Zero people
- More than one person

---

## FR-3 Face Detection

Requirement

Detect human face.

Output

```json
{
    "detected": true,
    "confidence": 98
}
```

---

## FR-4 Face Visibility

Check visibility of

- Eyes
- Nose
- Mouth
- Chin
- Forehead

Return

0–100 score

---

## FR-5 Blur Detection

Measure image sharpness.

Possible Methods

- Variance of Laplacian
- Tenengrad
- Brenner Gradient

Output

```json
{
    "blur_score":91
}
```

---

## FR-6 Lighting Detection

Evaluate

- Underexposed
- Overexposed
- Balanced

Return

Brightness Score

Contrast Score

---

## FR-7 Face Distance

Determine if employee is

- Too close
- Good
- Too far

Based on

Face occupies configurable percentage of frame.

---

## FR-8 Face Orientation

Determine

- Front
- Left
- Right
- Up
- Down

Also calculate

Yaw

Pitch

Roll

---

## FR-9 Dress Detection

Detect employee uniform.

Configuration must be editable.

Example

```yaml
uniform:

  required: true

  tshirt:

      enabled: true

      allowed_colors:

      - Red

      confidence: 70
```

The application must allow changing:

- Colors
- Confidence threshold
- Required / Optional

without changing code.

---

## FR-10 Cap Detection

Configuration

```yaml
cap:

    required: true

    color:

      - Red
```

Return

Present

Not Present

Unknown

---

## FR-11 Accessory Detection

Should support configurable accessories.

Examples

- Apron
- Jacket
- ID Card
- Gloves
- Mask
- Hair Net

Each accessory must support

```yaml
required: true

enabled: true

color:
```

No code modification should be required.

---

## FR-12 Screenshot Detection

System should attempt to identify

- Screenshot
- Edited image
- Cropped screenshot

Possible Indicators

- Missing camera metadata
- Screen aspect ratios
- Compression artifacts
- UI borders
- Pixel duplication
- EXIF mismatch

Output

Low

Medium

High Risk

---

## FR-13 Image Authenticity

Attempt to detect

- Reused image
- Forwarded image
- Screenshot
- Edited image

This is heuristic-based and should return a confidence level rather than a definitive decision.

---

## FR-14 Compliance Score

Return

```json
{
  "overall_score":88,
  "status":"PASS"
}
```

---

# 8. Configurable Rules

The application shall expose a configuration file.

Example

```yaml
uniform:

  tshirt_required: true

  tshirt_colors:

    - Red

cap:

  required: true

  colors:

    - Red

gloves:

  required: false

hairnet:

  required: false

minimum_face_size: 30

maximum_face_size: 70

minimum_blur_score: 60

minimum_visibility: 75

minimum_brightness: 50

minimum_overall_score: 80
```

Business users should be able to modify these values without changing source code.

---

# 9. Compliance Score

| Validation | Weight |
|------------|--------|
| One Person | 15 |
| Face Detected | 15 |
| Face Visibility | 10 |
| Blur | 10 |
| Lighting | 10 |
| Face Distance | 10 |
| Face Orientation | 10 |
| Uniform | 15 |
| Cap | 5 |
| Accessories | 5 |
| Screenshot Risk | 5 |

Total = 100

---

# 10. Result Levels

| Score | Status |
|---------|---------|
| 95–100 | Excellent |
| 85–94 | Pass |
| 70–84 | Review |
| Below 70 | Reject |

---

# 11. API Response

```json
{
  "status":"PASS",

  "overall_score":91,

  "checks":{

      "person_count":1,

      "face_detected":true,

      "face_visibility":94,

      "blur_score":89,

      "lighting":82,

      "distance":"Good",

      "orientation":"Front",

      "uniform":{

          "detected":true,

          "color":"Red",

          "confidence":92

      },

      "cap":{

          "detected":true,

          "color":"Red"

      },

      "accessories":{

          "gloves":false,

          "apron":true

      },

      "screenshot_risk":"Low"

  },

  "recommendations":[

      "Increase lighting",

      "Move slightly closer"

  ]
}
```

---

# 12. Non-Functional Requirements

## Deployment

- Render Free Tier
- CPU only
- No GPU

---

## Memory

Target Maximum RAM

**≤ 250 MB**

Absolute Maximum

**≤ 350 MB**

---

## Accuracy

**Accuracy is prioritized over latency.**

Expected targets:

| Module | Target Accuracy |
|---------|----------------:|
| Person Detection | >99% |
| Face Detection | >99% |
| Blur Detection | >98% |
| Face Orientation | >95% |
| Face Visibility | >95% |
| Lighting Detection | >95% |
| Distance Detection | >95% |
| Uniform Detection | >90% (configurable colors) |
| Cap Detection | >90% |
| Screenshot Risk Detection | Heuristic only (advisory, not guaranteed) |

---

# 13. Suggested Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI |
| Image Processing | OpenCV |
| Face Detection & Landmarks | MediaPipe |
| Numerical Processing | NumPy |
| Configuration | YAML |
| Image Validation | Pillow |
| Optional ML Runtime | ONNX Runtime (only if lightweight models are introduced) |
| Deployment | Render Free Tier |

---

# 14. Future Enhancements

- Face identity verification
- Liveness detection
- QR/GPS correlation
- Counter cleanliness scoring
- Uniform wrinkle detection
- Dirt/stain detection on clothing
- Android offline inference
- Continuous model improvement from supervisor feedback
- Multi-image validation (selfie + stall photo)

---

# 15. Success Criteria

The system will be considered successful if it:

- Automatically validates attendance selfies without human intervention for the majority of cases.
- Produces consistent compliance scores across locations.
- Runs reliably within the memory limits of the Render Free Tier.
- Allows business teams to update uniform colors, accessories, thresholds, and scoring rules through configuration rather than code changes.
- Returns clear, actionable feedback so employees can immediately retake non-compliant photos.