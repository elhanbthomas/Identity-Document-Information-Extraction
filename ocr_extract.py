import cv2
import numpy as np
import json
import os
import re
import pytesseract
from pytesseract import Output

from id_preprocessing import preprocess_document


# ============================================================
# 1. LOAD TEMPLATE JSON
# ============================================================

def load_template(template_json_path):

    with open(template_json_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    return template


# ============================================================
# 2. LOAD AND PREPROCESS THE INCOMING DOCUMENT IMAGE
# ============================================================

def load_and_preprocess_incoming_image(image_path, target_width=1200):

    result = preprocess_document(image_path, target_width=target_width)

    # Use the color "final" image for alignment (not the binarized
    # ocr_ready one) since ORB keypoint matching needs texture/gradient
    # detail that binarization destroys.
    return result["final"]


# ============================================================
# 3. EXTRACT ORB KEYPOINTS FOR THE WHOLE INCOMING IMAGE (ONCE)
# ============================================================

def extract_orb_keypoints_full_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


# ============================================================
# 4. MATCH ONE ANCHOR'S STORED DESCRIPTORS AGAINST INCOMING IMAGE
# ============================================================

def match_anchor_to_incoming(anchor, template_w, template_h,
                              incoming_kp, incoming_des):

    descriptor_path = anchor["descriptor_ref"]

    if descriptor_path is None or not os.path.exists(descriptor_path):
        print(f"Skipping anchor '{anchor['id']}': no descriptor file found.")
        return [], []

    data = np.load(descriptor_path)
    anchor_kp_coords = data["keypoint_coords"]   # coords relative to the crop
    anchor_descriptors = data["descriptors"]

    if anchor_descriptors is None or len(anchor_descriptors) == 0:
        return [], []

    if incoming_des is None or len(incoming_des) == 0:
        return [], []

    # Convert anchor's normalized box back to pixel coords in the
    # full template image, so we can offset the crop-relative keypoints.
    box = anchor["box"]
    box_x_px = box["x"] * template_w
    box_y_px = box["y"] * template_h

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(anchor_descriptors, incoming_des, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        # Lowe's ratio test to keep only distinctive matches
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    template_pts = []
    incoming_pts = []

    for m in good_matches:
        ax, ay = anchor_kp_coords[m.queryIdx]
        # Shift back into full-template-image coordinate space
        full_template_x = ax + box_x_px
        full_template_y = ay + box_y_px

        template_pts.append([full_template_x, full_template_y])
        incoming_pts.append(incoming_kp[m.trainIdx].pt)

    return template_pts, incoming_pts


# ============================================================
# 5. COMBINE MATCHES FROM ALL ANCHORS AND COMPUTE HOMOGRAPHY
# ============================================================

def compute_alignment_homography(template, incoming_image):

    template_w = template["template_image"]["width"]
    template_h = template["template_image"]["height"]

    incoming_kp, incoming_des = extract_orb_keypoints_full_image(incoming_image)

    all_template_pts = []
    all_incoming_pts = []

    for anchor in template["anchors"]:

        t_pts, i_pts = match_anchor_to_incoming(
            anchor, template_w, template_h, incoming_kp, incoming_des
        )

        all_template_pts.extend(t_pts)
        all_incoming_pts.extend(i_pts)

    if len(all_template_pts) < 4:
        print("Not enough matched points across anchors to align document.")
        return None

    template_pts_np = np.array(all_template_pts, dtype=np.float32)
    incoming_pts_np = np.array(all_incoming_pts, dtype=np.float32)

    homography, mask = cv2.findHomography(
        template_pts_np,
        incoming_pts_np,
        cv2.RANSAC,
        5.0
    )

    if homography is None:
        print("Homography computation failed.")
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    print(f"Alignment homography computed using {inliers} inlier matches "
          f"out of {len(all_template_pts)} candidate matches.")

    return homography


# ============================================================
# 6. PROJECT A FIELD'S TEMPLATE BOX INTO THE INCOMING IMAGE
# ============================================================

def project_field_box(field, homography, template_w, template_h):

    box = field["box"]

    x_px = box["x"] * template_w
    y_px = box["y"] * template_h
    w_px = box["width"] * template_w
    h_px = box["height"] * template_h

    corners = np.array([
        [x_px, y_px],
        [x_px + w_px, y_px],
        [x_px + w_px, y_px + h_px],
        [x_px, y_px + h_px]
    ], dtype=np.float32).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(corners, homography)
    projected = projected.reshape(-1, 2)

    x_min = int(np.min(projected[:, 0]))
    y_min = int(np.min(projected[:, 1]))
    x_max = int(np.max(projected[:, 0]))
    y_max = int(np.max(projected[:, 1]))

    return x_min, y_min, x_max, y_max


# ============================================================
# 7. CROP A FIELD REGION SAFELY WITHIN IMAGE BOUNDS
# ============================================================

def safe_crop(image, x_min, y_min, x_max, y_max, padding=4):

    h, w = image.shape[:2]

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    if x_max <= x_min or y_max <= y_min:
        return None

    return image[y_min:y_max, x_min:x_max]


# ============================================================
# 8. RUN OCR ON A SINGLE FIELD CROP (ENGLISH ONLY)
# ============================================================

def ocr_field_crop(crop, psm=7):

    if crop is None or crop.size == 0:
        return "", 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale small crops slightly; helps Tesseract on small ID card fonts
    scale = 2
    resized = cv2.resize(
        gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

    config = f"--oem 3 --psm {psm}"

    # lang="eng" only -> any Hindi/regional script present is
    # effectively ignored/skipped rather than transcribed.
    data = pytesseract.image_to_data(
        resized,
        lang="eng",
        config=config,
        output_type=Output.DICT
    )

    words = []
    confidences = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = float(data["conf"][i])

        if word and conf > 0:
            words.append(word)
            confidences.append(conf)

    raw_text = " ".join(words).strip()
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0

    return raw_text, avg_confidence


# ============================================================
# 9. CLEAN / VALIDATE OCR TEXT BASED ON FIELD TYPE
# ============================================================

def clean_field_value(field_id, raw_text):

    text = raw_text.strip()

    if field_id in ("dob",):
        match = re.search(r"\d{2}[/\-.]\d{2}[/\-.]\d{4}", text)
        return match.group(0) if match else text

    if field_id in ("gender",):
        upper = text.upper()
        if "MALE" in upper and "FEMALE" not in upper:
            return "MALE"
        if "FEMALE" in upper:
            return "FEMALE"
        if re.search(r"\bM\b", upper):
            return "M"
        if re.search(r"\bF\b", upper):
            return "F"
        return text

    if field_id in ("aadhar_no", "aadhaar_no"):
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) == 12:
            return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"
        return text

    if field_id in ("passport_no",):
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        match = re.search(r"[A-Z0-9]{8}", cleaned)
        return match.group(0) if match else cleaned

    # name, surname, firstname, and anything else -> keep as-is,
    # just trimmed and collapsed whitespace
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# 10. RUN OCR OVER EVERY FIELD IN THE TEMPLATE
# ============================================================

def extract_all_fields(template, incoming_image, homography):

    template_w = template["template_image"]["width"]
    template_h = template["template_image"]["height"]

    results = []

    for field in template["fields"]:

        x_min, y_min, x_max, y_max = project_field_box(
            field, homography, template_w, template_h
        )

        crop = safe_crop(incoming_image, x_min, y_min, x_max, y_max)

        psm = field.get("ocr_psm", 7)
        raw_text, confidence = ocr_field_crop(crop, psm=psm)

        cleaned_value = clean_field_value(field["id"], raw_text)

        results.append({
            "field_id": field["id"],
            "label": field["label"],
            "value": cleaned_value,
            "raw_ocr_text": raw_text,
            "confidence": round(confidence, 1),
            "box_in_image": (x_min, y_min, x_max, y_max)
        })

    return results


# ============================================================
# 11. DRAW BOUNDING BOXES + CONFIDENCE ON THE IMAGE
# ============================================================

def draw_results_on_image(incoming_image, results, output_path):

    annotated = incoming_image.copy()

    for r in results:
        x_min, y_min, x_max, y_max = r["box_in_image"]

        color = (0, 255, 0) if r["confidence"] >= 60 else (0, 0, 255)

        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)

        label_text = f"{r['label']}: {r['confidence']:.0f}%"

        cv2.putText(
            annotated,
            label_text,
            (x_min, max(y_min - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    print(f"Annotated result image saved: {output_path}")


# ============================================================
# 12. DISPLAY FINAL FIELD NAME -> VALUE OUTPUT
# ============================================================

def print_extracted_fields(results):

    print("\n" + "=" * 50)
    print(" EXTRACTED FIELDS")
    print("=" * 50)

    for r in results:
        print(f"{r['label']:20s} : {r['value']:30s} "
              f"(confidence: {r['confidence']:.1f}%)")

    print("=" * 50 + "\n")


# ============================================================
# 13. END-TO-END PIPELINE
# ============================================================

def run_ocr_pipeline(template_json_path, incoming_image_path,
                      output_annotated_image_path):

    template = load_template(template_json_path)

    incoming_image = load_and_preprocess_incoming_image(incoming_image_path)

    homography = compute_alignment_homography(template, incoming_image)

    if homography is None:
        print("Could not align incoming document to template. Aborting.")
        return None

    results = extract_all_fields(template, incoming_image, homography)

    draw_results_on_image(incoming_image, results, output_annotated_image_path)

    print_extracted_fields(results)

    return results


# ============================================================
# 14. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # HARDCODED SECTION START
    # Change these three values to switch document type/variant
    # and the input image being tested.
    # ========================================================

    DOCUMENT_TYPE = "aadhaar"                       # HARDCODED
    VARIANT = "aadhaar_new"                      # HARDCODED

    TEMPLATE_JSON_PATH = "annotations/aadhaar_new.json"   # HARDCODED
    INCOMING_IMAGE_PATH = "templates/aadhar_new.jpg" # HARDCODED

    OUTPUT_ANNOTATED_IMAGE_PATH = "results/aadhaar_test_result.jpg"

    # ========================================================
    # HARDCODED SECTION END
    # ========================================================

    run_ocr_pipeline(
        TEMPLATE_JSON_PATH,
        INCOMING_IMAGE_PATH,
        OUTPUT_ANNOTATED_IMAGE_PATH
    )