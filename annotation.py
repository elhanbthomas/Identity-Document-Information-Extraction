import cv2
import numpy as np
import json
import os
from datetime import datetime

from id_preprocessing import preprocess_document

# ============================================================
# GLOBAL STATE (kept minimal, passed via a dict to callbacks)
# ============================================================

def create_state(image):
    return {
        "image": image,
        "display": image.copy(),
        "drawing": False,
        "start_point": None,
        "current_box": None,
        "annotations": [],       # list of dicts: {id, label, type, box(px)}
        "mode": "field",         # "field" or "anchor"
        "counter": 0
    }


# ============================================================
# MOUSE CALLBACK - DRAG TO DRAW BOX
# ============================================================

def mouse_callback(event, x, y, flags, state):

    if event == cv2.EVENT_LBUTTONDOWN:
        state["drawing"] = True
        state["start_point"] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if state["drawing"]:
            temp = state["display"].copy()
            cv2.rectangle(
                temp,
                state["start_point"],
                (x, y),
                (0, 255, 0) if state["mode"] == "field" else (0, 165, 255),
                2
            )
            cv2.imshow("Annotate", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        state["drawing"] = False
        end_point = (x, y)
        box = normalize_box_pixels(state["start_point"], end_point)

        if box["w"] < 5 or box["h"] < 5:
            print("Box too small, ignored.")
            return

        state["current_box"] = box
        prompt_and_store_label(state)


# ============================================================
# CONVERT TWO CLICK POINTS INTO A TOP-LEFT / WIDTH / HEIGHT BOX
# ============================================================

def normalize_box_pixels(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    return {
        "x": x_min,
        "y": y_min,
        "w": x_max - x_min,
        "h": y_max - y_min
    }


# ============================================================
# PROMPT FOR LABEL NAME IN TERMINAL, THEN STORE ANNOTATION
# ============================================================

def prompt_and_store_label(state):

    box = state["current_box"]
    box_type = state["mode"]

    label = input(f"Enter label name for this {box_type} box: ").strip()

    if not label:
        print("Empty label, box discarded.")
        return

    state["counter"] += 1
    ann_id = f"{box_type}_{state['counter']:03d}"

    state["annotations"].append({
        "id": ann_id,
        "label": label,
        "type": box_type,
        "box_px": box
    })

    draw_annotation_on_display(state, state["annotations"][-1])
    print(f"Saved: {ann_id} -> '{label}'")


# ============================================================
# DRAW A SAVED ANNOTATION PERMANENTLY ON THE DISPLAY IMAGE
# ============================================================

def draw_annotation_on_display(state, annotation):

    box = annotation["box_px"]
    color = (0, 255, 0) if annotation["type"] == "field" else (0, 165, 255)

    top_left = (box["x"], box["y"])
    bottom_right = (box["x"] + box["w"], box["y"] + box["h"])

    cv2.rectangle(state["display"], top_left, bottom_right, color, 2)
    cv2.putText(
        state["display"],
        annotation["label"],
        (box["x"], max(box["y"] - 5, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1
    )

    cv2.imshow("Annotate", state["display"])


# ============================================================
# UNDO LAST ANNOTATION
# ============================================================

def undo_last(state):

    if not state["annotations"]:
        print("Nothing to undo.")
        return

    removed = state["annotations"].pop()
    print(f"Removed: {removed['id']}")

    # Redraw everything from scratch
    state["display"] = state["image"].copy()
    for ann in state["annotations"]:
        draw_annotation_on_display(state, ann)

    cv2.imshow("Annotate", state["display"])


# ============================================================
# RUN PREPROCESSING AND RETURN THE IMAGE TO ANNOTATE
# ============================================================

def get_preprocessed_image(image_path, target_width=1200):

    print(f"Preprocessing '{image_path}' before annotation...")

    result = preprocess_document(image_path, target_width=target_width)

    # Use the color "final" image, NOT "ocr_ready".
    # Binarization destroys texture/gradient detail that
    # ORB/SIFT keypoint matching relies on for anchors.
    preprocessed_image = result["final"]

    return preprocessed_image


# ============================================================
# MAIN INTERACTIVE LOOP
# ============================================================

def run_annotation_session(preprocessed_image):

    if preprocessed_image is None:
        raise ValueError("No preprocessed image was provided.")

    state = create_state(preprocessed_image)

    cv2.namedWindow("Annotate", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Annotate", mouse_callback, state)
    cv2.imshow("Annotate", state["display"])

    print("\nControls:")
    print("  Drag mouse   -> draw a box")
    print("  'f'          -> switch to FIELD mode")
    print("  'a'          -> switch to ANCHOR mode")
    print("  'u'          -> undo last box")
    print("  's'          -> save and finish")
    print("  'q'          -> quit without saving\n")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('f'):
            state["mode"] = "field"
            print("Mode -> FIELD")

        elif key == ord('a'):
            state["mode"] = "anchor"
            print("Mode -> ANCHOR")

        elif key == ord('u'):
            undo_last(state)

        elif key == ord('s'):
            cv2.destroyAllWindows()
            return state

        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None


# ============================================================
# CONVERT PIXEL BOXES TO NORMALIZED FRACTIONAL COORDINATES
# ============================================================

def to_normalized_box(box_px, img_w, img_h):
    return {
        "x": box_px["x"] / img_w,
        "y": box_px["y"] / img_h,
        "width": box_px["w"] / img_w,
        "height": box_px["h"] / img_h
    }


# ============================================================
# EXTRACT KEYPOINT DESCRIPTORS FOR AN ANCHOR BOX (ORB)
# ============================================================

def extract_and_save_descriptors(image, box_px, save_path):

    x, y, w, h = box_px["x"], box_px["y"], box_px["w"], box_px["h"]
    crop = image[y:y + h, x:x + w]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None:
        print(f"Warning: no keypoints found for {save_path}")
        return 0

    kp_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    kp_size = np.array([kp.size for kp in keypoints], dtype=np.float32)
    kp_angle = np.array([kp.angle for kp in keypoints], dtype=np.float32)

    np.savez_compressed(
        save_path,
        keypoint_coords=kp_coords,
        keypoint_size=kp_size,
        keypoint_angle=kp_angle,
        descriptors=descriptors
    )

    return len(keypoints)


# ============================================================
# SAVE ANNOTATED IMAGE COPY TO A SEPARATE FOLDER
# ============================================================

def save_annotated_image_copy(state, output_dir, base_name):

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    cv2.imwrite(out_path, state["display"])
    return out_path


# ============================================================
# SAVE THE PREPROCESSED (UNANNOTATED) IMAGE ITSELF
# ============================================================

def save_preprocessed_image_copy(image, output_dir, base_name):

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_name}_preprocessed.jpg")
    cv2.imwrite(out_path, image)
    return out_path


# ============================================================
# BUILD FINAL JSON TEMPLATE FROM SESSION STATE
# ============================================================

def build_template_json(
    state,
    original_image_path,
    document_type,
    variant,
    descriptors_dir,
    annotated_images_dir,
    preprocessed_images_dir
):

    img_h, img_w = state["image"].shape[:2]
    base_name = os.path.splitext(os.path.basename(original_image_path))[0]

    os.makedirs(descriptors_dir, exist_ok=True)

    anchors = []
    fields = []

    for ann in state["annotations"]:

        norm_box = to_normalized_box(ann["box_px"], img_w, img_h)

        if ann["type"] == "anchor":

            descriptor_filename = f"{base_name}_{ann['id']}.npz"
            descriptor_path = os.path.join(descriptors_dir, descriptor_filename)

            kp_count = extract_and_save_descriptors(
                state["image"],
                ann["box_px"],
                descriptor_path
            )

            anchors.append({
                "id": ann["id"],
                "label": ann["label"],
                "type": "text_region",
                "box": norm_box,
                "descriptor_ref": descriptor_path,
                "keypoint_count": kp_count,
                "detector": "ORB"
            })

        else:

            fields.append({
                "id": ann["id"],
                "label": ann["label"],
                "box": norm_box,
                "value_type": "text",
                "validator_regex": None,
                "label_keywords": [],
                "ocr_lang": "eng",
                "ocr_psm": 7
            })

    annotated_image_path = save_annotated_image_copy(
        state,
        annotated_images_dir,
        base_name
    )

    preprocessed_image_path = save_preprocessed_image_copy(
        state["image"],
        preprocessed_images_dir,
        base_name
    )

    template = {
        "document_type": document_type,
        "variant": variant,
        "template_image": {
            "width": img_w,
            "height": img_h,
            "original_source_file": original_image_path,
            "preprocessed_copy": preprocessed_image_path,
            "annotated_copy": annotated_image_path
        },
        "anchors": anchors,
        "fields": fields,
        "metadata": {
            "created_by": "opencv_annotation_tool",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "sample_count_validated": 0,
            "preprocessing_applied": True
        }
    }

    return template


# ============================================================
# SAVE TEMPLATE JSON TO DISK
# ============================================================

def save_template_json(template, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    print(f"Template saved: {output_path}")


# ============================================================
# END-TO-END ENTRY POINT
# ============================================================

def annotate_document(
    image_path,
    document_type,
    variant,
    output_json_path,
    annotated_images_dir="annotated_images",
    descriptors_dir="descriptors",
    preprocessed_images_dir="preprocessed_images",
    target_width=1200
):

    # --------------------------------------------------------
    # STEP 1: Run the same preprocessing pipeline that will be
    # applied to incoming documents at runtime, so the template
    # and runtime images share the same coordinate space and
    # visual conditions.
    # --------------------------------------------------------

    preprocessed_image = get_preprocessed_image(image_path, target_width)

    # --------------------------------------------------------
    # STEP 2: Annotate the preprocessed image, not the raw file
    # --------------------------------------------------------

    state = run_annotation_session(preprocessed_image)

    if state is None:
        print("Annotation cancelled.")
        return None

    template = build_template_json(
        state,
        image_path,
        document_type,
        variant,
        descriptors_dir,
        annotated_images_dir,
        preprocessed_images_dir
    )

    save_template_json(template, output_json_path)
    return template


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    annotate_document(
        image_path="templates/passport_new.jpg",
        document_type="passport",
        variant="passport_new",
        output_json_path="annotations/passport_new.json",
        annotated_images_dir="annotated_images",
        descriptors_dir="descriptors",
        preprocessed_images_dir="preprocessed_images"
    )