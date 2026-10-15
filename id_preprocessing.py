import cv2
import numpy as np
import os

from PIL import Image, ExifTags


# ============================================================
# 1. LOAD AND VALIDATE IMAGE
# ============================================================

def load_image(path: str) -> np.ndarray:

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No such file: {path}"
        )

    try:

        # Open image using PIL
        pil_img = Image.open(path)

        # Correct EXIF orientation
        pil_img = apply_exif_orientation(pil_img)

        # Convert to RGB
        pil_img = pil_img.convert("RGB")

        # Convert RGB to OpenCV BGR
        img = cv2.cvtColor(
            np.array(pil_img),
            cv2.COLOR_RGB2BGR
        )

    except Exception as e:

        raise ValueError(
            f"Could not read image '{path}': {e}"
        )

    if img is None or img.size == 0:

        raise ValueError(
            f"Image '{path}' is empty or corrupt"
        )

    h, w = img.shape[:2]

    if h < 100 or w < 100:

        raise ValueError(
            f"Image resolution too low ({w}x{h})"
        )

    return img


# ============================================================
# 2. EXIF ORIENTATION CORRECTION
# ============================================================

def apply_exif_orientation(
    pil_img: Image.Image
) -> Image.Image:

    try:

        exif = pil_img._getexif()

        if exif is None:
            return pil_img

        orientation_key = next(
            k
            for k, v in ExifTags.TAGS.items()
            if v == "Orientation"
        )

        orientation = exif.get(
            orientation_key
        )

        rotations = {
            3: 180,
            6: 270,
            8: 90
        }

        if orientation in rotations:

            pil_img = pil_img.rotate(
                rotations[orientation],
                expand=True
            )

    except Exception:

        # If EXIF information is unavailable,
        # simply continue.
        pass

    return pil_img


# ============================================================
# 3. FIND DOCUMENT CONTOUR
# ============================================================

def find_document_contour(
    img: np.ndarray
):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edged = cv2.Canny(
        blurred,
        50,
        150
    )

    edged = cv2.dilate(
        edged,
        np.ones((3, 3), np.uint8),
        iterations=2
    )

    contours, _ = cv2.findContours(
        edged,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )[:5]

    img_area = (
        img.shape[0] *
        img.shape[1]
    )

    for c in contours:

        perimeter = cv2.arcLength(
            c,
            True
        )

        approx = cv2.approxPolyDP(
            c,
            0.02 * perimeter,
            True
        )

        area = cv2.contourArea(
            approx
        )

        # Document should be a quadrilateral
        # covering at least 20% of image.
        if (
            len(approx) == 4
            and area > 0.2 * img_area
        ):

            return approx.reshape(
                4,
                2
            )

    return None


# ============================================================
# 4. ORDER FOUR CORNER POINTS
# ============================================================

def order_points(
    pts: np.ndarray
) -> np.ndarray:

    rect = np.zeros(
        (4, 2),
        dtype="float32"
    )

    s = pts.sum(
        axis=1
    )

    # Top-left
    rect[0] = pts[
        np.argmin(s)
    ]

    # Bottom-right
    rect[2] = pts[
        np.argmax(s)
    ]

    diff = np.diff(
        pts,
        axis=1
    )

    # Top-right
    rect[1] = pts[
        np.argmin(diff)
    ]

    # Bottom-left
    rect[3] = pts[
        np.argmax(diff)
    ]

    return rect


# ============================================================
# 5. FOUR POINT PERSPECTIVE TRANSFORMATION
# ============================================================

def four_point_transform(
    img: np.ndarray,
    pts: np.ndarray
) -> np.ndarray:

    rect = order_points(
        pts.astype("float32")
    )

    (
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ) = rect

    width_a = np.linalg.norm(
        bottom_right -
        bottom_left
    )

    width_b = np.linalg.norm(
        top_right -
        top_left
    )

    max_width = max(
        int(width_a),
        int(width_b)
    )

    height_a = np.linalg.norm(
        top_right -
        bottom_right
    )

    height_b = np.linalg.norm(
        top_left -
        bottom_left
    )

    max_height = max(
        int(height_a),
        int(height_b)
    )

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype="float32"
    )

    M = cv2.getPerspectiveTransform(
        rect,
        dst
    )

    warped = cv2.warpPerspective(
        img,
        M,
        (
            max_width,
            max_height
        )
    )

    return warped


# ============================================================
# 6. CROP DOCUMENT
# ============================================================

def crop_to_document(
    img: np.ndarray
) -> np.ndarray:

    contour = find_document_contour(
        img
    )

    # If no document boundary is found,
    # keep original image.
    if contour is None:

        return img

    return four_point_transform(
        img,
        contour
    )


# ============================================================
# 7. DESKEW IMAGE
# ============================================================

def deskew(
    img: np.ndarray
) -> np.ndarray:

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.bitwise_not(
        gray
    )

    thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY |
        cv2.THRESH_OTSU
    )[1]

    coords = np.column_stack(
        np.where(thresh > 0)
    )

    if coords.shape[0] < 50:

        return img

    angle = cv2.minAreaRect(
        coords
    )[-1]

    if angle < -45:

        angle = -(90 + angle)

    else:

        angle = -angle

    # Ignore very small or very large rotations
    if (
        abs(angle) < 0.5
        or abs(angle) > 15
    ):

        return img

    h, w = img.shape[:2]

    center = (
        w // 2,
        h // 2
    )

    M = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# ============================================================
# 8. DENOISING
# ============================================================

def denoise(
    img: np.ndarray
) -> np.ndarray:

    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=7,
        hColor=7,
        templateWindowSize=7,
        searchWindowSize=21
    )


# ============================================================
# 9. GLARE / REFLECTION REDUCTION
# ============================================================

def reduce_glare(
    img: np.ndarray
) -> np.ndarray:

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(
        hsv
    )

    # Detect very bright pixels
    _, bright_mask = cv2.threshold(
        v,
        248,
        255,
        cv2.THRESH_BINARY
    )

    # Detect low saturation
    _, low_sat_mask = cv2.threshold(
        s,
        30,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Combine both conditions
    glare_mask = cv2.bitwise_and(
        bright_mask,
        low_sat_mask
    )

    # Find connected components
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            glare_mask,
            connectivity=8
        )
    )

    img_area = (
        img.shape[0] *
        img.shape[1]
    )

    max_blob_area = (
        0.01 *
        img_area
    )

    clean_mask = np.zeros_like(
        glare_mask
    )

    for i in range(
        1,
        num_labels
    ):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area <= max_blob_area:

            clean_mask[
                labels == i
            ] = 255

    total_glare_fraction = (
        cv2.countNonZero(
            clean_mask
        )
        / float(img_area)
    )

    # Do nothing if no glare
    # or too much area is detected.
    if (
        cv2.countNonZero(
            clean_mask
        ) == 0
        or total_glare_fraction > 0.08
    ):

        return img

    clean_mask = cv2.dilate(
        clean_mask,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    inpainted = cv2.inpaint(
        img,
        clean_mask,
        inpaintRadius=5,
        flags=cv2.INPAINT_TELEA
    )

    return inpainted


# ============================================================
# 10. CLAHE CONTRAST ENHANCEMENT
# ============================================================

def enhance_contrast(
    img: np.ndarray
) -> np.ndarray:

    # Convert BGR -> LAB
    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    l_eq = clahe.apply(
        l
    )

    merged = cv2.merge(
        (
            l_eq,
            a,
            b
        )
    )

    result = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )

    return result


# ============================================================
# 11. STANDARDIZE IMAGE SIZE
# ============================================================

def standardize_size(
    img: np.ndarray,
    target_width: int = 1200
) -> np.ndarray:

    h, w = img.shape[:2]

    if w == target_width:

        return img

    scale = (
        target_width /
        float(w)
    )

    new_size = (
        target_width,
        int(h * scale)
    )

    if scale < 1:

        interpolation = cv2.INTER_AREA

    else:

        interpolation = cv2.INTER_CUBIC

    return cv2.resize(
        img,
        new_size,
        interpolation=interpolation
    )


# ============================================================
# 12. OCR BINARIZATION
# ============================================================

def binarize_for_ocr(
    img: np.ndarray
) -> np.ndarray:

    # Convert to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce noise while preserving edges
    gray = cv2.bilateralFilter(
        gray,
        9,
        75,
        75
    )

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    return binary


# ============================================================
# 13. COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_document(
    input_path: str,
    target_width: int = 1200
) -> dict:

    steps = {}

    # --------------------------------------------------------
    # STEP 1: LOAD
    # --------------------------------------------------------

    print("1. Loading image...")

    img = load_image(
        input_path
    )

    steps["0_loaded"] = img

    # --------------------------------------------------------
    # STEP 2: CROP / PERSPECTIVE
    # --------------------------------------------------------

    print("2. Detecting document...")

    cropped = crop_to_document(
        img
    )

    steps["1_cropped"] = cropped

    # --------------------------------------------------------
    # STEP 3: DESKEW
    # --------------------------------------------------------

    print("3. Correcting skew...")

    deskewed = deskew(
        cropped
    )

    steps["2_deskewed"] = deskewed

    # --------------------------------------------------------
    # STEP 4: DENOISE
    # --------------------------------------------------------

    print("4. Removing noise...")

    denoised = denoise(
        deskewed
    )

    steps["3_denoised"] = denoised

    # --------------------------------------------------------
    # STEP 5: GLARE
    # --------------------------------------------------------

    print("5. Reducing glare...")

    deglared = reduce_glare(
        denoised
    )

    steps["4_deglared"] = deglared

    # --------------------------------------------------------
    # STEP 6: CONTRAST
    # --------------------------------------------------------

    print("6. Enhancing contrast...")

    enhanced = enhance_contrast(
        deglared
    )

    steps["5_enhanced"] = enhanced

    # --------------------------------------------------------
    # STEP 7: RESIZE
    # --------------------------------------------------------

    print("7. Standardizing size...")

    final = standardize_size(
        enhanced,
        target_width
    )

    steps["6_final"] = final

    # --------------------------------------------------------
    # STEP 8: OCR IMAGE
    # --------------------------------------------------------

    print("8. Creating OCR-ready image...")

    ocr_ready = binarize_for_ocr(
        final
    )

    steps["7_ocr_ready"] = ocr_ready

    return {
        "final": final,
        "ocr_ready": ocr_ready,
        "steps": steps
    }


# ============================================================
# 14. OCR FUNCTION
# ============================================================

def extract_text(
    image
):

    try:

        import pytesseract

    except ImportError:

        raise ImportError(
            "pytesseract is not installed.\n\n"
            "Run:\n"
            "python -m pip install pytesseract"
        )

    # --------------------------------------------------------
    # TESSERACT LOCATION
    # --------------------------------------------------------

    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]

    found_tesseract = False

    for path in tesseract_paths:

        if os.path.exists(path):

            pytesseract.pytesseract.tesseract_cmd = path

            found_tesseract = True

            break

    if not found_tesseract:

        # Try system PATH
        try:

            pytesseract.get_tesseract_version()

            found_tesseract = True

        except Exception:

            raise RuntimeError(
                "Tesseract OCR was not found.\n\n"
                "Please install Tesseract OCR on Windows."
            )

    # --------------------------------------------------------
    # OCR CONFIGURATION
    # --------------------------------------------------------

    config = "--oem 3 --psm 6"

    # --------------------------------------------------------
    # PERFORM OCR
    # --------------------------------------------------------

    text = pytesseract.image_to_string(
        image,
        lang="eng",
        config=config
    )

    return text.strip()


# ============================================================
# 15. FILE EXPLORER + OCR
# ============================================================

def main():

    import tkinter as tk
    from tkinter import filedialog, messagebox

    # --------------------------------------------------------
    # CREATE TKINTER WINDOW
    # --------------------------------------------------------

    root = tk.Tk()

    root.withdraw()

    # --------------------------------------------------------
    # OPEN FILE EXPLORER
    # --------------------------------------------------------

    input_path = filedialog.askopenfilename(
        title="Select Image for Preprocessing and OCR",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"
            ),
            (
                "JPEG Files",
                "*.jpg *.jpeg"
            ),
            (
                "PNG Files",
                "*.png"
            ),
            (
                "All Files",
                "*.*"
            )
        ]
    )

    # --------------------------------------------------------
    # IF USER CANCELS
    # --------------------------------------------------------

    if not input_path:

        print("No image selected.")

        root.destroy()

        return

    print()
    print("=" * 65)
    print(" IDENTITY DOCUMENT PREPROCESSING + OCR")
    print("=" * 65)
    print()

    print("Selected image:")
    print(input_path)

    print()

    # --------------------------------------------------------
    # OUTPUT FILENAMES
    # --------------------------------------------------------

    folder = os.path.dirname(
        input_path
    )

    filename = os.path.basename(
        input_path
    )

    base_name = os.path.splitext(
        filename
    )[0]

    processed_path = os.path.join(
        folder,
        base_name +
        "_processed.jpg"
    )

    ocr_image_path = os.path.join(
        folder,
        base_name +
        "_ocr_ready.jpg"
    )

    text_path = os.path.join(
        folder,
        base_name +
        "_text.txt"
    )

    try:

        # ====================================================
        # PREPROCESS
        # ====================================================

        result = preprocess_document(
            input_path
        )

        final_image = result[
            "final"
        ]

        ocr_ready = result[
            "ocr_ready"
        ]

        # ====================================================
        # SAVE PROCESSED IMAGE
        # ====================================================

        print()
        print("Saving processed image...")

        if not cv2.imwrite(
            processed_path,
            final_image
        ):

            raise RuntimeError(
                "Could not save processed image."
            )

        # ====================================================
        # SAVE OCR IMAGE
        # ====================================================

        print("Saving OCR-ready image...")

        if not cv2.imwrite(
            ocr_image_path,
            ocr_ready
        ):

            raise RuntimeError(
                "Could not save OCR-ready image."
            )

        # ====================================================
        # OCR
        # ====================================================

        print()
        print("Running OCR...")
        print()

        text = extract_text(
            ocr_ready
        )

        # ====================================================
        # SAVE TEXT
        # ====================================================

        with open(
            text_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(text)

        # ====================================================
        # DISPLAY TEXT IN TERMINAL
        # ====================================================

        print("=" * 65)
        print(" EXTRACTED TEXT")
        print("=" * 65)
        print()

        if text:

            print(text)

        else:

            print(
                "No text was detected."
            )

        print()
        print("=" * 65)
        print(" OUTPUT FILES")
        print("=" * 65)

        print()
        print(
            "Processed image:"
        )
        print(processed_path)

        print()
        print(
            "OCR-ready image:"
        )
        print(ocr_image_path)

        print()
        print(
            "Extracted text:"
        )
        print(text_path)

        # ====================================================
        # TEXT DISPLAY WINDOW
        # ====================================================

        text_window = tk.Toplevel()

        text_window.title(
            "Extracted Text"
        )

        text_window.geometry(
            "900x650"
        )

        # Heading
        heading = tk.Label(
            text_window,
            text="Extracted Text",
            font=(
                "Arial",
                18,
                "bold"
            )
        )

        heading.pack(
            pady=10
        )

        # Text box
        text_box = tk.Text(
            text_window,
            wrap=tk.WORD,
            font=(
                "Arial",
                14
            )
        )

        text_box.pack(
            fill=tk.BOTH,
            expand=True,
            padx=15,
            pady=10
        )

        # Insert OCR result
        text_box.insert(
            tk.END,
            text
        )

        # Allow user to select/copy text
        text_box.config(
            state=tk.NORMAL
        )

        # Close button
        close_button = tk.Button(
            text_window,
            text="Close",
            command=text_window.destroy,
            font=(
                "Arial",
                12
            ),
            padx=20,
            pady=5
        )

        close_button.pack(
            pady=10
        )

        # Show success message
        messagebox.showinfo(
            "OCR Complete",
            "Processing and OCR completed successfully!\n\n"
            "Text has been saved to:\n\n"
            + text_path
        )

        # Keep text window open
        text_window.mainloop()

    except Exception as e:

        print()
        print("=" * 65)
        print(" ERROR")
        print("=" * 65)
        print()
        print(str(e))

        messagebox.showerror(
            "Processing Error",
            str(e)
        )

    finally:

        try:
            root.destroy()

        except:
            pass


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()