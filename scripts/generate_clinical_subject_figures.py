"""
generate_clinical_subject_figures.py
------------------------------------
Generates publication-quality clinical subject comparison figures
for both internal institutional radiographs and external multi-center DENTEX benchmarks.
Follows the exact 4-quadrant FDI color convention established in the Clinical Decision Support GUI:
  - Quadrant 1 (Maxillary Right, FDI 18-11): Red (#EB3C3C -> RGB 235, 60, 60)
  - Quadrant 2 (Maxillary Left,  FDI 21-28): Blue (#3296FF -> RGB 50, 150, 255)
  - Quadrant 3 (Mandibular Left, FDI 31-38): Green (#28CD64 -> RGB 40, 205, 100)
  - Quadrant 4 (Mandibular Right, FDI 41-48): Orange (#F5A523 -> RGB 245, 165, 35)

Presented with a 2x2 anatomical arch grid legend to avoid wasting horizontal space and ensuring
the panoramic radiograph spans edge-to-edge with zero wasted side margins.
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from ultralytics import YOLO, RTDETR

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Image selections
# Patient 220: exactly matches GUI screenshot (30 teeth, impacted 48 in Q4)
own_img_path = BASE_DIR.parent / "yolov4" / "darknet" / "data" / "220.jpg"

# DENTEX multi-center benchmark image
dentex_imgs = sorted((BASE_DIR.parent / "yolov4" / "yolo11" / "external" / "dentex_yolo" / "images").glob("*.png"))
dentex_img_path = dentex_imgs[15]  # train_111.png (31 teeth)

# 2. Checkpoints
yolo_pt = BASE_DIR.parent / "yolov4" / "yolo11" / "runs" / "yolo11m_teeth_1280" / "weights" / "best.pt"
rtdetr_pt = BASE_DIR.parent / "yolov4" / "yolo11" / "runs" / "bench_rtdetr_l" / "weights" / "best.pt"

print("Loading models...")
yolo11m = YOLO(str(yolo_pt))
rtdetr_l = RTDETR(str(rtdetr_pt))

# Exact quadrant color convention from GUI v2
QUAD_COLORS_RGB = {
    "1": (235, 60, 60),    # Q1: Red (#EB3C3C) - Maxillary Right
    "2": (50, 150, 255),   # Q2: Blue (#3296FF) - Maxillary Left
    "3": (40, 205, 100),   # Q3: Green (#28CD64) - Mandibular Left
    "4": (245, 165, 35),   # Q4: Orange (#F5A523) - Mandibular Right
}


def predict_and_annotate(model, img_path):
    res = model.predict(str(img_path), imgsz=1280, conf=0.5, iou=0.5, verbose=False, device=0)[0]
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]

    # Dynamic line and font scaling
    box_thick = max(2, int(w_img / 1000))
    font_scale = max(0.42, w_img / 3800.0)
    font_thick = max(1, int(box_thick / 1.5))

    for b in res.boxes:
        c = int(b.cls.item())
        name = model.names[c].split("-")[0]
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]

        quad = name[0] if name and name[0] in QUAD_COLORS_RGB else "1"
        color = QUAD_COLORS_RGB[quad]

        # Draw bounding box
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, box_thick)

        # Draw filled FDI pill header
        label = f"{name}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        pill_h = th + 6
        pill_w = tw + 6

        # Position pill above box, or flip inside if near top edge
        pill_y1 = max(0, y1 - pill_h) if y1 >= pill_h else y1
        pill_y2 = pill_y1 + pill_h
        pill_x1 = max(0, x1)
        pill_x2 = min(w_img, pill_x1 + pill_w)

        cv2.rectangle(img_rgb, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)
        
        # High-contrast text: white on Q1/Q2/Q3, black on Q4 (Orange)
        text_color = (0, 0, 0) if quad == "4" else (255, 255, 255)
        cv2.putText(
            img_rgb,
            label,
            (pill_x1 + 3, pill_y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thick,
            cv2.LINE_AA,
        )

    return img_rgb, len(res.boxes), (h_img, w_img)


def add_2x2_quadrant_legend(fig):
    # In Matplotlib ncol=2 fills column 1 (indices 0, 1) then column 2 (indices 2, 3)
    # Column 1: Top-Left (Q1), Bottom-Left (Q4)
    # Column 2: Top-Right (Q2), Bottom-Right (Q3)
    q1_patch = mpatches.Patch(color="#EB3C3C", label="Quadrant 1: Maxillary Right (Teeth 18–11)")
    q4_patch = mpatches.Patch(color="#F5A523", label="Quadrant 4: Mandibular Right (Teeth 48–41)")
    q2_patch = mpatches.Patch(color="#3296FF", label="Quadrant 2: Maxillary Left (Teeth 21–28)")
    q3_patch = mpatches.Patch(color="#28CD64", label="Quadrant 3: Mandibular Left (Teeth 31–38)")
    
    fig.legend(
        handles=[q1_patch, q4_patch, q2_patch, q3_patch],
        loc="lower center",
        ncol=2,
        fontsize=10.5,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        bbox_to_anchor=(0.5, 0.01),
    )


def render_figure(img_path, title_a, title_b, out_file):
    img_a, n_a, (h, w) = predict_and_annotate(yolo11m, img_path)
    img_b, n_b, _ = predict_and_annotate(rtdetr_l, img_path)

    aspect = w / h
    fig_w = 12.0
    # Height dynamically computed to eliminate any wasted margins on sides
    fig_h = (fig_w / aspect) * 2.0 + 1.5

    fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h))
    plt.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.12, hspace=0.14)

    axs[0].imshow(img_a)
    axs[0].set_title(title_a.format(n=n_a), fontsize=12, fontweight="bold", pad=6)
    axs[0].axis("off")

    axs[1].imshow(img_b)
    axs[1].set_title(title_b.format(n=n_b), fontsize=12, fontweight="bold", pad=6)
    axs[1].axis("off")

    add_2x2_quadrant_legend(fig)
    plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"✅ Generated edge-to-edge figure: {out_file}")


def main():
    print(f"Generating internal subject evaluation for Patient 220 ({own_img_path.name})...")
    render_figure(
        own_img_path,
        "(a) Institutional Cohort: YOLO11m ({n} teeth localized)",
        "(b) Institutional Cohort: Vision Transformer RT-DETR-L ({n} teeth localized)",
        OUT_DIR / "figure_subject_own.png"
    )

    print(f"Generating DENTEX external evaluation from: {dentex_img_path.name}...")
    render_figure(
        dentex_img_path,
        "(a) External Multi-Center DENTEX Benchmark: YOLO11m ({n} teeth localized)",
        "(b) External Multi-Center DENTEX Benchmark: Vision Transformer RT-DETR-L ({n} teeth localized)",
        OUT_DIR / "figure_subject_dentex.png"
    )


if __name__ == "__main__":
    main()
