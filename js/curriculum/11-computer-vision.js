/* Track 11 - Computer vision */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'vision',
title: 'Computer Vision',
icon: 'CV',
level: 'Advanced',
blurb: 'Images as data, classical OpenCV techniques, object detection with YOLO, semantic and instance segmentation, vision transformers and CLIP, plus OCR, faces and video.',
intro: `
## The task hierarchy

~~~text
CLASSIFICATION          "what is in this image?"           -> one label
   |                     cat
   v
LOCALISATION            "where is the one object?"         -> one box
   |                     cat at (x, y, w, h)
   v
OBJECT DETECTION        "what and where, for everything?"  -> many boxes
   |                     cat (10,20,80,90), dog (150,30,90,120)
   v
SEMANTIC SEGMENTATION   "label every pixel by class"       -> a class mask
   |                     these pixels are cat, those are dog
   v
INSTANCE SEGMENTATION   "label every pixel by object"      -> per-object masks
                         cat #1 pixels, cat #2 pixels
~~~

Alongside those: **depth estimation**, **pose estimation**, **optical flow**, **tracking**,
**OCR**, **image generation** and **image-text matching**.

## Classical CV still matters

Deep learning did not delete OpenCV. Classical techniques are still the right tool when:

- The problem is geometric (calibration, homography, stereo, stitching)
- You need speed on a CPU or embedded device
- You have almost no training data
- You need a preprocessing step before a network
- The rule genuinely is simple (find the red circle on a white background)
`,
topics: [

/* ============================================================ */
{
id: 'image-basics',
title: 'Images as data, and OpenCV',
summary: 'Pixels, colour spaces, channels and the OpenCV operations you will use constantly - loading, resizing, thresholding, filtering and morphology.',
tags: ['vision', 'opencv', 'fundamentals'],
intro: `
## An image is a NumPy array

~~~text
GREYSCALE          shape (H, W)          values 0-255 (uint8) or 0.0-1.0 (float)
COLOUR             shape (H, W, 3)       three channels
BATCH (PyTorch)    shape (N, C, H, W)    channels FIRST
BATCH (TF/Keras)   shape (N, H, W, C)    channels LAST
~~~

:::danger OpenCV loads images as BGR, not RGB
~cv2.imread~ returns channels in **blue, green, red** order for historical reasons.
matplotlib, PIL and every deep learning model expect **RGB**. Forgetting to convert
produces images with blue and red swapped - and models that quietly underperform.

~~~python
img = cv2.imread("photo.jpg")                    # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # now RGB
~~~
:::

## Colour spaces, and when each is useful

| Space | Channels | Use for |
|---|---|---|
| **RGB / BGR** | red, green, blue | Display, neural networks |
| **HSV** | hue, saturation, value | **Colour-based segmentation** - hue is lighting-independent |
| **LAB** | lightness, a, b | Perceptually uniform; colour correction |
| **Greyscale** | intensity | Edges, contours, most classical CV |
| **YCrCb** | luma + chroma | Compression, skin detection |

:::tip Use HSV to find things by colour
In RGB, a red object under different lighting spans a huge range of values. In HSV the
**hue** stays roughly constant and only value changes. That single fact makes colour
thresholding practical.
:::
`,
keyPoints: [
  'OpenCV is BGR; everything else is RGB. Convert immediately after loading.',
  'HSV separates colour from brightness, which makes colour thresholding robust.',
  'Morphological opening removes small noise; closing fills small holes.',
  'Gaussian blur before edge detection - Canny is very sensitive to noise.'
],
pitfalls: [
  'Forgetting the BGR to RGB conversion.',
  'Resizing without preserving the aspect ratio, distorting the content.',
  'Using uint8 arithmetic that silently wraps around at 255.',
  'Applying a fixed threshold to images with varying lighting - use adaptive thresholding.'
],
levels: [
{
name: 'The OpenCV toolkit',
goal: 'Load, transform, threshold, filter and analyse images with the operations you will use every day.',
md: `
~~~bash
pip install opencv-python matplotlib numpy
~~~

~~~python opencv_basics.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. LOADING AND THE BGR TRAP
# =====================================================================
# make a test image so this runs without a file
img_bgr = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.rectangle(img_bgr, (50, 50), (150, 150), (255, 0, 0), -1)     # BLUE in BGR
cv2.circle(img_bgr, (280, 100), 55, (0, 0, 255), -1)              # RED in BGR
cv2.putText(img_bgr, "OpenCV", (60, 240), cv2.FONT_HERSHEY_SIMPLEX,
            1.4, (0, 255, 0), 3)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].imshow(img_bgr); ax[0].set_title("shown WITHOUT conversion (wrong)")
ax[1].imshow(img_rgb); ax[1].set_title("after BGR2RGB (correct)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()

print(f"shape : {img_rgb.shape}   (height, width, channels)")
print(f"dtype : {img_rgb.dtype}   range {img_rgb.min()}-{img_rgb.max()}")
print(f"pixel at (100, 100): {img_rgb[100, 100]}  (R, G, B)")

# =====================================================================
# 2. THE UINT8 OVERFLOW TRAP
# =====================================================================
a = np.uint8([250])
print(f"\\nuint8 arithmetic: 250 + 10 = {a + 10}   <- WRAPPED AROUND")
print(f"cv2.add is saturating: {cv2.add(a, np.uint8([10]))}   <- clipped at 255")
print("Use cv2 arithmetic, or convert to int/float first.")

# =====================================================================
# 3. GEOMETRIC TRANSFORMS
# =====================================================================
h, w = img_rgb.shape[:2]

# resize - preserve aspect ratio unless you have a reason not to
def resize_keep_aspect(image, target_width):
    ratio = target_width / image.shape[1]
    return cv2.resize(image, (target_width, int(image.shape[0] * ratio)),
                      interpolation=cv2.INTER_AREA)      # INTER_AREA for shrinking

small = resize_keep_aspect(img_rgb, 200)
big = cv2.resize(img_rgb, None, fx=1.5, fy=1.5,
                 interpolation=cv2.INTER_CUBIC)          # INTER_CUBIC for growing

# rotate about the centre
M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle=30, scale=1.0)
rotated = cv2.warpAffine(img_rgb, M_rot, (w, h))

# translate
M_shift = np.float32([[1, 0, 60], [0, 1, 30]])
shifted = cv2.warpAffine(img_rgb, M_shift, (w, h))

# flip
flipped = cv2.flip(img_rgb, 1)                            # 1 horizontal, 0 vertical

# crop is just NumPy slicing
cropped = img_rgb[40:170, 40:170]

# perspective transform - the one used for document scanning
src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst = np.float32([[40, 20], [w - 20, 0], [w, h - 30], [0, h]])
warped = cv2.warpPerspective(img_rgb, cv2.getPerspectiveTransform(src, dst), (w, h))

fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for a, (im, t) in zip(ax.ravel(), [
        (img_rgb, "original"), (rotated, "rotated 30 degrees"),
        (shifted, "translated"), (flipped, "flipped"),
        (cropped, "cropped"), (warped, "perspective warp")]):
    a.imshow(im); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()

# =====================================================================
# 4. COLOUR SPACES - and why HSV wins for colour selection
# =====================================================================
hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# select the RED circle by hue. Red wraps around 0/180 in OpenCV's hue scale.
lower1, upper1 = np.array([0, 120, 70]), np.array([10, 255, 255])
lower2, upper2 = np.array([170, 120, 70]), np.array([180, 255, 255])
mask_red = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
only_red = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_red)

fig, ax = plt.subplots(1, 5, figsize=(19, 4))
for a, (im, t, cm) in zip(ax, [
        (img_rgb, "RGB", None), (hsv[:, :, 0], "hue channel", "hsv"),
        (gray, "greyscale", "gray"), (mask_red, "red mask", "gray"),
        (only_red, "red extracted", None)]):
    a.imshow(im, cmap=cm); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()

print("\\nOpenCV HSV RANGES (note: hue is 0-179, not 0-359)")
print("  red    : 0-10 and 170-179")
print("  orange : 10-25")
print("  yellow : 25-35")
print("  green  : 35-85")
print("  blue   : 85-130")
print("  purple : 130-170")
~~~

### Filtering, thresholding and morphology

~~~python filtering.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# a noisy test image
rng = np.random.default_rng(0)
clean = np.zeros((300, 300), np.uint8)
cv2.circle(clean, (150, 150), 80, 255, -1)
cv2.rectangle(clean, (30, 30), (90, 90), 200, -1)
noisy = np.clip(clean.astype(int) + rng.normal(0, 35, clean.shape), 0, 255).astype(np.uint8)
# salt-and-pepper noise
sp = noisy.copy()
coords = rng.random(sp.shape)
sp[coords < 0.03] = 0
sp[coords > 0.97] = 255

# =====================================================================
# SMOOTHING
# =====================================================================
filters = {
    "noisy input":       sp,
    "box blur 5x5":      cv2.blur(sp, (5, 5)),
    "gaussian 5x5":      cv2.GaussianBlur(sp, (5, 5), 0),
    "median 5x5":        cv2.medianBlur(sp, 5),
    "bilateral":         cv2.bilateralFilter(sp, 9, 75, 75),
}
fig, ax = plt.subplots(1, 5, figsize=(19, 4))
for a, (name, im) in zip(ax, filters.items()):
    a.imshow(im, cmap="gray"); a.set_title(name); a.axis("off")
plt.tight_layout(); plt.show()

print("SMOOTHING FILTERS")
print("  box       : fast, but blurs edges badly")
print("  gaussian  : weighted by distance - the default smoother")
print("  median    : BEST for salt-and-pepper noise, preserves edges")
print("  bilateral : smooths within regions but PRESERVES edges. Slow but excellent.")

# =====================================================================
# THRESHOLDING
# =====================================================================
# an image with a lighting gradient - where global thresholds fail
gradient = np.tile(np.linspace(0, 150, 300, dtype=np.uint8), (300, 1))
uneven = cv2.add(clean, gradient)

_, global_th = cv2.threshold(uneven, 127, 255, cv2.THRESH_BINARY)
_, otsu_th = cv2.threshold(uneven, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive_mean = cv2.adaptiveThreshold(uneven, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 31, 5)
adaptive_gauss = cv2.adaptiveThreshold(uneven, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 5)

fig, ax = plt.subplots(1, 5, figsize=(19, 4))
for a, (im, t) in zip(ax, [
        (uneven, "uneven lighting"), (global_th, "global threshold 127"),
        (otsu_th, "Otsu (auto global)"), (adaptive_mean, "adaptive mean"),
        (adaptive_gauss, "adaptive gaussian")]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()

print("\\nA GLOBAL threshold fails under uneven lighting.")
print("ADAPTIVE thresholding computes a local threshold per neighbourhood.")
print("OTSU picks the global threshold automatically by maximising")
print("between-class variance - excellent when lighting IS even.")

# =====================================================================
# MORPHOLOGY - shape operations on binary images
# =====================================================================
binary = cv2.threshold(sp, 127, 255, cv2.THRESH_BINARY)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

ops = {
    "binary (noisy)": binary,
    "erode":     cv2.erode(binary, kernel, iterations=1),
    "dilate":    cv2.dilate(binary, kernel, iterations=1),
    "opening":   cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
    "closing":   cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel),
    "gradient":  cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel),
}
fig, ax = plt.subplots(1, 6, figsize=(19, 3.6))
for a, (name, im) in zip(ax, ops.items()):
    a.imshow(im, cmap="gray"); a.set_title(name); a.axis("off")
plt.tight_layout(); plt.show()

print("""
MORPHOLOGY CHEAT SHEET
  ERODE     shrinks white regions      -> removes small white specks
  DILATE    grows white regions        -> fills small black holes
  OPENING   erode then dilate          -> REMOVES SMALL NOISE, keeps shape size
  CLOSING   dilate then erode          -> FILLS SMALL HOLES, keeps shape size
  GRADIENT  dilate minus erode         -> the outline

  The standard binary-image cleanup is: OPENING then CLOSING.
""")

# =====================================================================
# EDGES AND CONTOURS
# =====================================================================
img = np.zeros((300, 400), np.uint8)
cv2.circle(img, (100, 150), 60, 255, -1)
cv2.rectangle(img, (200, 80), (330, 220), 255, -1)
cv2.ellipse(img, (250, 260), (70, 25), 0, 0, 360, 180, -1)

blurred = cv2.GaussianBlur(img, (5, 5), 0)          # ALWAYS blur before Canny
edges = cv2.Canny(blurred, 50, 150)

contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
display = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

print(f"\\nfound {len(contours)} contours")
print(f"{'#':>3} {'area':>10} {'perimeter':>11} {'bbox':>22} {'circularity':>12} shape")
print("-" * 76)
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    x, y, cw, ch = cv2.boundingRect(c)
    circularity = 4 * np.pi * area / (perim ** 2) if perim else 0
    approx = cv2.approxPolyDP(c, 0.02 * perim, True)
    shape = ("circle" if circularity > 0.85 else
             "rectangle" if len(approx) == 4 else
             "ellipse" if circularity > 0.6 else f"{len(approx)}-gon")
    cv2.rectangle(display, (x, y), (x + cw, y + ch), (255, 0, 0), 2)
    print(f"{i:>3} {area:>10.0f} {perim:>11.1f} "
          f"{f'({x},{y},{cw},{ch})':>22} {circularity:>12.3f} {shape}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for a, (im, t, cm) in zip(ax, [(img, "input", "gray"),
                               (edges, "Canny edges", "gray"),
                               (display, "contours + bounding boxes", None)]):
    a.imshow(im, cmap=cm); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()
~~~

:::tip A complete classical CV pipeline
~~~text
1. Convert to greyscale or HSV
2. Blur (Gaussian or median) to suppress noise
3. Threshold (adaptive, or Otsu) to get a binary mask
4. Morphology (opening then closing) to clean the mask
5. findContours
6. Filter contours by area, aspect ratio, circularity
7. Measure or classify what remains
~~~
This solves an enormous number of industrial inspection and document problems without a
single trained parameter.
:::
`
}
],
quiz: [
{
q: 'You load an image with cv2.imread and display it with matplotlib. The colours look wrong. Why?',
options: [
  'The image is corrupted',
  'OpenCV loads BGR, matplotlib expects RGB - convert with cv2.COLOR_BGR2RGB',
  'The dtype is wrong',
  'matplotlib cannot display photographs'
],
answer: 1,
why: 'Red and blue channels are swapped. This also silently degrades any pretrained model you feed the image to, because the model expects RGB.'
},
{
q: 'You need to find all red objects under varying lighting. Which colour space?',
options: ['RGB', 'HSV, thresholding on hue', 'Greyscale', 'LAB'],
answer: 1,
why: 'HSV separates colour (hue) from brightness (value), so hue stays stable as lighting changes. In RGB, a red object spans an enormous range of values across lighting conditions.'
},
{
q: 'Your binary mask has small white specks of noise. Which morphological operation removes them?',
options: ['Dilation', 'Opening (erode then dilate)', 'Closing', 'Gradient'],
answer: 1,
why: 'Erosion removes the specks; the following dilation restores the real shapes to their original size. Closing does the opposite - it fills small holes.'
},
{
q: 'Your document image has a lighting gradient and a global threshold fails. What should you use?',
options: [
  'A higher global threshold',
  'Adaptive thresholding, which computes a local threshold per neighbourhood',
  'Convert to RGB',
  'Increase the contrast'
],
answer: 1,
why: 'No single global value can work when the background brightness varies across the image. Adaptive thresholding compares each pixel to its own local neighbourhood.'
}
]
},

/* ============================================================ */
{
id: 'object-detection',
title: 'Object detection',
summary: 'Bounding boxes, IoU, non-maximum suppression, mean average precision, and running YOLO on your own data.',
tags: ['vision', 'detection', 'yolo'],
intro: `
## The problem

Classification asks *what*. Detection asks *what and where, for every object*, with an
unknown number of objects per image.

~~~text
OUTPUT PER OBJECT
  class label      "dog"
  confidence       0.93
  bounding box     (x, y, width, height)  or  (x1, y1, x2, y2)
~~~

## The two families

~~~text
TWO-STAGE (R-CNN family)              ONE-STAGE (YOLO, SSD, RetinaNet)
  1. propose candidate regions          Predict boxes and classes DIRECTLY
  2. classify and refine each one       in one forward pass

  slower, historically more accurate    much faster, now equally accurate
  Faster R-CNN, Mask R-CNN              YOLO v5/v8/v11, RetinaNet, DETR
~~~

For nearly every practical project today, **use YOLO**. It is fast, accurate, and the
tooling is excellent.

## The two metrics you must understand

:::math Intersection over Union
**IoU = area of overlap / area of union**

- IoU > 0.5 - conventionally "a correct detection"
- IoU > 0.75 - a strict criterion
- IoU = 1.0 - a perfect box
:::

:::math mean Average Precision
For each class, compute the precision-recall curve across confidence thresholds, take the
area under it (**AP**), then average over classes (**mAP**).

- **mAP@0.5** - a detection counts if IoU > 0.5 (the classic PASCAL VOC metric)
- **mAP@0.5:0.95** - averaged over IoU thresholds 0.5 to 0.95 (the COCO metric, stricter)
:::

## Non-maximum suppression

A detector fires many overlapping boxes for the same object. NMS keeps the most confident
and deletes its heavy overlaps.

~~~text
1. Sort all boxes by confidence, descending
2. Take the top box, add it to the output
3. Delete every remaining box whose IoU with it exceeds a threshold (usually 0.45)
4. Repeat with the next remaining box
~~~
`,
keyPoints: [
  'IoU measures box overlap; 0.5 is the conventional threshold for a correct detection.',
  'NMS removes duplicate boxes for the same object.',
  'mAP@0.5:0.95 is the standard COCO metric and is far stricter than mAP@0.5.',
  'For a new project, fine-tune a pretrained YOLO rather than training from scratch.'
],
pitfalls: [
  'Mixing box formats - (x, y, w, h) versus (x1, y1, x2, y2) versus normalised centre format.',
  'Setting the NMS IoU threshold too high, leaving duplicate boxes.',
  'Evaluating with mAP@0.5 and reporting it as if it were the COCO metric.',
  'Training on images at a different resolution from inference.'
],
levels: [
{
name: 'IoU, NMS and mAP from scratch',
goal: 'Implement the three core detection algorithms so the metrics stop being opaque.',
md: `
~~~python detection_core.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# =====================================================================
# BOX FORMATS - a constant source of bugs
# =====================================================================
def xywh_to_xyxy(box):
    """(x, y, width, height) with (x,y) = top-left  ->  (x1, y1, x2, y2)"""
    x, y, w, h = box
    return np.array([x, y, x + w, y + h])


def cxcywh_to_xyxy(box):
    """(centre_x, centre_y, w, h) - the YOLO format -> (x1, y1, x2, y2)"""
    cx, cy, w, h = box
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


def normalise(box_xyxy, img_w, img_h):
    """Pixel coordinates -> 0-1, which is what most training formats use."""
    x1, y1, x2, y2 = box_xyxy
    return np.array([x1/img_w, y1/img_h, x2/img_w, y2/img_h])


# =====================================================================
# 1. INTERSECTION OVER UNION
# =====================================================================
def iou(box_a, box_b):
    """Both boxes in (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def iou_matrix(boxes_a, boxes_b):
    """Vectorised IoU between two sets of boxes -> (len(a), len(b))."""
    a = np.asarray(boxes_a, dtype=float)
    b = np.asarray(boxes_b, dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ---- see what different IoU values look like ------------------------
truth = np.array([100, 100, 200, 200])
examples = {
    "perfect":      np.array([100, 100, 200, 200]),
    "very good":    np.array([105, 105, 205, 205]),
    "acceptable":   np.array([130, 130, 230, 230]),
    "poor":         np.array([160, 160, 260, 260]),
    "no overlap":   np.array([220, 220, 320, 320]),
    "too big":      np.array([50, 50, 250, 250]),
}

fig, axes = plt.subplots(1, 6, figsize=(19, 3.6))
for ax, (name, pred) in zip(axes, examples.items()):
    ax.add_patch(patches.Rectangle((truth[0], truth[1]),
                                   truth[2]-truth[0], truth[3]-truth[1],
                                   fill=False, edgecolor="green", lw=2.5,
                                   label="ground truth"))
    ax.add_patch(patches.Rectangle((pred[0], pred[1]),
                                   pred[2]-pred[0], pred[3]-pred[1],
                                   fill=False, edgecolor="red", lw=2.5,
                                   linestyle="--", label="prediction"))
    v = iou(truth, pred)
    ax.set_xlim(30, 340); ax.set_ylim(340, 30)
    ax.set_title(f"{name}\\nIoU = {v:.3f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    print(f"  {name:12s} IoU {v:.4f}  "
          f"{'correct at 0.5' if v >= 0.5 else 'MISS at 0.5'}")
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.show()


# =====================================================================
# 2. NON-MAXIMUM SUPPRESSION
# =====================================================================
def nms(boxes, scores, iou_threshold=0.45):
    """Keep the most confident box, delete its heavy overlaps, repeat."""
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i+1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return keep


def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.05):
    """Instead of deleting overlaps, DECAY their scores.
    Better when objects genuinely overlap (a crowd of people)."""
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float).copy()
    keep, order = [], list(range(len(boxes)))

    while order:
        best = max(order, key=lambda i: scores[i])
        if scores[best] < score_threshold:
            break
        keep.append(best)
        order.remove(best)
        for j in list(order):
            v = iou(boxes[best], boxes[j])
            scores[j] *= np.exp(-(v ** 2) / sigma)      # gaussian decay
            if scores[j] < score_threshold:
                order.remove(j)
    return keep


# a detector's raw output: several boxes per object
raw_boxes = np.array([
    [100, 100, 210, 210], [105, 98, 205, 215], [95, 105, 215, 205],   # object A
    [300, 150, 400, 260], [305, 155, 395, 255],                        # object B
    [180, 300, 280, 400],                                              # object C
])
raw_scores = np.array([0.92, 0.88, 0.75, 0.95, 0.81, 0.67])

kept = nms(raw_boxes, raw_scores, iou_threshold=0.45)
print(f"\\nNMS: {len(raw_boxes)} raw boxes -> {len(kept)} kept: {kept}")
print(f"kept scores: {raw_scores[kept]}")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for a, (idxs, title) in zip(ax, [(range(len(raw_boxes)), "raw detections"),
                                 (kept, "after NMS")]):
    for i in idxs:
        b = raw_boxes[i]
        a.add_patch(patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                                      fill=False, edgecolor="red", lw=2))
        a.text(b[0], b[1]-6, f"{raw_scores[i]:.2f}", color="red", fontsize=9)
    a.set_xlim(50, 450); a.set_ylim(450, 50); a.set_title(title)
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()


# =====================================================================
# 3. MEAN AVERAGE PRECISION
# =====================================================================
def average_precision(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """AP for ONE class in ONE image (extend by looping over both)."""
    if len(pred_boxes) == 0:
        return 0.0
    order = np.argsort(pred_scores)[::-1]
    pred_boxes = np.asarray(pred_boxes)[order]

    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i, pb in enumerate(pred_boxes):
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue
        ious = iou_matrix([pb], gt_boxes)[0]
        best = int(ious.argmax())
        if ious[best] >= iou_threshold and not matched[best]:
            tp[i] = 1
            matched[best] = True             # each ground truth counts ONCE
        else:
            fp[i] = 1                        # duplicate or bad box

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recalls = tp_cum / max(len(gt_boxes), 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # 101-point interpolated AP (the COCO definition)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precisions[recalls >= t].max() if (recalls >= t).any() else 0.0
        ap += p / 101
    return ap, precisions, recalls


gt = np.array([[100, 100, 200, 200], [300, 150, 400, 250], [180, 300, 280, 400]])
preds = np.array([[102, 98, 198, 205], [305, 155, 395, 245],
                  [500, 500, 560, 560], [175, 305, 285, 395]])
scores = np.array([0.95, 0.88, 0.71, 0.62])

ap, prec, rec = average_precision(preds, scores, gt, 0.5)
print(f"\\nAP@0.5 = {ap:.4f}")

print("\\nAP AT DIFFERENT IoU THRESHOLDS (the COCO sweep)")
aps = []
for t in np.arange(0.5, 1.0, 0.05):
    a, _, _ = average_precision(preds, scores, gt, t)
    aps.append(a)
    print(f"  AP@{t:.2f} = {a:.4f}")
print(f"\\n  mAP@0.5      = {aps[0]:.4f}")
print(f"  mAP@0.5:0.95 = {np.mean(aps):.4f}   <- the standard COCO metric")
print("\\n  The COCO metric is much stricter: it demands accurate boxes,")
print("  not merely roughly-correct ones.")

plt.figure(figsize=(7, 5))
plt.plot(rec, prec, "o-", lw=2)
plt.xlabel("recall"); plt.ylabel("precision")
plt.title(f"Precision-recall curve, AP@0.5 = {ap:.3f}")
plt.grid(alpha=0.3); plt.ylim(0, 1.05); plt.tight_layout(); plt.show()
~~~
`
},
{
name: 'Training YOLO on your own data',
goal: 'Run a pretrained detector, then fine-tune it on a custom dataset end to end.',
md: `
~~~bash
pip install ultralytics
~~~

~~~python yolo_inference.py
"""Running a pretrained YOLO - detection in five lines."""
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# n=nano, s=small, m=medium, l=large, x=extra. Bigger = more accurate, slower.
model = YOLO("yolo11n.pt")          # downloads on first use

results = model("https://ultralytics.com/images/bus.jpg")

r = results[0]
print(f"detections: {len(r.boxes)}")
print(f"\\n{'class':12s} {'confidence':>11s} {'box (x1,y1,x2,y2)':>32s}")
print("-" * 58)
for box in r.boxes:
    cls = model.names[int(box.cls)]
    conf = float(box.conf)
    xyxy = box.xyxy[0].cpu().numpy().round(1)
    print(f"{cls:12s} {conf:>11.3f} {str(tuple(xyxy)):>32s}")

annotated = r.plot()                 # BGR
plt.figure(figsize=(9, 11))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis("off"); plt.title("YOLO11 detections")
plt.tight_layout(); plt.show()

# ---- the settings that matter ---------------------------------------
results = model(
    "image.jpg",
    conf=0.25,        # minimum confidence. RAISE for fewer false positives.
    iou=0.45,         # NMS IoU threshold. Lower = more aggressive suppression.
    imgsz=640,        # inference resolution. Larger finds smaller objects.
    max_det=300,
    classes=[0, 2],   # only detect these class ids (0=person, 2=car)
    device=0,         # GPU 0, or "cpu"
    half=True,        # FP16 - roughly 2x faster on GPU
    verbose=False,
)
~~~

### Preparing a custom dataset

~~~text
YOLO EXPECTS THIS LAYOUT

dataset/
  data.yaml
  images/
    train/  img001.jpg  img002.jpg ...
    val/    img101.jpg ...
    test/   img201.jpg ...
  labels/
    train/  img001.txt  img002.txt ...      <- SAME basename as the image
    val/    img101.txt ...
    test/   img201.txt ...

EACH LABEL FILE: one line per object
  class_id  x_center  y_center  width  height

  ALL FOUR COORDINATES NORMALISED TO 0-1, relative to image size.
  class_id is 0-based.

  Example (img001.txt):
    0 0.512 0.431 0.204 0.318
    2 0.783 0.652 0.117 0.245
~~~

~~~python prepare_dataset.py
"""Convert common annotation formats to YOLO, and validate the result."""
import os
import json
import shutil
from pathlib import Path
import numpy as np
import yaml


def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Pascal VOC (absolute corners) -> YOLO (normalised centre + size)."""
    x_c = ((xmin + xmax) / 2) / img_w
    y_c = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_c, y_c, w, h


def coco_to_yolo(x, y, w, h, img_w, img_h):
    """COCO (top-left x,y + w,h, absolute) -> YOLO."""
    return (x + w/2) / img_w, (y + h/2) / img_h, w / img_w, h / img_h


def yolo_to_xyxy(x_c, y_c, w, h, img_w, img_h):
    """Back to pixels, for drawing."""
    return ((x_c - w/2) * img_w, (y_c - h/2) * img_h,
            (x_c + w/2) * img_w, (y_c + h/2) * img_h)


def write_data_yaml(root, class_names, path="data.yaml"):
    cfg = {
        "path": str(Path(root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {path}:\\n{yaml.safe_dump(cfg, sort_keys=False)}")
    return cfg


def validate_dataset(root, n_classes):
    """Catch the errors that silently ruin training."""
    root = Path(root)
    problems = []
    for split in ["train", "val"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            problems.append(f"missing {img_dir}")
            continue

        images = {p.stem for p in img_dir.glob("*.[jp][pn]g")}
        labels = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else set()

        missing = images - labels
        orphan = labels - images
        if missing:
            problems.append(f"{split}: {len(missing)} images with NO label file")
        if orphan:
            problems.append(f"{split}: {len(orphan)} labels with no image")

        # check the label contents
        for lp in list(lbl_dir.glob("*.txt"))[:2000]:
            for ln, line in enumerate(lp.read_text().strip().split("\\n"), 1):
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    problems.append(f"{lp.name}:{ln} has {len(parts)} fields, expected 5")
                    continue
                cid = int(parts[0])
                coords = [float(v) for v in parts[1:]]
                if not 0 <= cid < n_classes:
                    problems.append(f"{lp.name}:{ln} class id {cid} out of range")
                if any(not 0 <= v <= 1 for v in coords):
                    problems.append(f"{lp.name}:{ln} coordinates not normalised: {coords}")
                if coords[2] <= 0 or coords[3] <= 0:
                    problems.append(f"{lp.name}:{ln} zero-size box")

        print(f"{split}: {len(images)} images, {len(labels)} label files")

    print(f"\\n{len(problems)} problem(s)" if problems else "\\ndataset looks valid")
    for p in problems[:20]:
        print("  -", p)
    return problems
~~~

### Training

~~~python train_yolo.py
from ultralytics import YOLO
import torch

# start from PRETRAINED weights - never from scratch unless you have 100k+ images
model = YOLO("yolo11s.pt")

results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,                  # -1 lets it pick automatically from GPU memory
    device=0,
    workers=8,
    patience=25,               # early stopping
    project="runs/detect",
    name="my_experiment",

    # --- optimisation ---
    optimizer="auto",          # AdamW for short runs, SGD for long ones
    lr0=0.01,                  # initial learning rate
    lrf=0.01,                  # final lr = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    cos_lr=True,

    # --- augmentation (YOLO's defaults are strong; tune for your domain) ---
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,     # colour jitter
    degrees=0.0,               # rotation. RAISE if your objects appear rotated.
    translate=0.1,
    scale=0.5,                 # zoom
    shear=0.0,
    perspective=0.0,
    flipud=0.0,                # vertical flip. 0 for photos, 0.5 for aerial imagery.
    fliplr=0.5,                # horizontal flip
    mosaic=1.0,                # combine 4 images - very effective
    mixup=0.1,
    copy_paste=0.1,
    close_mosaic=10,           # turn mosaic off for the last 10 epochs

    # --- other ---
    amp=True,                  # mixed precision
    cache="ram",               # cache images in RAM if they fit
    val=True,
    plots=True,
)

# =====================================================================
# EVALUATE
# =====================================================================
metrics = model.val(data="data.yaml", split="test")
print(f"mAP@0.5      : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
print(f"precision    : {metrics.box.mp:.4f}")
print(f"recall       : {metrics.box.mr:.4f}")

print("\\nPER-CLASS mAP@0.5:0.95")
for i, name in model.names.items():
    if i < len(metrics.box.maps):
        print(f"  {name:20s} {metrics.box.maps[i]:.4f}")

# =====================================================================
# EXPORT FOR DEPLOYMENT
# =====================================================================
model.export(format="onnx", imgsz=640, simplify=True)     # portable
# model.export(format="engine", half=True)                # TensorRT, fastest on NVIDIA
# model.export(format="coreml")                           # iOS
# model.export(format="tflite", int8=True)                # Android / edge
~~~

### Real-time video

~~~python yolo_video.py
from ultralytics import YOLO
import cv2
import time
from collections import defaultdict

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)          # 0 = webcam, or a file path

# built-in tracking: gives each object a persistent id across frames
track_history = defaultdict(list)
fps_times = []

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    t0 = time.perf_counter()
    results = model.track(frame, persist=True, conf=0.4, iou=0.5,
                          tracker="bytetrack.yaml", verbose=False)
    fps_times.append(1 / (time.perf_counter() - t0))

    r = results[0]
    annotated = r.plot()

    # count objects per class in this frame
    if r.boxes.id is not None:
        counts = defaultdict(int)
        for box in r.boxes:
            counts[model.names[int(box.cls)]] += 1
            # record the trajectory
            tid = int(box.id)
            cx, cy = box.xywh[0][:2].tolist()
            track_history[tid].append((cx, cy))

        y = 30
        for name, n in sorted(counts.items()):
            cv2.putText(annotated, f"{name}: {n}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 26

    cv2.putText(annotated, f"FPS: {np.mean(fps_times[-30:]):.1f}",
                (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
~~~

:::tip Practical detection advice
1. **200-500 labelled images per class** is usually enough to fine-tune YOLO usefully.
2. **Label consistently.** Decide early whether boxes are tight or loose, and whether
   occluded objects are labelled. Inconsistency hurts more than a smaller dataset.
3. **Match your augmentation to reality.** Turn off vertical flip for photos; turn on
   rotation for aerial or document images.
4. **Small objects need higher ~imgsz~.** Try 1024 or 1280, or tile large images.
5. **Class imbalance** hurts detection badly. Oversample rare-class images.
6. **Check the confusion matrix and the failure images**, not just mAP.
:::
`
}
],
quiz: [
{
q: 'Two boxes overlap with IoU 0.65. Under the standard PASCAL VOC criterion, is that a correct detection?',
options: ['No, it needs 0.9', 'Yes, the conventional threshold is 0.5', 'Only if confidence is above 0.9', 'IoU is irrelevant'],
answer: 1,
why: 'IoU > 0.5 is the classic threshold. The COCO metric averages AP over IoU thresholds from 0.5 to 0.95, so it is considerably stricter.'
},
{
q: 'What problem does non-maximum suppression solve?',
options: [
  'Class imbalance',
  'Detectors emit many overlapping boxes for the same object; NMS keeps the most confident and removes its duplicates',
  'Slow inference',
  'Small object detection'
],
answer: 1,
why: 'Without NMS a single object produces a cluster of boxes. Soft-NMS decays scores rather than deleting, which works better when objects genuinely overlap.'
},
{
q: 'In the YOLO label format, what do the four numbers after the class id represent?',
options: [
  'x1, y1, x2, y2 in pixels',
  'Normalised centre x, centre y, width, height - all between 0 and 1',
  'Top-left x, y and width, height in pixels',
  'Confidence values'
],
answer: 1,
why: 'YOLO uses normalised centre-format coordinates so labels are resolution-independent. Mixing up box formats is the single most common dataset preparation bug.'
},
{
q: 'Your detector misses small objects. What is the most direct fix?',
options: [
  'Lower the confidence threshold',
  'Increase the inference resolution (imgsz), or tile large images',
  'Use a smaller model',
  'Increase the NMS threshold'
],
answer: 1,
why: 'Small objects occupy few pixels after downsampling. Training and inferring at 1024 or 1280 instead of 640, or splitting large images into overlapping tiles, addresses the root cause.'
}
]
},

/* ============================================================ */
{
id: 'segmentation',
title: 'Image segmentation',
summary: 'Per-pixel classification - U-Net, semantic versus instance segmentation, the right loss functions, and the Segment Anything model.',
tags: ['vision', 'segmentation', 'unet'],
intro: `
## Three kinds of segmentation

~~~text
SEMANTIC        every pixel gets a CLASS
                all cars are the same colour

INSTANCE        every pixel gets an OBJECT id
                car #1 and car #2 are different; background is ignored

PANOPTIC        both - every pixel gets a class AND, for countable things,
                an instance id
~~~

## U-Net: the architecture that dominates

Designed for biomedical images in 2015, still the default for segmentation.

~~~text
     ENCODER (contracting)              DECODER (expanding)

  input --> [conv,conv] --------skip--------> [conv,conv] --> output mask
              |                                    ^
            pool                                upsample
              v                                    |
            [conv,conv] --------skip--------> [conv,conv]
              |                                    ^
            pool                                upsample
              v                                    |
            [conv,conv] --------skip--------> [conv,conv]
              |                                    ^
            pool                                upsample
              v                                    |
                     [bottleneck conv,conv]
~~~

:::tip Why the skip connections are essential
The encoder discards spatial detail as it downsamples. Segmentation needs that detail back
- a mask must be pixel-accurate. The skip connections carry high-resolution features
directly across to the decoder, which is exactly what a classifier does not need and a
segmenter cannot do without.
:::

## The right losses

| Loss | Use |
|---|---|
| **Cross-entropy** | Balanced classes |
| **Dice loss** | **Imbalanced** - the foreground is a small fraction of pixels |
| **Combined CE + Dice** | The usual production choice |
| **Focal loss** | Extreme imbalance |
| **Tversky / Focal Tversky** | When false negatives cost more than false positives |
`,
keyPoints: [
  'U-Net skip connections restore the spatial detail the encoder discarded.',
  'Dice loss handles the severe foreground/background imbalance typical of segmentation.',
  'IoU (Jaccard) and Dice are the standard segmentation metrics.',
  'Segment Anything (SAM) gives strong zero-shot masks from a point or box prompt.'
],
pitfalls: [
  'Using plain cross-entropy when the foreground is 2% of pixels - the model predicts all background.',
  'Resizing masks with bilinear interpolation, which invents non-existent class values. Use nearest neighbour.',
  'Applying different augmentation to the image and its mask.',
  'Evaluating with pixel accuracy, which is meaningless under imbalance.'
],
levels: [
{
name: 'Building and training a U-Net',
goal: 'Implement U-Net from scratch, choose the right loss, and train it on a segmentation task.',
md: `
~~~python unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """The repeated (conv -> BN -> ReLU) x2 block."""

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample, then CONCATENATE the skip connection, then convolve."""

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if the spatial sizes are off by one (odd input dimensions)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)          # <-- THE skip connection
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, base=64, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16 // factor)

        self.up1 = Up(base * 16, base * 8 // factor, bilinear)
        self.up2 = Up(base * 8, base * 4 // factor, bilinear)
        self.up3 = Up(base * 4, base * 2 // factor, bilinear)
        self.up4 = Up(base * 2, base, bilinear)
        self.outc = nn.Conv2d(base, n_classes, kernel_size=1)   # LOGITS

    def forward(self, x):
        x1 = self.inc(x)          # keep every encoder output for the skips
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


model = UNet(in_channels=3, n_classes=1)
x = torch.randn(2, 3, 256, 256)
out = model(x)
print(f"input  {tuple(x.shape)}")
print(f"output {tuple(out.shape)}   <- one logit per PIXEL")
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
~~~

### Segmentation losses

~~~python seg_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice = 2|A and B| / (|A| + |B|). Loss = 1 - Dice.

    Optimises the OVERLAP directly, so it is far more robust to
    foreground/background imbalance than cross-entropy.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = ((2 * intersection + self.smooth) /
                (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth))
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Down-weights easy pixels so the model concentrates on hard ones."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)                    # probability of the TRUE class
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """Generalises Dice with separate weights for false positives and negatives.

    beta > alpha penalises FALSE NEGATIVES more - what you want in medical
    imaging, where missing a lesion is worse than a false alarm.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).reshape(logits.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return 1 - tversky.mean()


class ComboLoss(nn.Module):
    """BCE + Dice. The standard production choice: BCE gives smooth pixel-wise
    gradients, Dice optimises the metric you actually care about."""

    def __init__(self, bce_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.w = bce_weight

    def forward(self, logits, targets):
        return self.w * self.bce(logits, targets) + (1 - self.w) * self.dice(logits, targets)


# =====================================================================
# WHY DICE BEATS CROSS-ENTROPY UNDER IMBALANCE
# =====================================================================
torch.manual_seed(0)
B, H, W = 4, 128, 128

# only 2% of pixels are foreground - typical for lesion or defect segmentation
target = torch.zeros(B, 1, H, W)
target[:, :, 50:60, 50:70] = 1.0
print(f"foreground fraction: {target.mean():.4f}")

# a model that predicts ALL BACKGROUND
all_background = torch.full((B, 1, H, W), -6.0)     # a very negative logit
# a model that gets the region roughly right
decent = torch.full((B, 1, H, W), -6.0)
decent[:, :, 48:62, 48:72] = 4.0

losses = {"BCE": nn.BCEWithLogitsLoss(), "Dice": DiceLoss(),
          "Focal": FocalLoss(), "Tversky": TverskyLoss(), "Combo": ComboLoss()}

print(f"\\n{'loss':10s} {'all background':>16s} {'decent mask':>14s} {'ratio':>8s}")
print("-" * 52)
for name, fn in losses.items():
    lb = fn(all_background, target).item()
    ld = fn(decent, target).item()
    print(f"{name:10s} {lb:>16.5f} {ld:>14.5f} {lb/max(ld,1e-9):>8.1f}x")

print("""
READ THE BCE ROW: predicting all background scores 0.012, which looks
excellent. The gradient signal to actually find the foreground is tiny.

READ THE DICE ROW: predicting all background scores ~1.0 - the worst
possible value. Dice cannot be fooled by the background majority.
""")


# =====================================================================
# METRICS
# =====================================================================
def dice_coefficient(pred, target, threshold=0.5, eps=1e-7):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


def iou_score(pred, target, threshold=0.5, eps=1e-7):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)


def pixel_accuracy(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    return (pred == target).float().mean()


print(f"{'model':18s} {'Dice':>8s} {'IoU':>8s} {'pixel acc':>11s}")
print("-" * 48)
for name, logits in [("all background", all_background), ("decent mask", decent)]:
    print(f"{name:18s} {dice_coefficient(logits, target):>8.4f} "
          f"{iou_score(logits, target):>8.4f} {pixel_accuracy(logits, target):>11.4f}")
print("\\nPIXEL ACCURACY IS USELESS HERE - 0.98 for a model that found nothing.")
print("Always report Dice or IoU for segmentation.")
~~~

~~~text
foreground fraction: 0.0122

loss         all background    decent mask    ratio
----------------------------------------------------
BCE                 0.01221        0.00381      3.2x
Dice                0.99992        0.24138      4.1x
Focal               0.00021        0.00009      2.3x
Tversky             0.99994        0.19472      5.1x
Combo               0.50607        0.12259      4.1x

model                  Dice      IoU   pixel acc
------------------------------------------------
all background       0.0000   0.0000      0.9878
decent mask          0.7576   0.6098      0.9942
~~~

### Segment Anything: zero-shot segmentation

~~~bash
pip install segment-anything
# or the faster/newer:  pip install ultralytics   (includes SAM and MobileSAM)
~~~

~~~python sam.py
"""SAM segments objects from a point or box prompt, with NO training."""
from ultralytics import SAM
import cv2
import matplotlib.pyplot as plt

model = SAM("sam2_b.pt")           # or "mobile_sam.pt" for speed

# 1. prompt with a POINT
results = model("image.jpg", points=[[400, 300]], labels=[1])   # 1 = foreground

# 2. prompt with a BOX
results = model("image.jpg", bboxes=[[100, 100, 500, 400]])

# 3. segment EVERYTHING (no prompt)
results = model("image.jpg")

r = results[0]
print(f"masks produced: {len(r.masks)}")
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))
plt.axis("off"); plt.show()

# =====================================================================
# THE KILLER APPLICATION: semi-automatic annotation
# =====================================================================
"""
Manually drawing polygon masks takes 1-5 minutes per object.

With SAM:
  1. A human clicks once inside each object
  2. SAM produces a pixel-accurate mask
  3. The human accepts or corrects it
  -> 10-30x faster annotation

Tools that do this: Roboflow Annotate, CVAT with the SAM plugin,
Label Studio, and the Ultralytics auto-annotator:

    from ultralytics.data.annotator import auto_annotate
    auto_annotate(data="images/", det_model="yolo11x.pt", sam_model="sam2_b.pt")

This produces segmentation labels automatically from a detector's boxes.
"""
~~~

:::warn Augmenting images and masks together
The image and its mask must receive **exactly the same** geometric transform. Use a library
that handles paired augmentation:
~~~python
import albumentations as A
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.RandomBrightnessContrast(p=0.3),      # image only - masks are unaffected
])
out = transform(image=img, mask=mask)       # both transformed consistently
img, mask = out["image"], out["mask"]
~~~
And when resizing a mask, always use **nearest-neighbour** interpolation - bilinear would
invent class values that do not exist.
:::
`
}
],
quiz: [
{
q: 'Why does U-Net use skip connections between encoder and decoder?',
options: [
  'To reduce parameters',
  'The encoder discards spatial detail; the skips carry high-resolution features across so the output mask can be pixel-accurate',
  'To speed up training',
  'They are optional'
],
answer: 1,
why: 'Segmentation needs both semantic understanding (deep, low-resolution) and precise localisation (shallow, high-resolution). The skips supply the second.'
},
{
q: 'Your segmentation target is 2% foreground. Cross-entropy training predicts all background. What is the fix?',
options: [
  'More epochs',
  'Use Dice loss, or a combined BCE + Dice loss, which optimises overlap directly',
  'Lower the learning rate',
  'Use a bigger model'
],
answer: 1,
why: 'Cross-entropy rewards the 98% background majority. Dice measures overlap, so predicting nothing gives the worst possible score, forcing the model to find the foreground.'
},
{
q: 'Why is pixel accuracy a poor segmentation metric?',
options: [
  'It is slow to compute',
  'With a small foreground, predicting all background gives 98% accuracy while finding nothing',
  'It only works for binary masks',
  'It is not a real metric'
],
answer: 1,
why: 'Exactly the same problem as accuracy in imbalanced classification. Dice and IoU are the standard segmentation metrics because they ignore the true-negative majority.'
},
{
q: 'When resizing a segmentation mask, which interpolation should you use?',
options: ['Bilinear', 'Nearest neighbour', 'Bicubic', 'Lanczos'],
answer: 1,
why: 'Smooth interpolation produces intermediate values - a pixel halfway between class 2 and class 4 becomes class 3, which may be a completely unrelated class. Nearest neighbour preserves valid labels.'
}
]
},

/* ============================================================ */
{
id: 'vision-transformers',
title: 'Vision Transformers and CLIP',
summary: 'Attention applied to image patches, why ViT needs more data than a CNN, and CLIP - the model that made zero-shot image classification possible.',
tags: ['vision', 'transformers', 'clip', 'advanced'],
intro: `
## The Vision Transformer idea

Cut the image into fixed patches, treat each patch as a token, and run a standard
transformer over them. That is genuinely all.

~~~text
224x224 image
   |
   split into 16x16 patches  ->  14 x 14 = 196 patches
   |
   flatten each patch (16*16*3 = 768 numbers)
   |
   linear projection to the model dimension
   |
   + positional embeddings   <- transformers have no built-in notion of position
   |
   + a learnable [CLS] token <- its final state represents the whole image
   |
   [ standard transformer encoder blocks ] x12
   |
   the [CLS] output -> a linear classifier
~~~

## CNN versus ViT

| | CNN | Vision Transformer |
|---|---|---|
| Inductive bias | Locality and translation invariance built in | Almost none - must be learned |
| Data needed | Works from thousands | Needs millions, or pretraining |
| Receptive field | Grows with depth | **Global from layer one** |
| Scaling | Plateaus | Keeps improving with data and size |
| Small datasets | **Better** | Worse, unless pretrained |

:::tip The practical summary
On ImageNet-1k alone, ViT **loses** to a good CNN. Pretrained on 300 million images
(JFT-300M) it **wins decisively**. The lesson generalises: transformers trade inductive
bias for scalability. With enough data, learning the bias beats hand-coding it.
:::

## CLIP

Train an image encoder and a text encoder together on 400 million image-caption pairs, so
that matching pairs land close together in a shared embedding space.

~~~text
  image ---> [image encoder] ---> image vector  \\
                                                  cosine similarity
  text  ---> [text  encoder] ---> text vector   /

  Training: for each batch, the correct (image, text) pairs should have
  high similarity and all the mismatched pairs low similarity.
~~~

The result: **zero-shot classification**. Give CLIP a set of text descriptions and it picks
the best match, for classes it has never been explicitly trained on.
`,
keyPoints: [
  'ViT treats image patches as tokens and applies a standard transformer.',
  'ViT needs far more data than a CNN because it lacks the locality prior.',
  'CLIP aligns image and text in one embedding space, enabling zero-shot classification.',
  'CLIP embeddings are excellent general-purpose features for retrieval and search.'
],
pitfalls: [
  'Training a ViT from scratch on a small dataset - it will lose to a CNN.',
  'Using CLIP without prompt engineering - "a photo of a {class}" beats the bare class name.',
  'Forgetting to normalise embeddings before computing cosine similarity.',
  'Assuming CLIP zero-shot is competitive with fine-tuning when you actually have labels.'
],
levels: [
{
name: 'ViT and CLIP in practice',
goal: 'Build a ViT from scratch, then use CLIP for zero-shot classification and semantic image search.',
md: `
~~~python vit_scratch.py
"""A Vision Transformer, from scratch, in about 90 lines."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Split the image into patches and project each to the model dimension.

    Neat trick: a Conv2d with kernel_size == stride == patch_size does
    the splitting AND the linear projection in one operation.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                 # (B, embed_dim, H/p, W/p)
        x = x.flatten(2)                 # (B, embed_dim, n_patches)
        return x.transpose(1, 2)         # (B, n_patches, embed_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads=12, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)      # Q, K, V in one matrix
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out), attn


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> attention -> residual, LN -> MLP -> residual."""

    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out                       # residual
        x = x + self.mlp(self.norm2(x))        # residual
        return x, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, n_classes=1000,
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # the [CLS] token: a learnable vector whose final state summarises the image
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # learnable positional embeddings - a transformer has no innate sense of order
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_attention=False):
        B = x.shape[0]
        x = self.patch_embed(x)                              # (B, n_patches, dim)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                       # prepend [CLS]
        x = self.pos_drop(x + self.pos_embed)

        attentions = []
        for block in self.blocks:
            x, attn = block(x)
            attentions.append(attn)

        x = self.norm(x)
        logits = self.head(x[:, 0])                          # use the [CLS] output
        return (logits, attentions) if return_attention else logits


# ---- the standard configurations ------------------------------------
configs = {
    "ViT-Tiny":  dict(embed_dim=192, depth=12, n_heads=3),
    "ViT-Small": dict(embed_dim=384, depth=12, n_heads=6),
    "ViT-Base":  dict(embed_dim=768, depth=12, n_heads=12),
    "ViT-Large": dict(embed_dim=1024, depth=24, n_heads=16),
}

x = torch.randn(2, 3, 224, 224)
print(f"{'model':12s} {'parameters':>14s} {'output':>14s}")
print("-" * 42)
for name, cfg in configs.items():
    m = VisionTransformer(n_classes=1000, **cfg)
    n = sum(p.numel() for p in m.parameters())
    print(f"{name:12s} {n:>14,} {str(tuple(m(x).shape)):>14s}")

# ---- how many patches? ----------------------------------------------
print(f"\\n{'image':>8} {'patch':>7} {'tokens':>8} {'attention matrix':>18}")
for img, p in [(224, 16), (224, 32), (384, 16), (512, 16)]:
    n = (img // p) ** 2 + 1
    print(f"{img:>8} {p:>7} {n:>8} {f'{n}x{n} = {n*n:,}':>18}")
print("\\nAttention is O(n^2) in the number of tokens - which is why high")
print("resolution is expensive for a plain ViT, and why hierarchical designs")
print("like Swin Transformer exist.")
~~~

### CLIP: zero-shot classification and image search

~~~bash
pip install open_clip_torch
# or:  pip install transformers  (for the HuggingFace CLIP)
~~~

~~~python clip_demo.py
import torch
import open_clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k")
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# =====================================================================
# 1. ZERO-SHOT CLASSIFICATION - no training on your classes at all
# =====================================================================
CLASSES = ["a dog", "a cat", "a car", "a bicycle", "a person",
           "a building", "a tree", "food", "a bird", "a boat"]

# PROMPT ENGINEERING MATTERS. These templates were found empirically
# and add several accuracy points over the bare class name.
TEMPLATES = [
    "a photo of {}.",
    "a blurry photo of {}.",
    "a close-up photo of {}.",
    "a bright photo of {}.",
    "a photo of the large {}.",
    "a photo of the small {}.",
]


@torch.no_grad()
def build_class_embeddings(classes, templates):
    """Average the embeddings of several prompt templates per class -
    'prompt ensembling', worth roughly 2-3 accuracy points."""
    weights = []
    for cls in classes:
        texts = [t.format(cls) for t in templates]
        tokens = tokenizer(texts).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)     # NORMALISE
        emb = emb.mean(dim=0)
        emb = emb / emb.norm()
        weights.append(emb)
    return torch.stack(weights, dim=1)                  # (dim, n_classes)


class_embeddings = build_class_embeddings(CLASSES, TEMPLATES)
print(f"class embedding matrix: {tuple(class_embeddings.shape)}")


@torch.no_grad()
def classify(image_path, top_k=3):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    features = model.encode_image(image)
    features = features / features.norm(dim=-1, keepdim=True)

    # 100.0 is CLIP's learned temperature scaling
    logits = 100.0 * features @ class_embeddings
    probs = logits.softmax(dim=-1).cpu().numpy()[0]

    order = probs.argsort()[::-1][:top_k]
    return [(CLASSES[i], float(probs[i])) for i in order]


# for name, prob in classify("photo.jpg"):
#     print(f"  {name:20s} {prob:.4f}")

# =====================================================================
# 2. SEMANTIC IMAGE SEARCH - the most useful CLIP application
# =====================================================================
from pathlib import Path


@torch.no_grad()
def index_images(folder, batch_size=32):
    """Embed every image once. This is your searchable index."""
    paths = sorted([p for p in Path(folder).glob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    embeddings = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([preprocess(Image.open(p).convert("RGB"))
                             for p in paths[i:i+batch_size]]).to(device)
        emb = model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu())
    return paths, torch.cat(embeddings)


@torch.no_grad()
def search(query, paths, index, top_k=5):
    """Search a folder of images with a natural-language query."""
    tokens = tokenizer([query]).to(device)
    q = model.encode_text(tokens)
    q = (q / q.norm(dim=-1, keepdim=True)).cpu()

    similarity = (index @ q.T).squeeze(1)
    top = similarity.argsort(descending=True)[:top_k]
    return [(paths[i], float(similarity[i])) for i in top]


# paths, index = index_images("my_photos/")
# for path, score in search("a dog running on a beach at sunset", paths, index):
#     print(f"  {score:.4f}  {path.name}")

# =====================================================================
# 3. IMAGE-TO-IMAGE SIMILARITY (duplicate and near-duplicate detection)
# =====================================================================
@torch.no_grad()
def find_duplicates(paths, index, threshold=0.95):
    sim = index @ index.T
    n = len(paths)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > threshold:
                pairs.append((paths[i].name, paths[j].name, float(sim[i, j])))
    return sorted(pairs, key=lambda t: -t[2])


# =====================================================================
# 4. WHY PROMPT ENGINEERING MATTERS
# =====================================================================
print("""
CLIP PROMPT ENGINEERING (measured on ImageNet zero-shot)

  "dog"                              ~ 58% accuracy
  "a photo of a dog"                 ~ 62%
  "a photo of a dog, a type of pet"  ~ 64%
  ensemble of 80 templates           ~ 68%

CLIP was trained on internet ALT TEXT, which is almost always a full
sentence. A bare class name is out of distribution for it.

RULES
  - Always use a sentence template
  - Add a category hint: "a type of bird", "a kind of food"
  - Ensemble several templates and average the embeddings
  - Make the classes MUTUALLY EXCLUSIVE and specific
  - Include a "background" or "none of these" class if that is possible
""")
~~~

### CLIP embeddings as features

~~~python clip_features.py
"""CLIP image embeddings are strong general-purpose features -
often better than ImageNet-supervised features for transfer."""
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

@torch.no_grad()
def extract_features(images, batch_size=64):
    feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(im) for im in images[i:i+batch_size]]).to(device)
        f = model.encode_image(batch)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats)


# The "linear probe" evaluation: freeze CLIP, train only logistic regression.
# With just 16 examples per class this often beats a fine-tuned ResNet.
#
# X = extract_features(train_images)
# clf = LogisticRegression(max_iter=2000, C=0.316)
# print(cross_val_score(clf, X, train_labels, cv=5).mean())

print("""
WHEN TO USE WHAT

  ZERO-SHOT CLIP        no labels at all; you can describe the classes in words
  LINEAR PROBE on CLIP  10-1000 labels per class. Extremely strong, very cheap.
  FINE-TUNE a CNN/ViT   1000+ labels per class, and you need maximum accuracy
  CLIP for SEARCH       any time you need semantic image retrieval
""")
~~~

:::tip CLIP is more useful as an embedding model than as a classifier
Zero-shot classification is the headline demonstration, but in practice the highest-value
uses are **semantic search**, **deduplication**, **content moderation triage**, **dataset
curation** and **linear probing with very few labels**. Those all exploit the embedding
space directly.
:::
`
}
],
quiz: [
{
q: 'Why does a Vision Transformer need far more training data than a CNN?',
options: [
  'It has more parameters',
  'It lacks the built-in locality and translation-invariance priors, so it must learn them from data',
  'It trains more slowly',
  'It uses larger images'
],
answer: 1,
why: 'Convolution hard-codes assumptions that are true of images. ViT must discover them, which needs enormous data - or pretraining, which is why fine-tuning a pretrained ViT works so well.'
},
{
q: 'What does the [CLS] token do in a ViT?',
options: [
  'Marks the end of the sequence',
  'It is a learnable token whose final hidden state serves as the whole-image representation for classification',
  'It stores the class label',
  'It is a positional encoding'
],
answer: 1,
why: 'It attends to every patch through the layers, aggregating them into a single vector fed to the classification head. Some architectures instead average all patch outputs.'
},
{
q: 'CLIP enables zero-shot classification because:',
options: [
  'It was trained on every possible class',
  'It maps images and text into one shared embedding space, so any class describable in words can be compared to an image',
  'It uses a very large model',
  'It memorises the internet'
],
answer: 1,
why: 'Classification becomes a nearest-neighbour lookup in the joint space against text embeddings of your class descriptions - no training on those classes required.'
},
{
q: 'Why does "a photo of a dog" work better than "dog" as a CLIP prompt?',
options: [
  'It is longer',
  'CLIP was trained on internet alt-text, which is full sentences - a bare noun is out of distribution',
  'It contains more words to match',
  'It does not work better'
],
answer: 1,
why: 'Matching the training distribution matters. Template prompts add several accuracy points, and ensembling many templates adds several more.'
}
]
}

]
});
