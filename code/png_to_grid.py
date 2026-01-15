import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Color legend used in the PNG
LEGEND = {
    "fluid":  (255, 255, 255),   # white
    "wall":   (0,   0,   0),     # black
    "inlet":  (255, 0,   0),     # red
    "outlet": (128, 0, 128),     # purple
}

# Integer codes used internally by the solver
CODES = {
    "fluid":  0,
    "wall":   1,
    "inlet":  2,
    "outlet": 3,
}

# Tolerances for robust color detection
DEFAULT_TOL = 10
DEFAULT_WALL_THRESH = 245

# Tolerances for robust color detection
L_INLET = 0
L_OUTLET = 1
L_WALL = 2

# PNG -> mask (robust against compression / anti-aliasing)
def _png_to_mask_hybrid(path, tol=DEFAULT_TOL, wall_thresh=DEFAULT_WALL_THRESH):
    # Load PNG as RGB image
    im = Image.open(path).convert("RGB")
    arr = np.array(im, dtype=np.uint8)
    h, w = arr.shape[:2]

    # Default everything to fluid
    mask = np.full((h, w), CODES["fluid"], dtype=np.int16)
    matched = np.zeros((h, w), dtype=bool)

    # match legend colors within tolerance
    for name, rgb in LEGEND.items():
        rgb = np.array(rgb, dtype=np.int16)
        diff = np.abs(arr.astype(np.int16) - rgb[None, None, :])
        close = np.all(diff <= tol, axis=2)
        mask[close] = CODES[name]
        matched |= close

    # dark pixels -> walls, bright pixels -> fluid
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
    fallback = ~matched
    mask[fallback & (gray < wall_thresh)] = CODES["wall"]
    mask[fallback & (gray >= wall_thresh)] = CODES["fluid"]

    return mask


# Merge wall pixels into axis-aligned rectangles
def _wall_rectangles_from_mask(mask, wall_code):
    H, W = mask.shape
    rects = []
    active = {}

    for y in range(H):
        row = (mask[y] == wall_code).astype(np.uint8)

        # find continious wall segments in row
        segments = []
        x = 0
        while x < W:
            if row[x] == 1:
                x0 = x
                while x < W and row[x] == 1:
                    x += 1
                segments.append((x0, x))
            else:
                x += 1

        new_active = {}
        used_prev = set()

        # extend rectangle vertically
        for (x0, x1) in segments:
            key = (x0, x1)
            if key in active:
                rx0, rx1, ry0, _ = active[key]
                new_active[key] = (rx0, rx1, ry0, y + 1)
                used_prev.add(key)
            else:
                new_active[key] = (x0, x1, y, y + 1)

        # close rectangles that did not continue
        for key, r in active.items():
            if key not in used_prev:
                rects.append(r)

        active = new_active

    # close any remaining rectangles
    rects.extend(active.values())
    return rects


# PNG -> grid dictionairy
def png_to_grid(
    png_path,
    *,
    dx=1.0,
    tol=DEFAULT_TOL,
    wall_thresh=DEFAULT_WALL_THRESH,
    flip_y=True,
):
    """
    Use in simulate.py as:
      grid = png_to_grid("test.png")

    Returns dict with:
      mask: (H,W) int16 (y-up if flip_y)
      rects: wall rectangles [(x0,x1,y0,y1)] in pixel coords (y-up if flip_y)
      W,H,dx
      box: (xmin,xmax,ymin,ymax)
      labels: (left,right,bottom,top) outer boundary label IDs
      codes: CODES
    """
    # build mask in image coordinates
    mask_img = _png_to_mask_hybrid(png_path, tol=tol, wall_thresh=wall_thresh)
    mask = np.flipud(mask_img) if flip_y else mask_img
    H, W = mask.shape

    # detect inlet/outlet markers on domain edges
    left_has_inlet = np.any(mask[:, 0] == CODES["inlet"])
    right_has_outlet = np.any(mask[:, -1] == CODES["outlet"])

    left_label = L_INLET if left_has_inlet else L_INLET
    right_label = L_OUTLET if right_has_outlet else L_OUTLET
    bottom_label = L_WALL
    top_label = L_WALL

    # extracts wall rectangles
    rects = _wall_rectangles_from_mask(mask, CODES["wall"])

    xmin, xmax = 0.0, W * dx
    ymin, ymax = 0.0, H * dx

    return {
        "mask": mask.astype(np.int16),
        "rects": rects,
        "W": W,
        "H": H,
        "dx": float(dx),
        "box": (xmin, xmax, ymin, ymax),
        "labels": (left_label, right_label, bottom_label, top_label),
        "codes": CODES,
        "legend": LEGEND,
        "source_png": png_path,
        "flip_y": bool(flip_y),
    }


# Debug for visual if actually working, mask + wall rectangles plot
def plot_grid(grid, show=True):
    mask = grid["mask"]
    rects = grid["rects"]

    H, W = mask.shape

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        mask,
        origin="lower",
        cmap="gray",
        interpolation="nearest"
    )

    # draw each wall rectangle
    for (x0, x1, y0, y1) in rects:
        w = x1 - x0
        h = y1 - y0
        rect = patches.Rectangle(
            (x0, y0),
            w,
            h,
            linewidth=1.0,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title("PNG grid + wall rectangles")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(0, W + 1, max(1, W // 10)))
    ax.set_yticks(np.arange(0, H + 1, max(1, H // 10)))
    ax.grid(True, alpha=0.3)

    plt.colorbar(im, ax=ax, label="mask code")

    if show:
        plt.show()

    return fig, ax
