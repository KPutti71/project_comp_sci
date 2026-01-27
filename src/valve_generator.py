'''
Sript creates a tesla valve based of dx, dy, raidus and the with of the valve.
The png of the valve is sxported as: "data/valve.txt".
'''

import numpy as np
from shapely import Point
from shapely import LineString, Point
import matplotlib.pyplot as plt
from PIL import Image


# Color legend used in the PNG
LEGEND = {
    "fluid":  (255, 255, 255),   # white
    "wall":   (0,   0,   0),     # black
    "inlet":  (255 / 255, 0,   0),     # red
    "outlet": (128 / 255, 0, 128 / 255),     # purple
}


# Current position of drawing point.
current = {
    "x": 0.0,
    "y": 0.0
}


def plot(elements):
    size_px = 500
    dpi = 1
    figsize = size_px / dpi

    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    for element in elements:
        ax.fill(*element.exterior.xy, color="white")

    ax.set_aspect("equal")
    ax.axis("off")

    path = "data/valve.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.show()

    img = Image.open(path)
    w, h = img.size
    cropped = img.crop((w * 0.1, 0, w * 0.935, h))
    cropped.save(path)

    return path


# Creates a line.
def make_line(dx, dy, width):
    result = LineString([(current["x"], current["y"]),
    (current["x"] + dx, current["y"] + dy)]).buffer(width)

    current["x"] += dx
    current["y"] += dy

    return [result]


# Creates a bend of 180 degrees.
def make_bend(dx, dy, radius, width):
    assert(dx > 0)

    # Smoothness of the bend.
    sn = 64

    a = 1/2 * np.pi - np.arctan(dy / dx)
    curve = np.linspace(np.pi - a, -a) if dy < 0 else np.linspace(-a, np.pi - a)
    move_x = np.cos(curve[0]) * radius
    move_y = np.sin(curve[0]) * radius

    result = [Point(current["x"] + np.cos(a) * radius + move_x, current["y"] + np.sin(a) * radius + move_y).buffer(width) for a in curve]

    current["x"] -= 2 * np.cos(curve[-1]) * radius
    current["y"] -= 2 * np.sin(curve[-1]) * radius

    return result


def generate_valve(width, dx, dy, radius, N):
    dx = 1
    dy /= dx
    width /= dx
    radius /= dx

    # All elements to be drawn in the end.
    elements = []

    elements += make_line(dx, 0, width)

    for i in range(N):
        old_y = current["y"]
        old_x = current["x"]

        elements += make_line(dx, dy, width)
        elements += make_bend(dx, dy, radius, width)
        elements += make_line(-((current["y"] - old_y * dx) / dy), -(current["y"] - old_y), width)

        current["x"] -= (current["x"] - old_x) / 2
        current["y"] += dy / dx * (current["x"] - old_x)

        old_y = current["y"]
        old_x = current["x"]

        elements += make_line(dx, -dy, width)
        elements += make_bend(dx, -dy, radius, width)
        elements += make_line(((current["y"] - old_y * dx) / dy), -(current["y"] - old_y), width)

        if i < N - 1:
            current["x"] -= (current["x"] - old_x) / 2
            current["y"] -= dy / dx * (current["x"] - old_x)
        else:
            elements += make_line(current["y"] / dy, -current["y"], width)

    elements += make_line(dx, 0, width)

    plot(elements)


if __name__ == "__main__":
    generate_valve(0.04, 1, 0.9, 0.12, 1)
