import numpy as np
from shapely import Point
from shapely import LineString, Point
import matplotlib.pyplot as plt


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
    fig, ax = plt.subplots()
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    for element in elements:
        plt.fill(*element.exterior.xy, color="white")

    plt.fill(*elements[-1].exterior.xy, color=LEGEND["outlet"])
    plt.fill(*elements[-2].exterior.xy, color=LEGEND["inlet"])

    plt.axis("equal")
    ax.axis("off")
    plt.show()


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


def draw_tesla_valve(width, dx, dy, radius):
    dx = 1
    dy /= dx
    width /= dx
    radius /= dx

    # All elements to be drawn in the end.
    elements = []

    elements += make_line(dx, 0, width)

    for i in range(10):
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

        if i < 9:
            current["x"] -= (current["x"] - old_x) / 2
            current["y"] -= dy / dx * (current["x"] - old_x)

    elements += make_line(dx, 0, width)

    elements += [Point(0, 0).buffer(width * 1.1)]
    elements += [Point(current["x"], current["y"]).buffer(width * 1.1)]

    plot(elements)


if __name__ == "__main__":
    draw_tesla_valve(0.07, 2, 0.35, 0.1)
