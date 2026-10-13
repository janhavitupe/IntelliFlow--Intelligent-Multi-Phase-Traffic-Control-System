"""
phase_geometry.py

Paths of the 16 movements through the junction, and a conflict check.

Vehicles drive on the LEFT. Each arm is described in a frame where the arm
points up and its incoming traffic drives down (+y) into the junction; the
four incoming lanes sit right of the centre line at these lane-width
offsets: U-turn 0.5, right 1.5, straight 2.5, left 3.5. The junction spans
-4..4 lane widths. Arms are rotated clockwise: North 0, East 90, South 180,
West 270 degrees. The simulation viewer draws exactly these paths.

Two movements CONFLICT if their paths cross, or if they end in the same exit
lane. Movements from the same approach never conflict (they queue apart).
"""
import math
from itertools import combinations

H = 4.0
LANE_X = {"UTURN": 0.5, "RIGHT": 1.5, "STRAIGHT": 2.5, "LEFT": 3.5}
ROTATION = {"North": 0, "East": 1, "South": 2, "West": 3}       # clockwise quarter turns


def _curve(movement):
    """Bezier control points (arm frame, lane widths) and exit direction."""
    x = LANE_X[movement]
    if movement == "STRAIGHT":
        return [[x, -H], [x, -H / 3], [x, H / 3], [x, H]], (0, 1)
    if movement == "LEFT":
        return [[x, -H], [x, -3.75], [3.75, -x], [H, -x]], (1, 0)
    if movement == "RIGHT":
        return [[x, -H], [x, -0.5], [-0.5, x], [-H, x]], (-1, 0)
    return [[0.5, -H], [0.5, -H + 2.6], [-0.5, -H + 2.6], [-0.5, -H]], (0, -1)   # U-turn


def movement_path(approach, movement, samples=40):
    """Polyline of a movement in junction coordinates (y down), incl. 3 lane widths of exit."""
    p, exit_dir = _curve(movement)
    pts = []
    for i in range(samples + 1):
        t = i / samples
        u = 1 - t
        pts.append((u**3 * p[0][0] + 3 * u * u * t * p[1][0] + 3 * u * t * t * p[2][0] + t**3 * p[3][0],
                    u**3 * p[0][1] + 3 * u * u * t * p[1][1] + 3 * u * t * t * p[2][1] + t**3 * p[3][1]))
    end = pts[-1]
    pts += [(end[0] + exit_dir[0] * 0.3 * d, end[1] + exit_dir[1] * 0.3 * d) for d in range(1, 11)]
    k = ROTATION[approach] * math.pi / 2
    return [(x * math.cos(k) - y * math.sin(k), x * math.sin(k) + y * math.cos(k)) for x, y in pts]


def _segments_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)


def conflict(m1, m2):
    """'cross', 'merge' or None for two (approach, movement) pairs."""
    if m1[0] == m2[0]:
        return None
    p1, p2 = movement_path(*m1), movement_path(*m2)
    if math.dist(p1[-1], p2[-1]) < 0.3:
        return "merge"
    for i in range(len(p1) - 1):
        for j in range(len(p2) - 1):
            if _segments_cross(p1[i], p1[i + 1], p2[j], p2[j + 1]):
                return "cross"
    return None


def phase_conflicts(phase):
    """List of (movement_id, movement_id, kind) conflicts inside one Phase."""
    moves = [(m.approach, m.movement_type.name) for m in phase.movements]
    out = []
    for a, b in combinations(moves, 2):
        kind = conflict(a, b)
        if kind:
            out.append((f"{a[0]}_{a[1]}", f"{b[0]}_{b[1]}", kind))
    return out
