import math
import pygame
from pygame.math import Vector2


def draw_belt(
    surface: pygame.Surface,
    color,
    pos1: Vector2,
    r1: float,
    pos2: Vector2,
    r2: float,
    width: int = 2,
) -> None:
    """Draw a belt (uncrossed) connecting two pulleys.

    Draws the two external-tangent lines and the arcs that wrap around the
    far side of each pulley.
    """
    dx = pos2.x - pos1.x
    dy = pos2.y - pos1.y
    d = math.hypot(dx, dy)
    if d < abs(r1 - r2) + 1:
        return  # circles overlap, skip

    alpha = math.atan2(dy, dx)

    # Normal direction (φ) for external tangent:
    # cos(φ − α) = (r1 − r2) / d
    ratio = (r1 - r2) / d
    ratio = max(-1.0, min(1.0, ratio))  # clamp for safety
    offset = math.acos(ratio)

    phi_top = alpha + offset
    phi_bot = alpha - offset

    # Tangent points on each circle  (radius × normal direction)
    p1_top = (pos1.x + r1 * math.cos(phi_top), pos1.y + r1 * math.sin(phi_top))
    p2_top = (pos2.x + r2 * math.cos(phi_top), pos2.y + r2 * math.sin(phi_top))
    p1_bot = (pos1.x + r1 * math.cos(phi_bot), pos1.y + r1 * math.sin(phi_bot))
    p2_bot = (pos2.x + r2 * math.cos(phi_bot), pos2.y + r2 * math.sin(phi_bot))

    # Straight belt segments
    pygame.draw.line(surface, color, p1_top, p2_top, width)
    pygame.draw.line(surface, color, p1_bot, p2_bot, width)

    # Arcs wrapping around the *far* side of each pulley (away from the other).
    # For circle 1 the arc goes from phi_top → phi_bot going the *long* way
    # around (through angle alpha + π, i.e. the back side).
    # For circle 2 the arc goes from phi_bot → phi_top the long way around
    # (through angle alpha, the front side — which is the far side for circle 2).
    _draw_arc_between(
        surface,
        color,
        pos1,
        r1,
        phi_bot,
        phi_top,
        away_angle=alpha + math.pi,
        width=width,
    )
    _draw_arc_between(
        surface, color, pos2, r2, phi_top, phi_bot, away_angle=alpha, width=width
    )


def _draw_arc_between(
    surface: pygame.Surface,
    color,
    center: Vector2,
    r: float,
    angle_a: float,
    angle_b: float,
    away_angle: float,
    width: int,
    segments: int = 40,
) -> None:
    """Draw the arc from angle_a to angle_b that passes through away_angle."""
    # Normalise angles to [0, 2π)
    a = angle_a % (2 * math.pi)
    b = angle_b % (2 * math.pi)
    away = away_angle % (2 * math.pi)

    # Choose sweep direction (CW or CCW) so the arc passes through away_angle.
    # CCW sweep from a:
    ccw_span = (b - a) % (2 * math.pi)
    # Does away lie inside that CCW arc?
    away_in_ccw = ((away - a) % (2 * math.pi)) < ccw_span

    if away_in_ccw:
        span = ccw_span
    else:
        span = ccw_span - 2 * math.pi  # negative = clockwise

    points = []
    for k in range(segments + 1):
        t = a + span * k / segments
        points.append((center.x + r * math.cos(t), center.y + r * math.sin(t)))

    if len(points) >= 2:
        pygame.draw.lines(surface, color, False, points, width)


def draw_gear(
    surface: pygame.Surface,
    color,
    pos: Vector2,
    r: float,
    n: int,
    theta: float = 0.0,
    tooth_depth: float | None = None,
    tooth_width: float = 0.4,
    gap_width: float = 0.3,
    line_width: int = 0,
) -> None:
    """Draw a gear as a filled (or outlined) polygon.

    Parameters
    ----------
    surface     : pygame.Surface to draw on
    color       : fill/line color (RGB or RGBA)
    x, y        : centre of the gear in pixels
    r           : pitch radius in pixels
    n           : number of teeth
    theta       : rotation offset in radians (default 0)
    tooth_depth : radial height of each tooth in pixels
                  (defaults to r * 0.1)
    tooth_width : fraction of one tooth period occupied by the flat tooth tip
                  (0 – 1, default 0.4)
    gap_width   : fraction of one tooth period occupied by the flat gap floor
                  between teeth (0 – 1, default 0.3); tooth_width + gap_width
                  should be <= 1 to leave room for the angled flanks
    line_width  : 0 = filled polygon, >0 = outline only
    """
    if tooth_depth is None:
        tooth_depth = r * 0.1

    outer_r = r + tooth_depth
    inner_r = max(r - tooth_depth, 0.0)

    angle_step = 2 * math.pi / n
    half_tip = angle_step * tooth_width / 2
    half_gap = angle_step * gap_width / 2

    points: list[tuple[float, float]] = []
    for i in range(n):
        mid = theta + i * angle_step

        # Trailing edge of the gap floor (inner radius) — tooth flank rises here
        points.append(
            (
                pos.x + inner_r * math.cos(mid - angle_step / 2 + half_gap),
                pos.y + inner_r * math.sin(mid - angle_step / 2 + half_gap),
            )
        )

        # Leading edge of tooth tip (outer radius)
        points.append(
            (
                pos.x + outer_r * math.cos(mid - half_tip),
                pos.y + outer_r * math.sin(mid - half_tip),
            )
        )

        # Trailing edge of tooth tip (outer radius)
        points.append(
            (
                pos.x + outer_r * math.cos(mid + half_tip),
                pos.y + outer_r * math.sin(mid + half_tip),
            )
        )

        # Leading edge of the next gap floor (inner radius) — flank descends here
        points.append(
            (
                pos.x + inner_r * math.cos(mid + angle_step / 2 - half_gap),
                pos.y + inner_r * math.sin(mid + angle_step / 2 - half_gap),
            )
        )

    pygame.draw.polygon(surface, color, points, line_width)
