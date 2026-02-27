from dataclasses import dataclass
import math
import numpy as np
from draw import draw_belt, draw_gear
import pygame
from pygame.math import Vector2


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

FPS = 60


@dataclass
class DragState:
    gear: "Gear"
    start_angle: float  # angle of click point relative to gear.theta at mousedown


@dataclass
class Gear:
    pos: Vector2
    r: float
    n: int
    theta: float = 0.0
    vel: float = 0.0
    inverse_inertia: float = 1.0

    def hitbox(self) -> pygame.Rect:
        return pygame.Rect(
            self.pos.x - self.r, self.pos.y - self.r, self.r * 2, self.r * 2
        )

    def contains(self, point: Vector2) -> bool:
        """True if point lies within the circular gear hitbox."""
        return self.pos.distance_to(point) <= self.r


@dataclass
class Constraint:
    i: int  # first gear index
    j: int  # second gear index
    jacobian: tuple[float, float]  # (J_i, J_j) coefficients


def solve_constraint(c: Constraint, gears: list[Gear]) -> tuple[float, float]:
    """Solve a single constraint and return impulses for gears c.i and c.j."""
    b = 0
    J = np.array(c.jacobian)
    v = np.array([gears[c.i].vel, gears[c.j].vel])
    M_inv = np.array([[gears[c.i].inverse_inertia, 0], [0, gears[c.j].inverse_inertia]])
    _lambda = (b - J @ v) / (J @ M_inv @ J.T)
    correction = M_inv @ J * _lambda
    return (float(correction[0]), float(correction[1]))


REF_R = 40  # reference radius for inertia scaling


def make_gear(pos: Vector2, r: float, n: int, theta: float = 0.0) -> Gear:
    """Create a gear with inertia derived from its radius (uniform density)."""
    return Gear(pos=pos, r=r, n=n, theta=theta, inverse_inertia=(REF_R / r) ** 2)


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    gears: list[Gear] = [
        make_gear(Vector2(200, 300), r=40, n=8, theta=math.pi / 8),
        make_gear(Vector2(320, 300), r=80, n=16),
        make_gear(Vector2(650, 300), r=50, n=10),
    ]

    constraints: list[Constraint] = [
        # Mesh: gear 0 and gear 1 (opposite rotations)
        # ω₀·r₀ + ω₁·r₁ = 0
        Constraint(i=0, j=1, jacobian=(gears[0].r, gears[1].r)),
        # Belt: gear 1 and gear 2 (same rotation direction)
        # ω₁·r₁ − ω₂·r₂ = 0
        Constraint(i=1, j=2, jacobian=(gears[1].r, -gears[2].r)),
    ]

    drag: DragState | None = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse = Vector2(event.pos)
                for gear in reversed(gears):
                    if gear.contains(mouse):
                        click_angle = math.atan2(
                            mouse.y - gear.pos.y, mouse.x - gear.pos.x
                        )
                        drag = DragState(
                            gear=gear, start_angle=click_angle - gear.theta
                        )
                        break

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drag = None

        screen.fill((0, 0, 0))

        if drag is not None:
            mouse = Vector2(pygame.mouse.get_pos())
            current_angle = math.atan2(
                mouse.y - drag.gear.pos.y, mouse.x - drag.gear.pos.x
            )
            target_theta = current_angle - drag.start_angle
            delta = (target_theta - drag.gear.theta + math.pi) % (2 * math.pi) - math.pi
            drag.gear.vel = 4 * delta / FPS

        impulses = [0.0] * len(gears)
        for c in constraints:
            imp_i, imp_j = solve_constraint(c, gears)
            impulses[c.i] += imp_i
            impulses[c.j] += imp_j
        for i, gear in enumerate(gears):
            gear.vel += impulses[i]

        # Draw belt between gear 1 and gear 2
        draw_belt(
            screen,
            (180, 180, 180),
            gears[1].pos,
            gears[1].r,
            gears[2].pos,
            gears[2].r,
            width=2,
        )

        for i, gear in enumerate(gears):
            gear.theta = (gear.theta + gear.vel) % (2 * math.pi)
            color = COLORS[i % len(COLORS)]
            draw_gear(
                screen, color, gear.pos, gear.r, gear.n, gear.theta, tooth_depth=8
            )

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
