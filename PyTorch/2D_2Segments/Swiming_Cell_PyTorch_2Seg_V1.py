"""
Cellule nageuse 2D — PyTorch PPO, réseau unique et IHM Tkinter.

Objectif pédagogique :
    - une cellule avec un flagelle articulé à deux segments ;
    - un seul réseau Actor-Critic partagé par 12 environnements ;
    - apprentissage PPO par lots avec GAE ;
    - évaluation séparée, sans modification des poids ;
    - visualisation de la cellule, des forces, des statistiques et des courbes.

Dépendance :
    py -m pip install torch

Lancement :
    py cellule_flagelle_ppo_pytorch.py

Test du moteur physique sans PyTorch :
    py cellule_flagelle_ppo_pytorch.py --test-physique

Entraînement sans interface :
    py cellule_flagelle_ppo_pytorch.py --headless 20

Les sauvegardes complètes sont placées dans :
    sauvegardes_pytorch/AAAAMMJJ_HHMMSS/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ""
except ModuleNotFoundError as exc:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Categorical = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORLD_WIDTH = 820
WORLD_HEIGHT = 680
CANVAS_WIDTH = 1720
CANVAS_HEIGHT = 720
WORLD_MARGIN = 18.0

CELL_RADIUS = 14.0
BODY_HALF_LENGTH = 19.0
BODY_HALF_WIDTH = 12.0
SEGMENT_LENGTH = 36.0

TARGET_RADIUS = 10.0
CAPTURE_RADIUS = 2.0 * TARGET_RADIUS
MIN_TARGET_DISTANCE = 190.0
TARGET_SEQUENCE_LENGTH = 512

MAX_BASE_ANGLE = math.radians(40.0)
MAX_INTERSEGMENT_ANGLE = math.radians(55.0)
MAX_JOINT_SPEED = 0.22
MOTOR_ACCELERATION = 0.055
JOINT_DAMPING = 0.88

DRAG_PARALLEL = 0.022
DRAG_PERPENDICULAR = 0.085
FORCE_TO_SPEED = 2.40
TORQUE_TO_ROTATION = 0.0040
SHAPE_PROPULSION = 3.20
MAX_CELL_SPEED = 2.40
MAX_CELL_ROTATION = 0.085

TIMEOUT_BASE = 400
TIMEOUT_PER_PIXEL = 4.0

PROGRESS_REWARD = 2.0
CAPTURE_REWARD = 3.0
TIME_PENALTY = 0.0010
ENERGY_PENALTY = 0.0004
WALL_PENALTY = 0.040

# Pénalité contextuelle : inactive pendant virages, demi-tours et
# repositionnements. Elle mesure la sinuosité sur une fenêtre glissante.
ZIGZAG_WINDOW_STEPS = 36
ZIGZAG_MIN_PROGRESS = 1.20
ZIGZAG_ALIGNMENT_LIMIT = math.radians(32.0)
ZIGZAG_ALIGNMENT_RATIO = 0.70
ZIGZAG_REPOSITION_GAIN = math.radians(8.0)
ZIGZAG_TARGET_EFFICIENCY = 0.82
ZIGZAG_PENALTY_COEFFICIENT = 0.0030

# Une phase cyclique fournit au réseau une horloge de battement.
GAIT_PERIOD = 32.0
OBSERVATION_SIZE = 11
HIDDEN_SIZE = 64
ACTION_COUNT = 9

# PPO : 12 mondes alimentent un seul réseau.
ENVIRONMENT_COUNT = 12
ROLLOUT_STEPS = 96
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.20
VALUE_CLIP = 0.20
LEARNING_RATE = 3.0e-4
ENTROPY_START = 0.012
ENTROPY_END = 0.002
ENTROPY_DECAY_UPDATES = 4000
VALUE_COEFFICIENT = 0.50
MAX_GRADIENT_NORM = 0.50
TARGET_KL = 0.030

SESSION_FILENAME = "checkpoint.pt"
CONFIG_FILENAME = "configuration.json"
HISTORY_FILENAME = "historique_entrainement.csv"
SESSION_FORMAT = "cellule-ppo-pytorch-v1"

ACTIONS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


# ---------------------------------------------------------------------------
# Outils mathématiques
# ---------------------------------------------------------------------------


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        magnitude = self.length()
        return self * (1.0 / magnitude) if magnitude > 1e-12 else Vec2()

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        return self.x * other.y - self.y * other.x


def direction(angle: float) -> Vec2:
    return Vec2(math.cos(angle), math.sin(angle))


def perpendicular(vector: Vec2) -> Vec2:
    return Vec2(-vector.y, vector.x)


def rotate_into_cell_frame(vector: Vec2, cell_angle: float) -> Vec2:
    cosine = math.cos(cell_angle)
    sine = math.sin(cell_angle)
    return Vec2(
        cosine * vector.x + sine * vector.y,
        -sine * vector.x + cosine * vector.y,
    )


def moving_average(values: list[float], window: int = 20) -> list[float]:
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def generate_target_sequence(seed: int, count: int) -> list[Vec2]:
    rng = random.Random(seed)
    targets: list[Vec2] = []
    previous = Vec2(WORLD_WIDTH * 0.5, WORLD_HEIGHT * 0.5)
    safe_margin = WORLD_MARGIN + CAPTURE_RADIUS + 10.0
    for _ in range(count):
        candidate = previous
        for _attempt in range(600):
            candidate = Vec2(
                rng.uniform(safe_margin, WORLD_WIDTH - safe_margin),
                rng.uniform(safe_margin, WORLD_HEIGHT - safe_margin),
            )
            if (candidate - previous).length() >= MIN_TARGET_DISTANCE:
                break
        targets.append(candidate)
        previous = candidate
    return targets


# ---------------------------------------------------------------------------
# Environnement physique
# ---------------------------------------------------------------------------


@dataclass
class CellState:
    position: Vec2 = field(
        default_factory=lambda: Vec2(WORLD_WIDTH * 0.5, WORLD_HEIGHT * 0.5)
    )
    angle: float = 0.0
    velocity: Vec2 = field(default_factory=Vec2)
    angular_velocity: float = 0.0
    joint_angle_1: float = 0.0
    joint_angle_2: float = 0.0
    joint_speed_1: float = 0.0
    joint_speed_2: float = 0.0
    gait_phase: float = 0.0


@dataclass
class EpisodeMetrics:
    reward: float
    captured: bool
    steps: int
    initial_distance: float
    normalized_time: float
    effective_speed: float
    path_efficiency: float
    energy: float
    wall_hits: int
    zigzag_penalty: float


@dataclass
class Telemetry:
    force_segment_1: Vec2 = field(default_factory=Vec2)
    force_segment_2: Vec2 = field(default_factory=Vec2)
    propulsion: Vec2 = field(default_factory=Vec2)
    velocity: Vec2 = field(default_factory=Vec2)
    target_direction: Vec2 = field(default_factory=Vec2)
    wall_reaction: Vec2 = field(default_factory=Vec2)
    torque: float = 0.0
    reward: float = 0.0
    distance_to_target: float = 0.0
    useful_speed: float = 0.0
    lateral_speed: float = 0.0
    heading_error: float = 0.0
    action_index: int = 4
    captured: bool = False
    timed_out: bool = False
    zigzag_window_efficiency: float = 0.0
    zigzag_penalty: float = 0.0
    zigzag_active: bool = False
    zigzag_status: str = "OBSERVATION"


@dataclass
class MotionSample:
    position: Vec2
    distance: float
    heading_error: float


@dataclass
class StepOutcome:
    reward: float
    terminal: bool
    metrics: EpisodeMetrics | None


class CellWorld:
    """Physique indépendante de PyTorch, réutilisée par tous les mondes."""

    def __init__(self, targets: list[Vec2]) -> None:
        if not targets:
            raise ValueError("Une séquence de cibles est obligatoire.")
        self.targets = targets
        self.target_index = 0
        self.target = targets[0]
        self.cell = CellState()
        self.previous_distance = self.distance_to_target()
        self.initial_distance = max(self.previous_distance, 1.0)
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_path_length = 0.0
        self.episode_energy = 0.0
        self.episode_wall_hits = 0
        self.episode_zigzag_penalty = 0.0
        self.total_steps = 0
        self.total_captures = 0
        self.total_timeouts = 0
        self.trajectory = [Vec2(self.cell.position.x, self.cell.position.y)]
        self.motion_window = [self._motion_sample()]
        self.telemetry = Telemetry(distance_to_target=self.previous_distance)

    def distance_to_target(self) -> float:
        return (self.target - self.cell.position).length()

    def heading_error(self) -> float:
        relative = self.target - self.cell.position
        target_angle = math.atan2(relative.y, relative.x)
        difference = target_angle - self.cell.angle
        return math.atan2(math.sin(difference), math.cos(difference))

    def _motion_sample(self) -> MotionSample:
        return MotionSample(
            position=Vec2(self.cell.position.x, self.cell.position.y),
            distance=self.distance_to_target(),
            heading_error=self.heading_error(),
        )

    def observations(self) -> list[float]:
        relative_target = rotate_into_cell_frame(
            self.target - self.cell.position, self.cell.angle
        )
        local_velocity = rotate_into_cell_frame(self.cell.velocity, self.cell.angle)
        return [
            clamp(relative_target.x / (WORLD_WIDTH * 0.5), -1.5, 1.5),
            clamp(relative_target.y / (WORLD_HEIGHT * 0.5), -1.5, 1.5),
            local_velocity.x / MAX_CELL_SPEED,
            local_velocity.y / MAX_CELL_SPEED,
            self.cell.angular_velocity / MAX_CELL_ROTATION,
            self.cell.joint_angle_1 / MAX_BASE_ANGLE,
            self.cell.joint_angle_2 / MAX_INTERSEGMENT_ANGLE,
            self.cell.joint_speed_1 / MAX_JOINT_SPEED,
            self.cell.joint_speed_2 / MAX_JOINT_SPEED,
            math.sin(self.cell.gait_phase),
            math.cos(self.cell.gait_phase),
        ]

    def flagellum_geometry(
        self,
    ) -> tuple[Vec2, Vec2, Vec2, Vec2, Vec2, Vec2, Vec2, Vec2]:
        body_axis = direction(self.cell.angle)
        attachment = self.cell.position - body_axis * BODY_HALF_LENGTH
        angle_1 = self.cell.angle + math.pi + self.cell.joint_angle_1
        tangent_1 = direction(angle_1)
        end_1 = attachment + tangent_1 * SEGMENT_LENGTH
        middle_1 = attachment + tangent_1 * (SEGMENT_LENGTH * 0.5)
        angle_2 = angle_1 + self.cell.joint_angle_2
        tangent_2 = direction(angle_2)
        end_2 = end_1 + tangent_2 * SEGMENT_LENGTH
        middle_2 = end_1 + tangent_2 * (SEGMENT_LENGTH * 0.5)
        return (
            attachment,
            end_1,
            end_2,
            middle_1,
            middle_2,
            tangent_1,
            tangent_2,
            body_axis,
        )

    @staticmethod
    def segment_drag(velocity: Vec2, tangent: Vec2) -> Vec2:
        normal = perpendicular(tangent)
        longitudinal = tangent * velocity.dot(tangent)
        lateral = normal * velocity.dot(normal)
        return longitudinal * (-DRAG_PARALLEL) + lateral * (-DRAG_PERPENDICULAR)

    def _apply_boundaries(self) -> tuple[bool, Vec2]:
        hit = False
        reaction = Vec2()
        minimum_x = WORLD_MARGIN + CELL_RADIUS
        maximum_x = WORLD_WIDTH - WORLD_MARGIN - CELL_RADIUS
        minimum_y = WORLD_MARGIN + CELL_RADIUS
        maximum_y = WORLD_HEIGHT - WORLD_MARGIN - CELL_RADIUS

        if self.cell.position.x < minimum_x:
            correction = minimum_x - self.cell.position.x
            self.cell.position.x = minimum_x
            self.cell.velocity.x = abs(self.cell.velocity.x) * 0.20
            reaction.x += correction + 0.3
            hit = True
        elif self.cell.position.x > maximum_x:
            correction = self.cell.position.x - maximum_x
            self.cell.position.x = maximum_x
            self.cell.velocity.x = -abs(self.cell.velocity.x) * 0.20
            reaction.x -= correction + 0.3
            hit = True

        if self.cell.position.y < minimum_y:
            correction = minimum_y - self.cell.position.y
            self.cell.position.y = minimum_y
            self.cell.velocity.y = abs(self.cell.velocity.y) * 0.20
            reaction.y += correction + 0.3
            hit = True
        elif self.cell.position.y > maximum_y:
            correction = self.cell.position.y - maximum_y
            self.cell.position.y = maximum_y
            self.cell.velocity.y = -abs(self.cell.velocity.y) * 0.20
            reaction.y -= correction + 0.3
            hit = True
        return hit, reaction

    def _assess_zigzag(
        self, wall_hit: bool
    ) -> tuple[float, float, bool, str]:
        current = self._motion_sample()
        if wall_hit:
            self.motion_window = [current]
            return 0.0, 0.0, False, "PAROI / REPOSITIONNEMENT"

        self.motion_window.append(current)
        maximum_points = ZIGZAG_WINDOW_STEPS + 1
        if len(self.motion_window) > maximum_points:
            del self.motion_window[: len(self.motion_window) - maximum_points]
        if len(self.motion_window) < maximum_points:
            return 0.0, 0.0, False, "OBSERVATION"

        path_length = sum(
            (following.position - previous.position).length()
            for previous, following in zip(
                self.motion_window, self.motion_window[1:]
            )
        )
        net_progress = self.motion_window[0].distance - self.motion_window[-1].distance
        efficiency = clamp(net_progress / max(path_length, 1e-9), -1.0, 1.0)
        aligned_fraction = sum(
            abs(sample.heading_error) <= ZIGZAG_ALIGNMENT_LIMIT
            for sample in self.motion_window
        ) / len(self.motion_window)
        heading_gain = (
            abs(self.motion_window[0].heading_error)
            - abs(self.motion_window[-1].heading_error)
        )

        if net_progress < ZIGZAG_MIN_PROGRESS:
            return 0.0, efficiency, False, "PAS DE CONVERGENCE"
        if abs(self.motion_window[-1].heading_error) > ZIGZAG_ALIGNMENT_LIMIT:
            return 0.0, efficiency, False, "VIRAGE / DEMI-TOUR"
        if heading_gain > ZIGZAG_REPOSITION_GAIN:
            return 0.0, efficiency, False, "ALIGNEMENT EN COURS"
        if aligned_fraction < ZIGZAG_ALIGNMENT_RATIO:
            return 0.0, efficiency, False, "REPOSITIONNEMENT"

        deficit = max(0.0, ZIGZAG_TARGET_EFFICIENCY - efficiency)
        penalty = ZIGZAG_PENALTY_COEFFICIENT * deficit
        if penalty <= 0.0:
            return 0.0, efficiency, False, "TRAJECTOIRE DIRECTE"
        return penalty, efficiency, True, "ZIGZAG PÉNALISÉ"

    def _advance_target(self) -> None:
        self.target_index = (self.target_index + 1) % len(self.targets)
        self.target = self.targets[self.target_index]
        self.previous_distance = self.distance_to_target()
        self.initial_distance = max(self.previous_distance, 1.0)
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_path_length = 0.0
        self.episode_energy = 0.0
        self.episode_wall_hits = 0
        self.episode_zigzag_penalty = 0.0
        self.motion_window = [self._motion_sample()]

    def step(self, action_index: int, remember_trajectory: bool) -> StepOutcome:
        command_1, command_2 = ACTIONS[action_index]
        self.cell.gait_phase = (
            self.cell.gait_phase + 2.0 * math.pi / GAIT_PERIOD
        ) % (2.0 * math.pi)

        self.cell.joint_speed_1 = clamp(
            self.cell.joint_speed_1 * JOINT_DAMPING
            + command_1 * MOTOR_ACCELERATION,
            -MAX_JOINT_SPEED,
            MAX_JOINT_SPEED,
        )
        self.cell.joint_speed_2 = clamp(
            self.cell.joint_speed_2 * JOINT_DAMPING
            + command_2 * MOTOR_ACCELERATION,
            -MAX_JOINT_SPEED,
            MAX_JOINT_SPEED,
        )
        self.cell.joint_angle_1 += self.cell.joint_speed_1
        self.cell.joint_angle_2 += self.cell.joint_speed_2

        if abs(self.cell.joint_angle_1) > MAX_BASE_ANGLE:
            self.cell.joint_angle_1 = clamp(
                self.cell.joint_angle_1, -MAX_BASE_ANGLE, MAX_BASE_ANGLE
            )
            self.cell.joint_speed_1 *= -0.22
        if abs(self.cell.joint_angle_2) > MAX_INTERSEGMENT_ANGLE:
            self.cell.joint_angle_2 = clamp(
                self.cell.joint_angle_2,
                -MAX_INTERSEGMENT_ANGLE,
                MAX_INTERSEGMENT_ANGLE,
            )
            self.cell.joint_speed_2 *= -0.22

        (
            _attachment,
            _end_1,
            _end_2,
            middle_1,
            middle_2,
            tangent_1,
            tangent_2,
            body_axis,
        ) = self.flagellum_geometry()

        velocity_middle_1 = perpendicular(tangent_1) * (
            SEGMENT_LENGTH * 0.5 * self.cell.joint_speed_1
        )
        velocity_joint_2 = perpendicular(tangent_1) * (
            SEGMENT_LENGTH * self.cell.joint_speed_1
        )
        velocity_middle_2 = velocity_joint_2 + perpendicular(tangent_2) * (
            SEGMENT_LENGTH
            * 0.5
            * (self.cell.joint_speed_1 + self.cell.joint_speed_2)
        )
        force_1 = self.segment_drag(velocity_middle_1, tangent_1)
        force_2 = self.segment_drag(velocity_middle_2, tangent_2)

        shape_rate = (
            self.cell.joint_angle_1 * self.cell.joint_speed_2
            - self.cell.joint_angle_2 * self.cell.joint_speed_1
        )
        shape_force = body_axis * (SHAPE_PROPULSION * shape_rate)
        force_1 = force_1 + shape_force * 0.5
        force_2 = force_2 + shape_force * 0.5
        total_force = force_1 + force_2

        torque = (middle_1 - self.cell.position).cross(force_1)
        torque += (middle_2 - self.cell.position).cross(force_2)
        desired_rotation = clamp(
            torque * TORQUE_TO_ROTATION,
            -MAX_CELL_ROTATION,
            MAX_CELL_ROTATION,
        )
        self.cell.angular_velocity = (
            0.40 * self.cell.angular_velocity + 0.60 * desired_rotation
        )
        self.cell.angle += self.cell.angular_velocity
        self.cell.angle = math.atan2(math.sin(self.cell.angle), math.cos(self.cell.angle))

        propulsion = total_force * FORCE_TO_SPEED
        old_position = Vec2(self.cell.position.x, self.cell.position.y)
        self.cell.velocity = self.cell.velocity * 0.32 + propulsion * 0.68
        if self.cell.velocity.length() > MAX_CELL_SPEED:
            self.cell.velocity = self.cell.velocity.normalized() * MAX_CELL_SPEED
        self.cell.position = self.cell.position + self.cell.velocity
        wall_hit, wall_reaction = self._apply_boundaries()

        displacement = (self.cell.position - old_position).length()
        energy = float(command_1 * command_1 + command_2 * command_2)
        self.episode_path_length += displacement
        self.episode_energy += energy
        self.episode_step += 1
        self.total_steps += 1
        if wall_hit:
            self.episode_wall_hits += 1

        current_distance = self.distance_to_target()
        current_heading_error = self.heading_error()
        target_direction = (self.target - self.cell.position).normalized()
        useful_speed = self.cell.velocity.dot(target_direction)
        lateral_speed = abs(self.cell.velocity.cross(target_direction))
        normalized_progress = (
            self.previous_distance - current_distance
        ) / max(self.initial_distance, 1.0)

        reward = PROGRESS_REWARD * normalized_progress
        reward -= TIME_PENALTY
        reward -= ENERGY_PENALTY * energy
        if wall_hit:
            reward -= WALL_PENALTY

        (
            zigzag_penalty,
            window_efficiency,
            zigzag_active,
            zigzag_status,
        ) = self._assess_zigzag(wall_hit)
        reward -= zigzag_penalty
        self.episode_zigzag_penalty += zigzag_penalty

        captured = current_distance <= CAPTURE_RADIUS
        timeout_limit = int(TIMEOUT_BASE + TIMEOUT_PER_PIXEL * self.initial_distance)
        timed_out = self.episode_step >= timeout_limit
        if captured:
            reward += CAPTURE_REWARD
            self.total_captures += 1
        elif timed_out:
            self.total_timeouts += 1

        self.episode_reward += reward
        self.previous_distance = current_distance
        terminal = captured or timed_out
        metrics: EpisodeMetrics | None = None
        if terminal:
            effective_speed = (
                self.initial_distance / max(self.episode_step, 1)
                if captured
                else 0.0
            )
            path_efficiency = (
                min(1.0, self.initial_distance / max(self.episode_path_length, 1e-9))
                if captured
                else 0.0
            )
            metrics = EpisodeMetrics(
                reward=self.episode_reward,
                captured=captured,
                steps=self.episode_step,
                initial_distance=self.initial_distance,
                normalized_time=self.episode_step / max(self.initial_distance, 1.0),
                effective_speed=effective_speed,
                path_efficiency=path_efficiency,
                energy=self.episode_energy,
                wall_hits=self.episode_wall_hits,
                zigzag_penalty=self.episode_zigzag_penalty,
            )
            self._advance_target()

        if remember_trajectory:
            self.trajectory.append(Vec2(self.cell.position.x, self.cell.position.y))
            if len(self.trajectory) > 650:
                del self.trajectory[:100]

        self.telemetry = Telemetry(
            force_segment_1=force_1,
            force_segment_2=force_2,
            propulsion=propulsion,
            velocity=self.cell.velocity,
            target_direction=(self.target - self.cell.position).normalized(),
            wall_reaction=wall_reaction,
            torque=torque,
            reward=reward,
            distance_to_target=self.distance_to_target(),
            useful_speed=useful_speed,
            lateral_speed=lateral_speed,
            heading_error=current_heading_error,
            action_index=action_index,
            captured=captured,
            timed_out=timed_out and not captured,
            zigzag_window_efficiency=window_efficiency,
            zigzag_penalty=zigzag_penalty,
            zigzag_active=zigzag_active,
            zigzag_status=zigzag_status,
        )
        return StepOutcome(reward, terminal, metrics)

    def to_dict(self) -> dict[str, Any]:
        def vector_data(vector: Vec2) -> list[float]:
            return [vector.x, vector.y]

        return {
            "targets": [vector_data(target) for target in self.targets],
            "target_index": self.target_index,
            "cell": {
                "position": vector_data(self.cell.position),
                "angle": self.cell.angle,
                "velocity": vector_data(self.cell.velocity),
                "angular_velocity": self.cell.angular_velocity,
                "joint_angle_1": self.cell.joint_angle_1,
                "joint_angle_2": self.cell.joint_angle_2,
                "joint_speed_1": self.cell.joint_speed_1,
                "joint_speed_2": self.cell.joint_speed_2,
                "gait_phase": self.cell.gait_phase,
            },
            "previous_distance": self.previous_distance,
            "initial_distance": self.initial_distance,
            "episode_step": self.episode_step,
            "episode_reward": self.episode_reward,
            "episode_path_length": self.episode_path_length,
            "episode_energy": self.episode_energy,
            "episode_wall_hits": self.episode_wall_hits,
            "episode_zigzag_penalty": self.episode_zigzag_penalty,
            "total_steps": self.total_steps,
            "total_captures": self.total_captures,
            "total_timeouts": self.total_timeouts,
            "trajectory": [vector_data(point) for point in self.trajectory],
            "motion_window": [
                {
                    "position": vector_data(sample.position),
                    "distance": sample.distance,
                    "heading_error": sample.heading_error,
                }
                for sample in self.motion_window
            ],
            "telemetry": {
                **asdict(self.telemetry),
                "force_segment_1": vector_data(self.telemetry.force_segment_1),
                "force_segment_2": vector_data(self.telemetry.force_segment_2),
                "propulsion": vector_data(self.telemetry.propulsion),
                "velocity": vector_data(self.telemetry.velocity),
                "target_direction": vector_data(self.telemetry.target_direction),
                "wall_reaction": vector_data(self.telemetry.wall_reaction),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CellWorld":
        def vector(values: Any) -> Vec2:
            if not isinstance(values, list) or len(values) != 2:
                raise ValueError("Vecteur de session invalide.")
            return Vec2(float(values[0]), float(values[1]))

        world = cls([vector(values) for values in data["targets"]])
        world.target_index = int(data["target_index"])
        world.target = world.targets[world.target_index]
        cell = data["cell"]
        world.cell = CellState(
            position=vector(cell["position"]),
            angle=float(cell["angle"]),
            velocity=vector(cell["velocity"]),
            angular_velocity=float(cell["angular_velocity"]),
            joint_angle_1=float(cell["joint_angle_1"]),
            joint_angle_2=float(cell["joint_angle_2"]),
            joint_speed_1=float(cell["joint_speed_1"]),
            joint_speed_2=float(cell["joint_speed_2"]),
            gait_phase=float(cell.get("gait_phase", 0.0)),
        )
        for name in (
            "previous_distance",
            "initial_distance",
            "episode_reward",
            "episode_path_length",
            "episode_energy",
            "episode_zigzag_penalty",
        ):
            setattr(world, name, float(data[name]))
        for name in (
            "episode_step",
            "episode_wall_hits",
            "total_steps",
            "total_captures",
            "total_timeouts",
        ):
            setattr(world, name, int(data[name]))
        world.trajectory = [vector(item) for item in data["trajectory"]]
        world.motion_window = [
            MotionSample(
                position=vector(item["position"]),
                distance=float(item["distance"]),
                heading_error=float(item["heading_error"]),
            )
            for item in data["motion_window"]
        ]
        telemetry = data["telemetry"]
        world.telemetry = Telemetry(
            force_segment_1=vector(telemetry["force_segment_1"]),
            force_segment_2=vector(telemetry["force_segment_2"]),
            propulsion=vector(telemetry["propulsion"]),
            velocity=vector(telemetry["velocity"]),
            target_direction=vector(telemetry["target_direction"]),
            wall_reaction=vector(telemetry["wall_reaction"]),
            torque=float(telemetry["torque"]),
            reward=float(telemetry["reward"]),
            distance_to_target=float(telemetry["distance_to_target"]),
            useful_speed=float(telemetry["useful_speed"]),
            lateral_speed=float(telemetry["lateral_speed"]),
            heading_error=float(telemetry["heading_error"]),
            action_index=int(telemetry["action_index"]),
            captured=bool(telemetry["captured"]),
            timed_out=bool(telemetry["timed_out"]),
            zigzag_window_efficiency=float(telemetry["zigzag_window_efficiency"]),
            zigzag_penalty=float(telemetry["zigzag_penalty"]),
            zigzag_active=bool(telemetry["zigzag_active"]),
            zigzag_status=str(telemetry["zigzag_status"]),
        )
        return world


# ---------------------------------------------------------------------------
# Réseau PyTorch et PPO
# ---------------------------------------------------------------------------


if TORCH_AVAILABLE:

    class PPOActorCritic(nn.Module):  # type: ignore[misc]
        """Tronc partagé 11 -> 64 -> 64, puis têtes Actor et Critic."""

        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE),
                nn.Tanh(),
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                nn.Tanh(),
            )
            self.actor = nn.Linear(HIDDEN_SIZE, ACTION_COUNT)
            self.critic = nn.Linear(HIDDEN_SIZE, 1)
            self._initialize()

        def _initialize(self) -> None:
            for layer in self.shared:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
            nn.init.zeros_(self.actor.bias)
            nn.init.orthogonal_(self.critic.weight, gain=1.0)
            nn.init.zeros_(self.critic.bias)

        def forward(self, observations: Any) -> tuple[Any, Any]:
            hidden = self.shared(observations)
            return self.actor(hidden), self.critic(hidden).squeeze(-1)

else:

    class PPOActorCritic:  # type: ignore[no-redef]
        def __init__(self) -> None:
            raise RuntimeError(
                "PyTorch est requis. Installez-le avec : py -m pip install torch"
            )


@dataclass
class PPOHistory:
    rewards: list[float] = field(default_factory=list)
    successes: list[float] = field(default_factory=list)
    path_efficiencies: list[float] = field(default_factory=list)
    effective_speeds: list[float] = field(default_factory=list)
    zigzag_penalties: list[float] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    approximate_kls: list[float] = field(default_factory=list)
    clip_fractions: list[float] = field(default_factory=list)


@dataclass
class PPOUpdateMetrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approximate_kl: float = 0.0
    clip_fraction: float = 0.0
    gradient_norm: float = 0.0
    entropy_coefficient: float = ENTROPY_START


class PPOTrainer:
    """Collecte des trajectoires et entraîne un unique réseau par PPO."""

    def __init__(self, seed: int = 1234, device_name: str = "cpu") -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch n'est pas installé.")
        self.seed = seed
        self.device_name = device_name
        self.device = torch.device(device_name)
        random.seed(seed)
        torch.manual_seed(seed)
        self.model = PPOActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.action_generator = torch.Generator(device=self.device)
        self.action_generator.manual_seed(seed + 100)
        self.shuffle_generator = torch.Generator(device="cpu")
        self.shuffle_generator.manual_seed(seed + 200)
        self.worlds = [
            CellWorld(
                generate_target_sequence(
                    seed + 1000 + index * 37, TARGET_SEQUENCE_LENGTH
                )
            )
            for index in range(ENVIRONMENT_COUNT)
        ]
        self.history = PPOHistory()
        self.updates = 0
        self.total_environment_steps = 0
        self.episodes = 0
        self.captures = 0
        self.last_episode: EpisodeMetrics | None = None
        self.last_update = PPOUpdateMetrics()

    def entropy_coefficient(self) -> float:
        fraction = min(1.0, self.updates / max(1, ENTROPY_DECAY_UPDATES))
        return ENTROPY_START + fraction * (ENTROPY_END - ENTROPY_START)

    def observations_tensor(self) -> Any:
        return torch.tensor(
            [world.observations() for world in self.worlds],
            dtype=torch.float32,
            device=self.device,
        )

    @staticmethod
    def sample_actions(logits: Any, generator: Any, deterministic: bool) -> Any:
        if deterministic:
            return logits.argmax(dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        return torch.multinomial(
            probabilities, num_samples=1, generator=generator
        ).squeeze(-1)

    def infer(
        self, observations: list[float], generator: Any, deterministic: bool
    ) -> int:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(
                [observations], dtype=torch.float32, device=self.device
            )
            logits, _value = self.model(tensor)
            action = self.sample_actions(logits, generator, deterministic)
        return int(action.item())

    def _record_episode(self, metrics: EpisodeMetrics) -> None:
        self.episodes += 1
        if metrics.captured:
            self.captures += 1
        self.last_episode = metrics
        self.history.rewards.append(metrics.reward)
        self.history.successes.append(1.0 if metrics.captured else 0.0)
        self.history.path_efficiencies.append(metrics.path_efficiency)
        self.history.effective_speeds.append(metrics.effective_speed)
        self.history.zigzag_penalties.append(metrics.zigzag_penalty)

    def success_rate(self, window: int = 20) -> float:
        values = self.history.successes[-window:]
        return sum(values) / len(values) if values else 0.0

    def ppo_update(self) -> PPOUpdateMetrics:
        self.model.train()
        observations: list[Any] = []
        actions: list[Any] = []
        old_log_probabilities: list[Any] = []
        rewards: list[Any] = []
        dones: list[Any] = []
        values: list[Any] = []

        current_observations = self.observations_tensor()
        for _step in range(ROLLOUT_STEPS):
            with torch.no_grad():
                logits, value = self.model(current_observations)
                action = self.sample_actions(
                    logits, self.action_generator, deterministic=False
                )
                distribution = Categorical(logits=logits)
                log_probability = distribution.log_prob(action)

            step_rewards: list[float] = []
            step_dones: list[float] = []
            for world, action_value in zip(self.worlds, action.tolist()):
                outcome = world.step(int(action_value), remember_trajectory=False)
                step_rewards.append(outcome.reward)
                step_dones.append(1.0 if outcome.terminal else 0.0)
                if outcome.metrics is not None:
                    self._record_episode(outcome.metrics)

            observations.append(current_observations)
            actions.append(action)
            old_log_probabilities.append(log_probability)
            values.append(value)
            rewards.append(
                torch.tensor(step_rewards, dtype=torch.float32, device=self.device)
            )
            dones.append(
                torch.tensor(step_dones, dtype=torch.float32, device=self.device)
            )
            current_observations = self.observations_tensor()

        with torch.no_grad():
            _next_logits, next_value = self.model(current_observations)

        observation_tensor = torch.stack(observations)
        action_tensor = torch.stack(actions)
        old_log_probability_tensor = torch.stack(old_log_probabilities)
        reward_tensor = torch.stack(rewards)
        done_tensor = torch.stack(dones)
        value_tensor = torch.stack(values)

        advantages = torch.zeros_like(reward_tensor)
        gae = torch.zeros(ENVIRONMENT_COUNT, device=self.device)
        bootstrap_value = next_value
        for step_index in reversed(range(ROLLOUT_STEPS)):
            not_terminal = 1.0 - done_tensor[step_index]
            delta = (
                reward_tensor[step_index]
                + GAMMA * bootstrap_value * not_terminal
                - value_tensor[step_index]
            )
            gae = delta + GAMMA * GAE_LAMBDA * not_terminal * gae
            advantages[step_index] = gae
            bootstrap_value = value_tensor[step_index]
        returns = advantages + value_tensor

        batch_observations = observation_tensor.reshape(-1, OBSERVATION_SIZE)
        batch_actions = action_tensor.reshape(-1)
        batch_old_log_probabilities = old_log_probability_tensor.reshape(-1)
        batch_old_values = value_tensor.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_advantages = (
            batch_advantages - batch_advantages.mean()
        ) / (batch_advantages.std(unbiased=False) + 1e-8)

        batch_size = batch_observations.shape[0]
        entropy_coefficient = self.entropy_coefficient()
        sums = {
            "policy": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip": 0.0,
            "gradient": 0.0,
        }
        minibatch_count = 0
        stop_early = False

        for _epoch in range(PPO_EPOCHS):
            permutation = torch.randperm(
                batch_size, generator=self.shuffle_generator
            ).to(self.device)
            for start in range(0, batch_size, MINIBATCH_SIZE):
                indices = permutation[start : start + MINIBATCH_SIZE]
                logits, new_values = self.model(batch_observations[indices])
                distribution = Categorical(logits=logits)
                new_log_probabilities = distribution.log_prob(batch_actions[indices])
                entropy = distribution.entropy().mean()

                log_ratio = (
                    new_log_probabilities
                    - batch_old_log_probabilities[indices]
                )
                ratio = log_ratio.exp()
                advantage_slice = batch_advantages[indices]
                unclipped = -advantage_slice * ratio
                clipped = -advantage_slice * torch.clamp(
                    ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP
                )
                policy_loss = torch.maximum(unclipped, clipped).mean()

                old_values = batch_old_values[indices]
                clipped_values = old_values + torch.clamp(
                    new_values - old_values, -VALUE_CLIP, VALUE_CLIP
                )
                value_loss_unclipped = (new_values - batch_returns[indices]).pow(2)
                value_loss_clipped = (
                    clipped_values - batch_returns[indices]
                ).pow(2)
                value_loss = 0.5 * torch.maximum(
                    value_loss_unclipped, value_loss_clipped
                ).mean()
                loss = (
                    policy_loss
                    + VALUE_COEFFICIENT * value_loss
                    - entropy_coefficient * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), MAX_GRADIENT_NORM
                )
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > PPO_CLIP).float().mean()
                    )
                sums["policy"] += float(policy_loss.item())
                sums["value"] += float(value_loss.item())
                sums["entropy"] += float(entropy.item())
                sums["kl"] += float(approximate_kl.item())
                sums["clip"] += float(clip_fraction.item())
                sums["gradient"] += float(gradient_norm.item())
                minibatch_count += 1
                if float(approximate_kl.item()) > TARGET_KL:
                    stop_early = True
                    break
            if stop_early:
                break

        divisor = max(1, minibatch_count)
        result = PPOUpdateMetrics(
            policy_loss=sums["policy"] / divisor,
            value_loss=sums["value"] / divisor,
            entropy=sums["entropy"] / divisor,
            approximate_kl=sums["kl"] / divisor,
            clip_fraction=sums["clip"] / divisor,
            gradient_norm=sums["gradient"] / divisor,
            entropy_coefficient=entropy_coefficient,
        )
        self.last_update = result
        self.updates += 1
        self.total_environment_steps += ROLLOUT_STEPS * ENVIRONMENT_COUNT
        self.history.policy_losses.append(result.policy_loss)
        self.history.value_losses.append(result.value_loss)
        self.history.entropies.append(result.entropy)
        self.history.approximate_kls.append(result.approximate_kl)
        self.history.clip_fractions.append(result.clip_fraction)
        return result

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": SESSION_FORMAT,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "action_generator_state": self.action_generator.get_state(),
            "shuffle_generator_state": self.shuffle_generator.get_state(),
            "seed": self.seed,
            "device_name": self.device_name,
            "worlds": [world.to_dict() for world in self.worlds],
            "history": asdict(self.history),
            "updates": self.updates,
            "total_environment_steps": self.total_environment_steps,
            "episodes": self.episodes,
            "captures": self.captures,
            "last_episode": asdict(self.last_episode) if self.last_episode else None,
            "last_update": asdict(self.last_update),
        }

    def restore_checkpoint(self, data: dict[str, Any]) -> None:
        if data.get("format") != SESSION_FORMAT:
            raise ValueError("Format de sauvegarde PyTorch incompatible.")
        self.model.load_state_dict(data["model_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        torch.set_rng_state(data["torch_rng_state"].cpu())
        self.action_generator.set_state(data["action_generator_state"])
        self.shuffle_generator.set_state(data["shuffle_generator_state"])
        self.worlds = [CellWorld.from_dict(item) for item in data["worlds"]]
        self.history = PPOHistory(**data["history"])
        self.updates = int(data["updates"])
        self.total_environment_steps = int(data["total_environment_steps"])
        self.episodes = int(data["episodes"])
        self.captures = int(data["captures"])
        last_episode = data.get("last_episode")
        self.last_episode = (
            EpisodeMetrics(**last_episode) if isinstance(last_episode, dict) else None
        )
        self.last_update = PPOUpdateMetrics(**data["last_update"])


class Evaluator:
    """Banc d'essai reproductible, sans optimizer.step()."""

    def __init__(self, trainer: PPOTrainer) -> None:
        self.trainer = trainer
        self.generator = torch.Generator(device=trainer.device)
        self.reset()

    def reset(self) -> None:
        self.world = CellWorld(
            generate_target_sequence(6060, TARGET_SEQUENCE_LENGTH)
        )
        self.episodes: list[EpisodeMetrics] = []
        self.generator.manual_seed(6061)

    def step(self, deterministic: bool) -> EpisodeMetrics | None:
        action = self.trainer.infer(
            self.world.observations(), self.generator, deterministic
        )
        outcome = self.world.step(action, remember_trajectory=True)
        if outcome.metrics is not None:
            self.episodes.append(outcome.metrics)
        return outcome.metrics

    @property
    def captures(self) -> int:
        return sum(metrics.captured for metrics in self.episodes)

    def average(self, attribute: str) -> float:
        return (
            sum(float(getattr(item, attribute)) for item in self.episodes)
            / len(self.episodes)
            if self.episodes
            else 0.0
        )

    def success_rate(self) -> float:
        return self.captures / len(self.episodes) if self.episodes else 0.0


# ---------------------------------------------------------------------------
# Interface Tkinter
# ---------------------------------------------------------------------------


class Application:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.root = tk.Tk()
        self.root.title("Cellule 2D — PyTorch PPO — réseau unique")
        self.root.geometry(f"{CANVAS_WIDTH}x{CANVAS_HEIGHT + 56}")
        self.root.minsize(CANVAS_WIDTH, CANVAS_HEIGHT + 56)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.trainer = PPOTrainer(seed=1234, device_name="cpu")
        self.evaluator = Evaluator(self.trainer)
        self.demo_world = CellWorld(
            generate_target_sequence(9090, TARGET_SEQUENCE_LENGTH)
        )
        self.demo_generator = torch.Generator(device=self.trainer.device)
        self.demo_generator.manual_seed(9091)

        self.learning_running = False
        self.evaluation_running = False
        self.evaluation_deterministic = False
        self.show_vectors = True
        self.capture_flash = 0
        self.training_busy = False
        self.closed = False
        self.cadences = (1, 50, 250)
        self.cadence_names = ("MAX", "NORMALE", "LENTE")
        self.cadence_index = 1

        toolbar = tk.Frame(self.root, bg="#18222f", padx=7, pady=7)
        toolbar.pack(fill="x")
        self.start_button = tk.Button(
            toolbar,
            text="Démarrer PPO",
            width=17,
            command=self.toggle_learning,
        )
        self.start_button.pack(side="left", padx=3)
        self.one_update_button = tk.Button(
            toolbar,
            text="1 mise à jour PPO",
            width=18,
            command=self.run_one_update,
        )
        self.one_update_button.pack(side="left", padx=3)
        self.evaluation_button = tk.Button(
            toolbar,
            text="Évaluer RN",
            width=15,
            command=self.start_evaluation,
        )
        self.evaluation_button.pack(side="left", padx=3)
        self.evaluation_mode_button = tk.Button(
            toolbar,
            text="Test : stochastique",
            width=19,
            command=self.toggle_evaluation_mode,
        )
        self.evaluation_mode_button.pack(side="left", padx=3)
        self.cadence_button = tk.Button(
            toolbar,
            text="Cadence : NORMALE",
            width=19,
            command=self.cycle_cadence,
        )
        self.cadence_button.pack(side="left", padx=3)
        self.vector_button = tk.Button(
            toolbar,
            text="Masquer vecteurs",
            width=16,
            command=self.toggle_vectors,
        )
        self.vector_button.pack(side="left", padx=3)
        tk.Button(
            toolbar,
            text="Sauver session",
            width=15,
            command=self.save_session,
        ).pack(side="left", padx=3)
        tk.Button(
            toolbar,
            text="Restaurer session",
            width=17,
            command=self.load_session,
        ).pack(side="left", padx=3)
        tk.Button(
            toolbar,
            text="Réinitialiser",
            width=14,
            command=self.reset_network,
        ).pack(side="right", padx=3)

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="#07111b",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.tick()
        self.training_cycle()

    def close(self) -> None:
        self.closed = True
        self.learning_running = False
        self.root.destroy()

    def toggle_learning(self) -> None:
        if self.evaluation_running:
            self.evaluation_running = False
            self.learning_running = True
            self.evaluation_button.configure(text="Évaluer RN")
        else:
            self.learning_running = not self.learning_running
        self.start_button.configure(
            text="Pause PPO" if self.learning_running else "Démarrer PPO"
        )
        self.one_update_button.configure(
            state="disabled" if self.learning_running else "normal"
        )

    def run_one_update(self) -> None:
        if self.training_busy:
            return
        self.learning_running = False
        self.evaluation_running = False
        self.start_button.configure(text="Démarrer PPO")
        self.evaluation_button.configure(text="Évaluer RN")
        self._perform_update()

    def start_evaluation(self) -> None:
        self.learning_running = False
        self.evaluation_running = True
        self.evaluator.reset()
        self.start_button.configure(text="Reprendre PPO")
        self.evaluation_button.configure(text="Relancer évaluation")
        self.one_update_button.configure(state="disabled")

    def toggle_evaluation_mode(self) -> None:
        self.evaluation_deterministic = not self.evaluation_deterministic
        mode = "déterministe" if self.evaluation_deterministic else "stochastique"
        self.evaluation_mode_button.configure(text=f"Test : {mode}")
        if self.evaluation_running:
            self.evaluator.reset()

    def cycle_cadence(self) -> None:
        self.cadence_index = (self.cadence_index + 1) % len(self.cadences)
        self.cadence_button.configure(
            text=f"Cadence : {self.cadence_names[self.cadence_index]}"
        )

    def toggle_vectors(self) -> None:
        self.show_vectors = not self.show_vectors
        self.vector_button.configure(
            text="Masquer vecteurs" if self.show_vectors else "Afficher vecteurs"
        )

    def _perform_update(self) -> None:
        self.training_busy = True
        try:
            self.trainer.ppo_update()
        except Exception as exc:
            self.learning_running = False
            self.start_button.configure(text="Démarrer PPO")
            self.one_update_button.configure(state="normal")
            self.messagebox.showerror("Erreur pendant PPO", str(exc))
        finally:
            self.training_busy = False

    def training_cycle(self) -> None:
        if self.closed:
            return
        if self.learning_running and not self.training_busy:
            self._perform_update()
        delay = self.cadences[self.cadence_index]
        self.root.after(delay, self.training_cycle)

    @staticmethod
    def session_root() -> Path:
        return Path(__file__).resolve().parent / "sauvegardes_pytorch"

    def _history_rows(self) -> list[list[Any]]:
        history = self.trainer.history
        length = max(
            len(history.rewards),
            len(history.policy_losses),
            1,
        )
        rows: list[list[Any]] = []
        for index in range(length):
            def value(values: list[float]) -> float | str:
                return values[index] if index < len(values) else ""

            rows.append(
                [
                    index + 1,
                    value(history.rewards),
                    value(history.successes),
                    value(history.path_efficiencies),
                    value(history.effective_speeds),
                    value(history.zigzag_penalties),
                    value(history.policy_losses),
                    value(history.value_losses),
                    value(history.entropies),
                    value(history.approximate_kls),
                    value(history.clip_fractions),
                ]
            )
        return rows

    def save_session(self) -> None:
        root = self.session_root()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = root / timestamp
        suffix = 1
        while directory.exists():
            directory = root / f"{timestamp}_{suffix:02d}"
            suffix += 1
        try:
            directory.mkdir(parents=True, exist_ok=False)
            checkpoint = self.trainer.checkpoint()
            checkpoint["saved_at"] = datetime.now().isoformat(timespec="seconds")
            checkpoint["demo_world"] = self.demo_world.to_dict()
            checkpoint["demo_generator_state"] = self.demo_generator.get_state()
            checkpoint["evaluator_world"] = self.evaluator.world.to_dict()
            checkpoint["evaluator_episodes"] = [
                asdict(item) for item in self.evaluator.episodes
            ]
            checkpoint["evaluator_generator_state"] = (
                self.evaluator.generator.get_state()
            )
            checkpoint["application"] = {
                "cadence_index": self.cadence_index,
                "show_vectors": self.show_vectors,
                "evaluation_deterministic": self.evaluation_deterministic,
            }
            torch.save(checkpoint, directory / SESSION_FILENAME)

            configuration = {
                "format": SESSION_FORMAT,
                "saved_at": checkpoint["saved_at"],
                "architecture": [OBSERVATION_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, 9, 1],
                "algorithm": "PPO + GAE",
                "environment_count": ENVIRONMENT_COUNT,
                "rollout_steps": ROLLOUT_STEPS,
                "batch_size": ENVIRONMENT_COUNT * ROLLOUT_STEPS,
                "ppo_epochs": PPO_EPOCHS,
                "minibatch_size": MINIBATCH_SIZE,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "ppo_clip": PPO_CLIP,
                "learning_rate": LEARNING_RATE,
                "device": self.trainer.device_name,
            }
            (directory / CONFIG_FILENAME).write_text(
                json.dumps(configuration, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            with (directory / HISTORY_FILENAME).open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream, delimiter=";")
                writer.writerow(
                    [
                        "index",
                        "reward",
                        "success",
                        "path_efficiency",
                        "effective_speed",
                        "zigzag_penalty",
                        "policy_loss",
                        "value_loss",
                        "entropy",
                        "approximate_kl",
                        "clip_fraction",
                    ]
                )
                writer.writerows(self._history_rows())
        except (OSError, RuntimeError, ValueError) as exc:
            self.messagebox.showerror("Sauvegarde impossible", str(exc))
            return
        self.messagebox.showinfo(
            "Session PyTorch sauvegardée",
            f"Checkpoint, configuration et historique :\n{directory}",
        )

    def load_session(self) -> None:
        root = self.session_root()
        root.mkdir(parents=True, exist_ok=True)
        selected = self.filedialog.askdirectory(
            title="Sélectionner un dossier de session PyTorch",
            initialdir=str(root),
            mustexist=True,
        )
        if not selected:
            return
        directory = Path(selected)
        checkpoint_path = directory / SESSION_FILENAME
        if not checkpoint_path.is_file():
            self.messagebox.showerror(
                "Dossier invalide",
                f"Le fichier {SESSION_FILENAME} est absent du dossier.",
            )
            return
        self.learning_running = False
        self.evaluation_running = False
        try:
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=self.trainer.device,
                    weights_only=True,
                )
            except TypeError:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=self.trainer.device,
                )
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint invalide.")
            self.trainer.restore_checkpoint(checkpoint)
            self.demo_world = CellWorld.from_dict(checkpoint["demo_world"])
            self.demo_generator.set_state(checkpoint["demo_generator_state"])
            self.evaluator = Evaluator(self.trainer)
            self.evaluator.world = CellWorld.from_dict(
                checkpoint["evaluator_world"]
            )
            self.evaluator.episodes = [
                EpisodeMetrics(**item)
                for item in checkpoint.get("evaluator_episodes", [])
            ]
            self.evaluator.generator.set_state(
                checkpoint["evaluator_generator_state"]
            )
            application = checkpoint.get("application", {})
            self.cadence_index = int(application.get("cadence_index", 1))
            self.show_vectors = bool(application.get("show_vectors", True))
            self.evaluation_deterministic = bool(
                application.get("evaluation_deterministic", False)
            )
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
            self.messagebox.showerror("Restauration impossible", str(exc))
            return

        self.start_button.configure(text="Démarrer PPO")
        self.one_update_button.configure(state="normal")
        self.evaluation_button.configure(text="Évaluer RN")
        mode = "déterministe" if self.evaluation_deterministic else "stochastique"
        self.evaluation_mode_button.configure(text=f"Test : {mode}")
        self.cadence_button.configure(
            text=f"Cadence : {self.cadence_names[self.cadence_index]}"
        )
        self.vector_button.configure(
            text="Masquer vecteurs" if self.show_vectors else "Afficher vecteurs"
        )
        self.messagebox.showinfo(
            "Session restaurée",
            "Réseau, optimiseur, mondes, historiques et générateurs restaurés.\n"
            "La session est volontairement en pause.",
        )

    def reset_network(self) -> None:
        if not self.messagebox.askyesno(
            "Réinitialiser PPO",
            "Effacer les poids, l'optimiseur et toutes les courbes ?",
        ):
            return
        self.learning_running = False
        self.evaluation_running = False
        seed = random.randrange(1, 1_000_000)
        self.trainer = PPOTrainer(seed=seed, device_name="cpu")
        self.evaluator = Evaluator(self.trainer)
        self.demo_world = CellWorld(
            generate_target_sequence(seed + 2, TARGET_SEQUENCE_LENGTH)
        )
        self.demo_generator = torch.Generator(device=self.trainer.device)
        self.demo_generator.manual_seed(seed + 3)
        self.start_button.configure(text="Démarrer PPO")
        self.one_update_button.configure(state="normal")
        self.evaluation_button.configure(text="Évaluer RN")

    def tick(self) -> None:
        if self.closed:
            return
        if self.learning_running and not self.training_busy:
            action = self.trainer.infer(
                self.demo_world.observations(),
                self.demo_generator,
                deterministic=False,
            )
            outcome = self.demo_world.step(action, remember_trajectory=True)
            if outcome.metrics is not None and outcome.metrics.captured:
                self.capture_flash = 12
        elif self.evaluation_running:
            metrics = self.evaluator.step(self.evaluation_deterministic)
            if metrics is not None and metrics.captured:
                self.capture_flash = 12
        if self.capture_flash > 0:
            self.capture_flash -= 1
        self.draw()
        self.root.after(33, self.tick)

    def visible_world(self) -> CellWorld:
        return self.evaluator.world if self.evaluation_running else self.demo_world

    def draw(self) -> None:
        self.canvas.delete("all")
        self.draw_world()
        self.draw_statistics()
        history = self.trainer.history
        self.draw_chart(
            1142,
            18,
            1418,
            342,
            "RÉCOMPENSE / RÉUSSITE",
            (
                ("récompense", history.rewards, "#78909c"),
                ("moyenne 20", moving_average(history.rewards, 20), "#62df79"),
            ),
        )
        self.draw_chart(
            1430,
            18,
            1708,
            342,
            "PERTES PPO",
            (
                ("Actor", history.policy_losses, "#47b7ff"),
                ("Critic", history.value_losses, "#ffad4d"),
            ),
            include_zero=True,
        )
        self.draw_chart(
            1142,
            354,
            1418,
            680,
            "EFFICACITÉ / SUCCÈS",
            (
                ("trajet", history.path_efficiencies, "#d58cff"),
                ("succès 20", moving_average(history.successes, 20), "#ffe066"),
            ),
            fixed_minimum=0.0,
            fixed_maximum=1.0,
        )
        normalized_entropy = [
            value / math.log(ACTION_COUNT) for value in history.entropies
        ]
        normalized_kl = [
            min(1.0, value / TARGET_KL) for value in history.approximate_kls
        ]
        self.draw_chart(
            1430,
            354,
            1708,
            680,
            "SANTÉ PPO (NORMALISÉE)",
            (
                ("entropie", normalized_entropy, "#66e0c2"),
                ("KL/cible", normalized_kl, "#ff7a90"),
                ("clip", history.clip_fractions, "#a98cff"),
            ),
            fixed_minimum=0.0,
            fixed_maximum=1.0,
        )

    def draw_world(self) -> None:
        c = self.canvas
        world = self.visible_world()
        for x in range(20, WORLD_WIDTH, 40):
            c.create_line(x, 20, x, WORLD_HEIGHT - 20, fill="#102536")
        for y in range(20, WORLD_HEIGHT, 40):
            c.create_line(20, y, WORLD_WIDTH - 20, y, fill="#102536")
        boundary_color = (
            "#ff8a65"
            if world.telemetry.wall_reaction.length() > 0
            else "#4b6475"
        )
        c.create_rectangle(
            WORLD_MARGIN,
            WORLD_MARGIN,
            WORLD_WIDTH - WORLD_MARGIN,
            WORLD_HEIGHT - WORLD_MARGIN,
            outline=boundary_color,
            width=3,
        )

        if len(world.trajectory) >= 2:
            coordinates: list[float] = []
            for point in world.trajectory:
                coordinates.extend((point.x, point.y))
            c.create_line(*coordinates, fill="#718792", width=2, smooth=True)

        target = world.target
        c.create_oval(
            target.x - CAPTURE_RADIUS,
            target.y - CAPTURE_RADIUS,
            target.x + CAPTURE_RADIUS,
            target.y + CAPTURE_RADIUS,
            outline="#f7d154",
            dash=(4, 4),
            width=2,
        )
        target_fill = "#ffffff" if self.capture_flash else "#ffd54f"
        c.create_oval(
            target.x - TARGET_RADIUS,
            target.y - TARGET_RADIUS,
            target.x + TARGET_RADIUS,
            target.y + TARGET_RADIUS,
            fill=target_fill,
            outline="#fff4b0",
            width=2,
        )
        c.create_text(
            target.x,
            target.y - CAPTURE_RADIUS - 10,
            text="CIBLE FIXE",
            fill="#ffe082",
            font=("Segoe UI", 9, "bold"),
        )

        (
            attachment,
            end_1,
            end_2,
            middle_1,
            middle_2,
            _tangent_1,
            _tangent_2,
            _body_axis,
        ) = world.flagellum_geometry()
        c.create_line(
            attachment.x,
            attachment.y,
            end_1.x,
            end_1.y,
            fill="#73d7ff",
            width=6,
            capstyle="round",
        )
        c.create_line(
            end_1.x,
            end_1.y,
            end_2.x,
            end_2.y,
            fill="#4eb7e5",
            width=5,
            capstyle="round",
        )
        for joint in (attachment, end_1):
            c.create_oval(
                joint.x - 4,
                joint.y - 4,
                joint.x + 4,
                joint.y + 4,
                fill="#d9f5ff",
                outline="#22789c",
            )

        body_points: list[float] = []
        cosine = math.cos(world.cell.angle)
        sine = math.sin(world.cell.angle)
        for index in range(24):
            phi = 2.0 * math.pi * index / 24
            local_x = BODY_HALF_LENGTH * math.cos(phi)
            local_y = BODY_HALF_WIDTH * math.sin(phi)
            body_points.extend(
                (
                    world.cell.position.x + cosine * local_x - sine * local_y,
                    world.cell.position.y + sine * local_x + cosine * local_y,
                )
            )
        c.create_polygon(
            *body_points,
            fill="#67d8a2",
            outline="#d8ffea",
            width=2,
            smooth=True,
        )
        front = world.cell.position + direction(world.cell.angle) * 13
        c.create_oval(
            front.x - 3,
            front.y - 3,
            front.x + 3,
            front.y + 3,
            fill="#153d2d",
        )

        telemetry = world.telemetry
        if self.show_vectors:
            self.draw_arrow(middle_1, telemetry.force_segment_1, 80, "#ff9f43", "F1")
            self.draw_arrow(middle_2, telemetry.force_segment_2, 80, "#ff7f50", "F2")
            self.draw_arrow(
                world.cell.position, telemetry.propulsion, 72, "#46e66d", "PROP"
            )
            self.draw_arrow(
                world.cell.position, telemetry.velocity, 78, "#48a9ff", "V"
            )
            self.draw_arrow(
                world.cell.position,
                telemetry.target_direction,
                68,
                "#cf8cff",
                "CIBLE",
                dash=(5, 3),
            )
            if telemetry.wall_reaction.length() > 0:
                self.draw_arrow(
                    world.cell.position,
                    telemetry.wall_reaction,
                    30,
                    "#ff4b4b",
                    "PAROI",
                )

        c.create_text(
            30,
            WORLD_HEIGHT - 10,
            text="F1/F2 : forces   PROP : propulsion   V : vitesse   phase : horloge du battement",
            anchor="sw",
            fill="#a9bbc6",
            font=("Segoe UI", 9),
        )

    def draw_statistics(self) -> None:
        c = self.canvas
        world = self.visible_world()
        telemetry = world.telemetry
        left, top, right, bottom = 842, 18, 1130, 680
        c.create_rectangle(
            left, top, right, bottom, fill="#0c1722", outline="#38566b", width=2
        )
        if self.learning_running:
            state, color = "APPRENTISSAGE PPO", "#66e58a"
        elif self.evaluation_running:
            mode = "DÉTERMINISTE" if self.evaluation_deterministic else "STOCHASTIQUE"
            state, color = f"ÉVALUATION {mode}", "#62c8ff"
        else:
            state, color = "PAUSE — MONDE FIGÉ", "#ffca5c"
        c.create_text(
            (left + right) * 0.5,
            top + 23,
            text="PYTORCH — RN UNIQUE",
            fill="#e9f4fa",
            font=("Segoe UI", 11, "bold"),
        )
        c.create_text(
            (left + right) * 0.5,
            top + 49,
            text=state,
            fill=color,
            font=("Segoe UI", 9, "bold"),
        )

        update = self.trainer.last_update
        ppo_info = (
            f"PPO / GAE\n"
            f"Mondes      : {ENVIRONMENT_COUNT}\n"
            f"Lot         : {ENVIRONMENT_COUNT * ROLLOUT_STEPS}\n"
            f"Mises à jour: {self.trainer.updates}\n"
            f"Pas simulés : {self.trainer.total_environment_steps}\n"
            f"Poursuites  : {self.trainer.episodes}\n"
            f"Captures    : {self.trainer.captures}\n"
            f"Réussite20  : {100*self.trainer.success_rate():5.1f}%\n"
            f"Loss Actor  : {update.policy_loss:+.5f}\n"
            f"Loss Critic : {update.value_loss:.5f}\n"
            f"Entropie    : {update.entropy:.4f}\n"
            f"KL approx.  : {update.approximate_kl:.5f}\n"
            f"Clip frac.  : {update.clip_fraction:.3f}"
        )
        c.create_text(
            left + 16,
            top + 76,
            text=ppo_info,
            anchor="nw",
            fill="#d8e7ef",
            font=("Consolas", 8),
        )

        speed = world.cell.velocity.length()
        command = ACTIONS[telemetry.action_index]
        movement_info = (
            f"CELLULE AFFICHÉE\n"
            f"Distance    : {telemetry.distance_to_target:7.2f}\n"
            f"Vitesse     : {speed:.4f}\n"
            f"Vers cible  : {telemetry.useful_speed:+.4f}\n"
            f"Latérale    : {telemetry.lateral_speed:.4f}\n"
            f"Erreur cap  : {math.degrees(telemetry.heading_error):+6.1f}°\n"
            f"Rotation    : {math.degrees(world.cell.angular_velocity):+.3f}°/pas\n"
            f"Phase       : {math.degrees(world.cell.gait_phase):6.1f}°\n"
            f"Articulation: {command}\n"
            f"Fenêtre η   : {telemetry.zigzag_window_efficiency:+.3f}\n"
            f"Pénalité    : {telemetry.zigzag_penalty:.6f}\n"
            f"État zigzag : {telemetry.zigzag_status}"
        )
        c.create_text(
            left + 16,
            top + 285,
            text=movement_info,
            anchor="nw",
            fill="#c5e3ef",
            font=("Consolas", 8),
        )

        if self.evaluation_running:
            episodes = self.evaluator.episodes
            eval_info = (
                f"ÉVALUATION FIGÉE\n"
                f"Poursuites : {len(episodes)}\n"
                f"Réussites  : {self.evaluator.captures}\n"
                f"Taux       : {100*self.evaluator.success_rate():5.1f}%\n"
                f"Pas moyens : {self.evaluator.average('steps'):.1f}\n"
                f"V.eff. moy.: {self.evaluator.average('effective_speed'):.4f}\n"
                f"Eff. trajet: {self.evaluator.average('path_efficiency'):.3f}\n"
                f"P.zigzag   : {self.evaluator.average('zigzag_penalty'):.4f}"
            )
        elif self.trainer.last_episode is None:
            eval_info = "DERNIÈRE POURSUITE\nEn attente"
        else:
            last = self.trainer.last_episode
            eval_info = (
                f"DERNIÈRE POURSUITE\n"
                f"Résultat   : {'SUCCÈS' if last.captured else 'ÉCHEC'}\n"
                f"Récompense : {last.reward:+.3f}\n"
                f"Pas        : {last.steps}\n"
                f"V.effective: {last.effective_speed:.4f}\n"
                f"Efficacité : {last.path_efficiency:.3f}\n"
                f"P.zigzag   : {last.zigzag_penalty:.4f}"
            )
        c.create_text(
            left + 16,
            top + 493,
            text=eval_info,
            anchor="nw",
            fill="#d9c8f3",
            font=("Consolas", 8),
        )
        c.create_text(
            (left + right) * 0.5,
            bottom - 17,
            text="Évaluation : aucun optimizer.step().",
            fill="#8fa9b8",
            font=("Segoe UI", 8),
        )

    def draw_arrow(
        self,
        origin: Vec2,
        vector: Vec2,
        scale: float,
        color: str,
        label: str,
        dash: tuple[int, int] | None = None,
    ) -> None:
        magnitude = vector.length()
        if magnitude < 1e-8:
            return
        display_length = min(92.0, max(7.0, magnitude * scale))
        endpoint = origin + vector.normalized() * display_length
        self.canvas.create_line(
            origin.x,
            origin.y,
            endpoint.x,
            endpoint.y,
            fill=color,
            width=2,
            arrow="last",
            dash=dash,
        )
        self.canvas.create_text(
            endpoint.x + 5,
            endpoint.y - 5,
            text=label,
            anchor="sw",
            fill=color,
            font=("Segoe UI", 8, "bold"),
        )

    def draw_chart(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        title: str,
        series: tuple[tuple[str, list[float], str], ...],
        include_zero: bool = False,
        fixed_minimum: float | None = None,
        fixed_maximum: float | None = None,
    ) -> None:
        c = self.canvas
        c.create_rectangle(
            left, top, right, bottom, fill="#0c1722", outline="#38566b", width=2
        )
        c.create_text(
            (left + right) * 0.5,
            top + 17,
            text=title,
            fill="#e9f4fa",
            font=("Segoe UI", 9, "bold"),
        )
        visible = [(name, values[-180:], color) for name, values, color in series]
        all_values = [value for _name, values, _color in visible for value in values]
        plot_left, plot_right = left + 34, right - 10
        plot_top, plot_bottom = top + 48, bottom - 30
        c.create_line(plot_left, plot_top, plot_left, plot_bottom, fill="#607d8b")
        c.create_line(
            plot_left, plot_bottom, plot_right, plot_bottom, fill="#607d8b"
        )
        if not all_values:
            c.create_text(
                (plot_left + plot_right) * 0.5,
                (plot_top + plot_bottom) * 0.5,
                text="En attente de données",
                fill="#68808d",
                font=("Segoe UI", 8),
            )
            return

        minimum = fixed_minimum if fixed_minimum is not None else min(all_values)
        maximum = fixed_maximum if fixed_maximum is not None else max(all_values)
        if include_zero:
            minimum = min(minimum, 0.0)
            maximum = max(maximum, 0.0)
        if abs(maximum - minimum) < 1e-12:
            maximum += 0.5
            minimum -= 0.5
        c.create_text(
            plot_left - 4,
            plot_top,
            text=f"{maximum:.3g}",
            anchor="e",
            fill="#8fa9b8",
            font=("Consolas", 7),
        )
        c.create_text(
            plot_left - 4,
            plot_bottom,
            text=f"{minimum:.3g}",
            anchor="e",
            fill="#8fa9b8",
            font=("Consolas", 7),
        )

        legend_x = left + 10
        for name, values, color in visible:
            if len(values) == 1:
                x_values = [plot_left]
            else:
                x_values = [
                    plot_left
                    + index * (plot_right - plot_left) / (len(values) - 1)
                    for index in range(len(values))
                ]
            points: list[float] = []
            for x_value, value in zip(x_values, values):
                y_value = plot_bottom - (
                    (value - minimum) / (maximum - minimum)
                ) * (plot_bottom - plot_top)
                points.extend((x_value, y_value))
            if len(points) >= 4:
                c.create_line(*points, fill=color, width=2, smooth=True)
            elif points:
                c.create_oval(
                    points[0] - 2,
                    points[1] - 2,
                    points[0] + 2,
                    points[1] + 2,
                    fill=color,
                    outline="",
                )
            c.create_text(
                legend_x,
                bottom - 14,
                text=name,
                anchor="w",
                fill=color,
                font=("Segoe UI", 7),
            )
            legend_x += max(58, len(name) * 6 + 12)


# ---------------------------------------------------------------------------
# Exécution et tests simples
# ---------------------------------------------------------------------------


def test_physics() -> None:
    world = CellWorld(generate_target_sequence(123, 20))
    rng = random.Random(456)
    origin = Vec2(world.cell.position.x, world.cell.position.y)
    for _ in range(5000):
        world.step(rng.randrange(ACTION_COUNT), remember_trajectory=False)
    displacement = (world.cell.position - origin).length()
    assert all(math.isfinite(value) for value in world.observations())
    assert -MAX_BASE_ANGLE <= world.cell.joint_angle_1 <= MAX_BASE_ANGLE
    assert (
        -MAX_INTERSEGMENT_ANGLE
        <= world.cell.joint_angle_2
        <= MAX_INTERSEGMENT_ANGLE
    )
    restored = CellWorld.from_dict(world.to_dict())
    assert restored.to_dict() == world.to_dict()
    print(
        "PHYSIQUE OK "
        f"pas={world.total_steps} déplacement={displacement:.2f} "
        f"captures={world.total_captures}"
    )


def run_headless(update_count: int) -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch n'est pas installé. Exécutez : py -m pip install torch"
        )
    trainer = PPOTrainer(seed=1234)
    for index in range(update_count):
        metrics = trainer.ppo_update()
        print(
            f"MAJ {index + 1:4d} "
            f"pas={trainer.total_environment_steps:8d} "
            f"épisodes={trainer.episodes:5d} "
            f"succès20={100*trainer.success_rate():5.1f}% "
            f"actor={metrics.policy_loss:+.5f} "
            f"critic={metrics.value_loss:.5f} "
            f"KL={metrics.approximate_kl:.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        type=int,
        metavar="MISES_A_JOUR",
        help="entraîne sans IHM pendant ce nombre de mises à jour PPO",
    )
    parser.add_argument(
        "--test-physique",
        action="store_true",
        help="teste uniquement la physique, sans exiger PyTorch",
    )
    arguments = parser.parse_args()

    if arguments.test_physique:
        test_physics()
        return
    if not TORCH_AVAILABLE:
        message = (
            "PyTorch est nécessaire pour cette version.\n"
            "Installez-le avec :\n\n"
            "    py -m pip install torch\n\n"
            f"Détail : {TORCH_IMPORT_ERROR}"
        )
        print(message, file=sys.stderr)
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("PyTorch absent", message)
            root.destroy()
        except Exception:
            pass
        raise SystemExit(1)
    if arguments.headless is not None:
        run_headless(max(0, arguments.headless))
        return

    application = Application()
    application.root.mainloop()


if __name__ == "__main__":
    main()