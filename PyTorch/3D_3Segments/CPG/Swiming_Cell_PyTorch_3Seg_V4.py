"""
Cellule nageuse 3D — PPO continu, réseau Actor-Critic unique et IHM Tkinter.

Cette expérience pédagogique étend la cellule à flagelle dans un cube 3D :

    - position, vitesse, orientation et rotation en trois dimensions ;
    - flagelle à trois segments, chaque articulation ayant deux axes ;
    - Actor continu à six commandes, Critic scalaire V(s) ;
    - apprentissage PPO avec GAE dans plusieurs mondes parallèles ;
    - curriculum automatique puis évaluation dans tout le volume ;
    - projection perspective réalisée directement dans Tkinter.

La physique représente une traînée visqueuse anisotrope et les couples produits
par le flagelle. Ce n'est pas un solveur de mécanique des fluides.

Dépendance :
    py -m pip install torch

Lancement :
    py cellule_flagelle_3d_ppo_pytorch.py

Test de la physique sans PyTorch :
    py cellule_flagelle_3d_ppo_pytorch.py --test-physique

Entraînement sans interface :
    py cellule_flagelle_3d_ppo_pytorch.py --headless 20

Sauvegardes :
    sauvegardes_pytorch_3d/AAAAMMJJ_HHMMSS/
"""

from __future__ import annotations

import argparse
import ast
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
    from torch.distributions import Normal

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ""
except ModuleNotFoundError as exc:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Normal = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CANVAS_WIDTH = 1760
CANVAS_HEIGHT = 760
WORLD_HALF_SIZE = 260.0
WORLD_MARGIN = 24.0

BODY_HALF_LENGTH = 20.0
BODY_RADIUS = 12.0
SEGMENT_COUNT = 3

# Morphologie simplifiée : le flagelle s'amincit en s'éloignant du corps.
# La longueur totale reste égale à celle de l'ancienne version (3 x 32 = 96),
# afin de modifier la répartition de l'effort sans allonger le flagelle.
SEGMENT_LENGTHS = (39.0, 32.0, 25.0)
SEGMENT_FORCE_WEIGHTS = (1.00, 0.78, 0.58)

TARGET_RADIUS = 11.0
CAPTURE_RADIUS = 22.0
MIN_TARGET_DISTANCE = 150.0
MAX_TARGET_DISTANCE = 390.0

# Deux courbures par articulation : lacet local et tangage local.
JOINT_LIMITS = (
    math.radians(25.0),
    math.radians(35.0),
    math.radians(40.0),
)
MAX_JOINT_SPEED = 0.16
MOTOR_ACCELERATION = 0.040
JOINT_DAMPING = 0.90

# Modèle visqueux simplifié. La traînée latérale supérieure à la traînée
# axiale permet à une onde non réciproque de produire une poussée nette.
DRAG_PARALLEL = 0.018
DRAG_PERPENDICULAR = 0.075
SHAPE_PROPULSION = 1.90
FORCE_TO_SPEED = 1.65
TORQUE_TO_ROTATION = 0.00115
LINEAR_DAMPING = 0.38
ANGULAR_DAMPING = 0.48
MAX_CELL_SPEED = 2.30
MAX_CELL_ROTATION = 0.055

GAIT_PERIOD = 36.0
TIMEOUT_BASE = 500
TIMEOUT_PER_UNIT = 5.0

PROGRESS_REWARD = 2.6
CAPTURE_REWARD = 4.0
TIME_PENALTY = 0.0012
ENERGY_PENALTY = 0.00035
WALL_PENALTY = 0.055
LATERAL_PENALTY = 0.00035

# Une posture immobile loin de la cible doit être moins intéressante qu'une
# exploration lente. Le délai laisse deux périodes complètes au flagelle pour
# construire son ondulation avant d'appliquer cette pénalité.
STAGNATION_SPEED = 0.035
STAGNATION_GRACE_STEPS = int(2.0 * GAIT_PERIOD)
STAGNATION_PENALTY = 0.0030

# 3 cible + 3 vitesse + 3 rotation + 6 angles + 6 vitesses + 2 phase.
OBSERVATION_SIZE = 23
ACTION_SIZE = 6
HIDDEN_SIZE = 128

ENVIRONMENT_COUNT = 12
ROLLOUT_STEPS = 96
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.20
VALUE_CLIP = 0.20
LEARNING_RATE = 3.0e-4
ENTROPY_START = 0.010
ENTROPY_END = 0.0015
ENTROPY_DECAY_UPDATES = 4500
VALUE_COEFFICIENT = 0.50
MAX_GRADIENT_NORM = 0.50
TARGET_KL = 0.030

SESSION_FORMAT = "cellule-ppo-pytorch-3d-v1"
SESSION_FILENAME = "checkpoint_3d.pt"
CONFIG_FILENAME = "configuration_3d.json"
HISTORY_FILENAME = "historique_3d.csv"

CURRICULUM_NAMES = (
    "CIBLE DANS UN CÔNE AVANT",
    "CIBLE DANS L'HÉMISPHÈRE AVANT",
    "CIBLE DANS TOUT LE CUBE",
)


# ---------------------------------------------------------------------------
# Outils mathématiques 3D
# ---------------------------------------------------------------------------


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        magnitude = self.length()
        return self / magnitude if magnitude > 1e-12 else Vec3()

    def limited(self, maximum: float) -> "Vec3":
        magnitude = self.length()
        return self * (maximum / magnitude) if magnitude > maximum else self


@dataclass
class Quaternion:
    """Quaternion unitaire transformant le repère du corps vers le monde."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def normalized(self) -> "Quaternion":
        magnitude = math.sqrt(
            self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
        )
        if magnitude < 1e-12:
            return Quaternion()
        return Quaternion(
            self.w / magnitude,
            self.x / magnitude,
            self.y / magnitude,
            self.z / magnitude,
        )

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def rotate(self, vector: Vec3) -> Vec3:
        pure = Quaternion(0.0, vector.x, vector.y, vector.z)
        result = self * pure * self.conjugate()
        return Vec3(result.x, result.y, result.z)

    def inverse_rotate(self, vector: Vec3) -> Vec3:
        return self.conjugate().rotate(vector)

    @staticmethod
    def from_axis_angle(axis: Vec3, angle: float) -> "Quaternion":
        unit = axis.normalized()
        half = 0.5 * angle
        sine = math.sin(half)
        return Quaternion(math.cos(half), unit.x * sine, unit.y * sine, unit.z * sine)

    @staticmethod
    def from_rotation_vector(rotation: Vec3) -> "Quaternion":
        angle = rotation.length()
        return (
            Quaternion.from_axis_angle(rotation / angle, angle)
            if angle > 1e-12
            else Quaternion()
        )


def random_unit_vector(rng: random.Random) -> Vec3:
    z = rng.uniform(-1.0, 1.0)
    azimuth = rng.uniform(0.0, 2.0 * math.pi)
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    return Vec3(radius * math.cos(azimuth), radius * math.sin(azimuth), z)


def moving_average(values: list[float], window: int = 20) -> list[float]:
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        result.append(running / min(index + 1, window))
    return result


# ---------------------------------------------------------------------------
# Environnement physique 3D
# ---------------------------------------------------------------------------


@dataclass
class CellState:
    position: Vec3 = field(default_factory=Vec3)
    orientation: Quaternion = field(default_factory=Quaternion)
    velocity: Vec3 = field(default_factory=Vec3)
    angular_velocity: Vec3 = field(default_factory=Vec3)
    # Pour chaque articulation : [lacet local, tangage local].
    joint_angles: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0] for _ in range(SEGMENT_COUNT)]
    )
    joint_speeds: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0] for _ in range(SEGMENT_COUNT)]
    )
    gait_phase: float = 0.0


@dataclass
class FlagellumGeometry:
    points: list[Vec3]
    middles: list[Vec3]
    tangents: list[Vec3]


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
    lateral_motion: float


@dataclass
class Telemetry:
    segment_forces: list[Vec3] = field(
        default_factory=lambda: [Vec3() for _ in range(SEGMENT_COUNT)]
    )
    propulsion: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    target_direction: Vec3 = field(default_factory=Vec3)
    wall_reaction: Vec3 = field(default_factory=Vec3)
    torque: Vec3 = field(default_factory=Vec3)
    reward: float = 0.0
    distance_to_target: float = 0.0
    useful_speed: float = 0.0
    lateral_speed: float = 0.0
    alignment: float = 0.0
    action: list[float] = field(default_factory=lambda: [0.0] * ACTION_SIZE)
    captured: bool = False
    timed_out: bool = False


@dataclass
class StepOutcome:
    reward: float
    terminal: bool
    metrics: EpisodeMetrics | None


class CellWorld3D:
    """Cube 3D et dynamique visqueuse simplifiée, indépendants de PyTorch."""

    def __init__(self, seed: int, curriculum_level: int = 0) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.curriculum_level = int(clamp(curriculum_level, 0, 2))
        self.cell = CellState()
        self.target = self._sample_target()
        self.previous_distance = self.distance_to_target()
        self.initial_distance = max(self.previous_distance, 1.0)
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_path_length = 0.0
        self.episode_energy = 0.0
        self.episode_wall_hits = 0
        self.episode_lateral_motion = 0.0
        self.total_steps = 0
        self.total_captures = 0
        self.total_timeouts = 0
        self.trajectory = [Vec3()]
        self.telemetry = Telemetry(distance_to_target=self.previous_distance)

    def distance_to_target(self) -> float:
        return (self.target - self.cell.position).length()

    def body_forward(self) -> Vec3:
        return self.cell.orientation.rotate(Vec3(1.0, 0.0, 0.0)).normalized()

    def _curriculum_direction(self) -> Vec3:
        if self.curriculum_level >= 2:
            return random_unit_vector(self.rng)

        # Le cône et l'hémisphère sont construits dans le repère de la cellule,
        # puis orientés vers le monde. x local représente l'avant.
        minimum_x = math.cos(math.radians(25.0)) if self.curriculum_level == 0 else 0.0
        local_x = self.rng.uniform(minimum_x, 1.0)
        radius = math.sqrt(max(0.0, 1.0 - local_x * local_x))
        azimuth = self.rng.uniform(0.0, 2.0 * math.pi)
        local = Vec3(local_x, radius * math.cos(azimuth), radius * math.sin(azimuth))
        return self.cell.orientation.rotate(local).normalized()

    def _inside_target_bounds(self, point: Vec3) -> bool:
        limit = WORLD_HALF_SIZE - WORLD_MARGIN - CAPTURE_RADIUS
        return all(abs(value) <= limit for value in (point.x, point.y, point.z))

    def _sample_target(self) -> Vec3:
        for _attempt in range(500):
            distance = self.rng.uniform(MIN_TARGET_DISTANCE, MAX_TARGET_DISTANCE)
            candidate = self.cell.position + self._curriculum_direction() * distance
            if self._inside_target_bounds(candidate):
                return candidate

        # Près d'une paroi, le cône demandé peut ne contenir aucun point valide.
        # Ce repli uniforme garantit toujours une cible atteignable.
        limit = WORLD_HALF_SIZE - WORLD_MARGIN - CAPTURE_RADIUS
        for _attempt in range(500):
            candidate = Vec3(
                self.rng.uniform(-limit, limit),
                self.rng.uniform(-limit, limit),
                self.rng.uniform(-limit, limit),
            )
            if (candidate - self.cell.position).length() >= MIN_TARGET_DISTANCE:
                return candidate
        return Vec3()

    def set_curriculum_level(self, level: int) -> None:
        self.curriculum_level = int(clamp(level, 0, 2))

    def observations(self) -> list[float]:
        relative_target = self.cell.orientation.inverse_rotate(
            self.target - self.cell.position
        )
        local_velocity = self.cell.orientation.inverse_rotate(self.cell.velocity)
        local_rotation = self.cell.orientation.inverse_rotate(
            self.cell.angular_velocity
        )
        values = [
            relative_target.x / WORLD_HALF_SIZE,
            relative_target.y / WORLD_HALF_SIZE,
            relative_target.z / WORLD_HALF_SIZE,
            local_velocity.x / MAX_CELL_SPEED,
            local_velocity.y / MAX_CELL_SPEED,
            local_velocity.z / MAX_CELL_SPEED,
            local_rotation.x / MAX_CELL_ROTATION,
            local_rotation.y / MAX_CELL_ROTATION,
            local_rotation.z / MAX_CELL_ROTATION,
        ]
        for index, angles in enumerate(self.cell.joint_angles):
            limit = JOINT_LIMITS[index]
            values.extend((angles[0] / limit, angles[1] / limit))
        for speeds in self.cell.joint_speeds:
            values.extend(
                (speeds[0] / MAX_JOINT_SPEED, speeds[1] / MAX_JOINT_SPEED)
            )
        values.extend((math.sin(self.cell.gait_phase), math.cos(self.cell.gait_phase)))
        if len(values) != OBSERVATION_SIZE:
            raise RuntimeError("Dimension d'observation 3D incohérente.")
        return values

    def flagellum_geometry(self) -> FlagellumGeometry:
        forward = self.body_forward()
        attachment = self.cell.position - forward * BODY_HALF_LENGTH
        points = [attachment]
        middles: list[Vec3] = []
        tangents: list[Vec3] = []
        frame = self.cell.orientation

        for segment_index, (yaw, pitch) in enumerate(self.cell.joint_angles):
            yaw_rotation = Quaternion.from_axis_angle(Vec3(0.0, 0.0, 1.0), yaw)
            pitch_rotation = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), pitch)
            frame = (frame * yaw_rotation * pitch_rotation).normalized()
            tangent = frame.rotate(Vec3(-1.0, 0.0, 0.0)).normalized()
            start = points[-1]
            length = SEGMENT_LENGTHS[segment_index]
            middles.append(start + tangent * (0.5 * length))
            points.append(start + tangent * length)
            tangents.append(tangent)
        return FlagellumGeometry(points, middles, tangents)

    @staticmethod
    def segment_drag(
        velocity: Vec3, tangent: Vec3, force_weight: float
    ) -> Vec3:
        """Force visqueuse anisotrope pondérée par la morphologie du segment."""
        parallel = tangent * velocity.dot(tangent)
        perpendicular_velocity = velocity - parallel
        return (
            parallel * (-DRAG_PARALLEL)
            + perpendicular_velocity * (-DRAG_PERPENDICULAR)
        ) * force_weight

    def _apply_boundaries(self) -> tuple[bool, Vec3]:
        hit = False
        reaction = Vec3()
        limit = WORLD_HALF_SIZE - WORLD_MARGIN - BODY_HALF_LENGTH
        coordinates = ("x", "y", "z")
        for name in coordinates:
            value = getattr(self.cell.position, name)
            velocity = getattr(self.cell.velocity, name)
            if value < -limit:
                setattr(self.cell.position, name, -limit)
                setattr(self.cell.velocity, name, abs(velocity) * 0.15)
                setattr(reaction, name, -limit - value + 0.5)
                hit = True
            elif value > limit:
                setattr(self.cell.position, name, limit)
                setattr(self.cell.velocity, name, -abs(velocity) * 0.15)
                setattr(reaction, name, limit - value - 0.5)
                hit = True
        return hit, reaction

    def _advance_target(self) -> None:
        self.target = self._sample_target()
        self.previous_distance = self.distance_to_target()
        self.initial_distance = max(self.previous_distance, 1.0)
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_path_length = 0.0
        self.episode_energy = 0.0
        self.episode_wall_hits = 0
        self.episode_lateral_motion = 0.0

    def step(self, action: list[float], remember_trajectory: bool) -> StepOutcome:
        if len(action) != ACTION_SIZE:
            raise ValueError("Une action 3D doit contenir six commandes.")
        bounded_action = [clamp(float(value), -1.0, 1.0) for value in action]
        old_geometry = self.flagellum_geometry()
        self.cell.gait_phase = (
            self.cell.gait_phase + 2.0 * math.pi / GAIT_PERIOD
        ) % (2.0 * math.pi)

        for segment_index in range(SEGMENT_COUNT):
            limit = JOINT_LIMITS[segment_index]
            for axis_index in range(2):
                action_index = 2 * segment_index + axis_index
                previous_angle = self.cell.joint_angles[segment_index][axis_index]
                commanded_speed = (
                    self.cell.joint_speeds[segment_index][axis_index]
                    * JOINT_DAMPING
                    + bounded_action[action_index] * MOTOR_ACCELERATION
                )
                commanded_speed = clamp(
                    commanded_speed, -MAX_JOINT_SPEED, MAX_JOINT_SPEED
                )
                angle = clamp(
                    previous_angle + commanded_speed, -limit, limit
                )

                # La vitesse mémorisée est le déplacement réellement effectué.
                # À une butée, une commande dirigée vers l'extérieur donne donc
                # une vitesse nulle au lieu d'une vitesse fictive.
                actual_speed = angle - previous_angle
                self.cell.joint_speeds[segment_index][axis_index] = actual_speed
                self.cell.joint_angles[segment_index][axis_index] = angle

        geometry = self.flagellum_geometry()
        segment_forces: list[Vec3] = []
        for old_middle, new_middle, tangent, force_weight in zip(
            old_geometry.middles,
            geometry.middles,
            geometry.tangents,
            SEGMENT_FORCE_WEIGHTS,
        ):
            deformation_velocity = new_middle - old_middle
            segment_forces.append(
                self.segment_drag(deformation_velocity, tangent, force_weight)
            )

        # Aire orientée parcourue par deux courbures successives. Cette petite
        # composante axiale stabilise la propulsion produite par l'onde.
        wave_rate = 0.0
        for index in range(SEGMENT_COUNT - 1):
            angle_a = self.cell.joint_angles[index]
            angle_b = self.cell.joint_angles[index + 1]
            speed_a = self.cell.joint_speeds[index]
            speed_b = self.cell.joint_speeds[index + 1]
            wave_rate += angle_a[0] * speed_b[0] - angle_b[0] * speed_a[0]
            wave_rate += angle_a[1] * speed_b[1] - angle_b[1] * speed_a[1]

        shape_force = self.body_forward() * (SHAPE_PROPULSION * wave_rate)
        total_force_weight = sum(SEGMENT_FORCE_WEIGHTS)
        for index in range(SEGMENT_COUNT):
            # La composante produite par l'onde est elle aussi moins forte à
            # l'extrémité, mais sa somme reste identique à l'ancienne version.
            shape_share = SEGMENT_FORCE_WEIGHTS[index] / total_force_weight
            segment_forces[index] = segment_forces[index] + shape_force * shape_share

        total_force = Vec3()
        torque = Vec3()
        for middle, force in zip(geometry.middles, segment_forces):
            total_force = total_force + force
            torque = torque + (middle - self.cell.position).cross(force)

        desired_velocity = (total_force * FORCE_TO_SPEED).limited(MAX_CELL_SPEED)
        self.cell.velocity = (
            self.cell.velocity * LINEAR_DAMPING
            + desired_velocity * (1.0 - LINEAR_DAMPING)
        ).limited(MAX_CELL_SPEED)
        desired_rotation = (torque * TORQUE_TO_ROTATION).limited(MAX_CELL_ROTATION)
        self.cell.angular_velocity = (
            self.cell.angular_velocity * ANGULAR_DAMPING
            + desired_rotation * (1.0 - ANGULAR_DAMPING)
        ).limited(MAX_CELL_ROTATION)

        rotation_increment = Quaternion.from_rotation_vector(
            self.cell.angular_velocity
        )
        self.cell.orientation = (
            rotation_increment * self.cell.orientation
        ).normalized()

        old_position = Vec3(
            self.cell.position.x, self.cell.position.y, self.cell.position.z
        )
        self.cell.position = self.cell.position + self.cell.velocity
        wall_hit, wall_reaction = self._apply_boundaries()
        displacement = (self.cell.position - old_position).length()

        self.episode_path_length += displacement
        energy = sum(value * value for value in bounded_action) / ACTION_SIZE
        self.episode_energy += energy
        self.episode_step += 1
        self.total_steps += 1
        if wall_hit:
            self.episode_wall_hits += 1

        current_distance = self.distance_to_target()
        target_direction = (self.target - self.cell.position).normalized()
        useful_speed = self.cell.velocity.dot(target_direction)
        lateral_vector = self.cell.velocity - target_direction * useful_speed
        lateral_speed = lateral_vector.length()
        self.episode_lateral_motion += lateral_speed
        normalized_progress = (
            self.previous_distance - current_distance
        ) / max(self.initial_distance, 1.0)

        reward = PROGRESS_REWARD * normalized_progress
        reward -= TIME_PENALTY
        reward -= ENERGY_PENALTY * energy
        reward -= LATERAL_PENALTY * lateral_speed
        if (
            self.episode_step >= STAGNATION_GRACE_STEPS
            and current_distance > CAPTURE_RADIUS
            and self.cell.velocity.length() < STAGNATION_SPEED
        ):
            reward -= STAGNATION_PENALTY
        if wall_hit:
            reward -= WALL_PENALTY

        captured = current_distance <= CAPTURE_RADIUS
        timeout_limit = int(TIMEOUT_BASE + TIMEOUT_PER_UNIT * self.initial_distance)
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
                lateral_motion=self.episode_lateral_motion,
            )
            self._advance_target()

        if remember_trajectory:
            self.trajectory.append(
                Vec3(self.cell.position.x, self.cell.position.y, self.cell.position.z)
            )
            if len(self.trajectory) > 700:
                del self.trajectory[:120]

        self.telemetry = Telemetry(
            segment_forces=segment_forces,
            propulsion=total_force * FORCE_TO_SPEED,
            velocity=self.cell.velocity,
            target_direction=(self.target - self.cell.position).normalized(),
            wall_reaction=wall_reaction,
            torque=torque,
            reward=reward,
            distance_to_target=self.distance_to_target(),
            useful_speed=useful_speed,
            lateral_speed=lateral_speed,
            alignment=self.body_forward().dot(target_direction),
            action=bounded_action,
            captured=captured,
            timed_out=timed_out and not captured,
        )
        return StepOutcome(reward, terminal, metrics)

    def to_dict(self) -> dict[str, Any]:
        def vector(value: Vec3) -> list[float]:
            return [value.x, value.y, value.z]

        return {
            "seed": self.seed,
            "rng_state": repr(self.rng.getstate()),
            "curriculum_level": self.curriculum_level,
            "target": vector(self.target),
            "cell": {
                "position": vector(self.cell.position),
                "orientation": asdict(self.cell.orientation),
                "velocity": vector(self.cell.velocity),
                "angular_velocity": vector(self.cell.angular_velocity),
                "joint_angles": self.cell.joint_angles,
                "joint_speeds": self.cell.joint_speeds,
                "gait_phase": self.cell.gait_phase,
            },
            "previous_distance": self.previous_distance,
            "initial_distance": self.initial_distance,
            "episode_step": self.episode_step,
            "episode_reward": self.episode_reward,
            "episode_path_length": self.episode_path_length,
            "episode_energy": self.episode_energy,
            "episode_wall_hits": self.episode_wall_hits,
            "episode_lateral_motion": self.episode_lateral_motion,
            "total_steps": self.total_steps,
            "total_captures": self.total_captures,
            "total_timeouts": self.total_timeouts,
            "trajectory": [vector(item) for item in self.trajectory],
            "telemetry": {
                "segment_forces": [vector(item) for item in self.telemetry.segment_forces],
                "propulsion": vector(self.telemetry.propulsion),
                "velocity": vector(self.telemetry.velocity),
                "target_direction": vector(self.telemetry.target_direction),
                "wall_reaction": vector(self.telemetry.wall_reaction),
                "torque": vector(self.telemetry.torque),
                "reward": self.telemetry.reward,
                "distance_to_target": self.telemetry.distance_to_target,
                "useful_speed": self.telemetry.useful_speed,
                "lateral_speed": self.telemetry.lateral_speed,
                "alignment": self.telemetry.alignment,
                "action": self.telemetry.action,
                "captured": self.telemetry.captured,
                "timed_out": self.telemetry.timed_out,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CellWorld3D":
        def vector(values: Any) -> Vec3:
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError("Vecteur 3D invalide dans la sauvegarde.")
            return Vec3(float(values[0]), float(values[1]), float(values[2]))

        world = cls(int(data["seed"]), int(data["curriculum_level"]))
        world.rng.setstate(ast.literal_eval(str(data["rng_state"])))
        world.target = vector(data["target"])
        cell = data["cell"]
        orientation = cell["orientation"]
        world.cell = CellState(
            position=vector(cell["position"]),
            orientation=Quaternion(
                float(orientation["w"]),
                float(orientation["x"]),
                float(orientation["y"]),
                float(orientation["z"]),
            ).normalized(),
            velocity=vector(cell["velocity"]),
            angular_velocity=vector(cell["angular_velocity"]),
            joint_angles=[[float(v) for v in pair] for pair in cell["joint_angles"]],
            joint_speeds=[[float(v) for v in pair] for pair in cell["joint_speeds"]],
            gait_phase=float(cell["gait_phase"]),
        )
        for name in (
            "previous_distance",
            "initial_distance",
            "episode_reward",
            "episode_path_length",
            "episode_energy",
            "episode_lateral_motion",
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
        telemetry = data["telemetry"]
        world.telemetry = Telemetry(
            segment_forces=[vector(item) for item in telemetry["segment_forces"]],
            propulsion=vector(telemetry["propulsion"]),
            velocity=vector(telemetry["velocity"]),
            target_direction=vector(telemetry["target_direction"]),
            wall_reaction=vector(telemetry["wall_reaction"]),
            torque=vector(telemetry["torque"]),
            reward=float(telemetry["reward"]),
            distance_to_target=float(telemetry["distance_to_target"]),
            useful_speed=float(telemetry["useful_speed"]),
            lateral_speed=float(telemetry["lateral_speed"]),
            alignment=float(telemetry["alignment"]),
            action=[float(value) for value in telemetry["action"]],
            captured=bool(telemetry["captured"]),
            timed_out=bool(telemetry["timed_out"]),
        )
        if len(world.cell.joint_angles) != SEGMENT_COUNT:
            raise ValueError("Nombre d'articulations incompatible.")
        return world


# ---------------------------------------------------------------------------
# Réseau Actor-Critic continu et PPO
# ---------------------------------------------------------------------------


if TORCH_AVAILABLE:

    class ContinuousActorCritic(nn.Module):  # type: ignore[misc]
        """Tronc partagé 23 -> 128 -> 128, Actor gaussien et Critic V(s)."""

        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE),
                nn.Tanh(),
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                nn.Tanh(),
            )
            self.actor_mean = nn.Linear(HIDDEN_SIZE, ACTION_SIZE)
            self.actor_log_std = nn.Parameter(torch.full((ACTION_SIZE,), -0.45))
            self.critic = nn.Linear(HIDDEN_SIZE, 1)
            self._initialize()

        def _initialize(self) -> None:
            for layer in self.shared:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            nn.init.zeros_(self.actor_mean.bias)
            nn.init.orthogonal_(self.critic.weight, gain=1.0)
            nn.init.zeros_(self.critic.bias)

        def forward(self, observations: Any) -> tuple[Any, Any, Any]:
            hidden = self.shared(observations)
            mean = self.actor_mean(hidden)
            log_std = self.actor_log_std.clamp(-3.0, 1.0).expand_as(mean)
            value = self.critic(hidden).squeeze(-1)
            return mean, log_std, value

else:

    class ContinuousActorCritic:  # type: ignore[no-redef]
        def __init__(self) -> None:
            raise RuntimeError("PyTorch est requis : py -m pip install torch")


@dataclass
class PPOHistory:
    rewards: list[float] = field(default_factory=list)
    successes: list[float] = field(default_factory=list)
    path_efficiencies: list[float] = field(default_factory=list)
    effective_speeds: list[float] = field(default_factory=list)
    lateral_motions: list[float] = field(default_factory=list)
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


class PPOTrainer3D:
    def __init__(self, seed: int = 1234, device_name: str = "cpu") -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch n'est pas installé.")
        self.seed = seed
        self.device_name = device_name
        self.device = torch.device(device_name)
        random.seed(seed)
        torch.manual_seed(seed)
        self.model = ContinuousActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.action_generator = torch.Generator(device=self.device)
        self.action_generator.manual_seed(seed + 100)
        self.shuffle_generator = torch.Generator(device="cpu")
        self.shuffle_generator.manual_seed(seed + 200)
        self.worlds = [
            CellWorld3D(seed + 1000 + index * 37, curriculum_level=0)
            for index in range(ENVIRONMENT_COUNT)
        ]
        self.history = PPOHistory()
        self.updates = 0
        self.total_environment_steps = 0
        self.episodes = 0
        self.captures = 0
        self.last_episode: EpisodeMetrics | None = None
        self.last_update = PPOUpdateMetrics()

    def curriculum_level(self) -> int:
        if self.updates < 250:
            return 0
        if self.updates < 800:
            return 1
        return 2

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
    def distribution(mean: Any, log_std: Any) -> Any:
        return Normal(mean, log_std.exp())

    @staticmethod
    def sample_raw_action(mean: Any, log_std: Any, generator: Any) -> Any:
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        return mean + log_std.exp() * noise

    def infer(
        self, observations: list[float], generator: Any, deterministic: bool
    ) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(
                [observations], dtype=torch.float32, device=self.device
            )
            mean, log_std, _value = self.model(tensor)
            raw = mean if deterministic else self.sample_raw_action(
                mean, log_std, generator
            )
            action = torch.tanh(raw)
        return [float(value) for value in action[0].tolist()]

    def _record_episode(self, metrics: EpisodeMetrics) -> None:
        self.episodes += 1
        if metrics.captured:
            self.captures += 1
        self.last_episode = metrics
        self.history.rewards.append(metrics.reward)
        self.history.successes.append(1.0 if metrics.captured else 0.0)
        self.history.path_efficiencies.append(metrics.path_efficiency)
        self.history.effective_speeds.append(metrics.effective_speed)
        self.history.lateral_motions.append(metrics.lateral_motion)

    def success_rate(self, window: int = 20) -> float:
        values = self.history.successes[-window:]
        return sum(values) / len(values) if values else 0.0

    def ppo_update(self) -> PPOUpdateMetrics:
        self.model.train()
        level = self.curriculum_level()
        for world in self.worlds:
            world.set_curriculum_level(level)

        observations: list[Any] = []
        raw_actions: list[Any] = []
        old_log_probabilities: list[Any] = []
        rewards: list[Any] = []
        dones: list[Any] = []
        values: list[Any] = []

        current_observations = self.observations_tensor()
        for _step in range(ROLLOUT_STEPS):
            with torch.no_grad():
                mean, log_std, value = self.model(current_observations)
                raw_action = self.sample_raw_action(
                    mean, log_std, self.action_generator
                )
                distribution = self.distribution(mean, log_std)
                log_probability = distribution.log_prob(raw_action).sum(dim=-1)
                bounded_action = torch.tanh(raw_action)

            step_rewards: list[float] = []
            step_dones: list[float] = []
            for world, action in zip(self.worlds, bounded_action.tolist()):
                outcome = world.step(action, remember_trajectory=False)
                step_rewards.append(outcome.reward)
                step_dones.append(1.0 if outcome.terminal else 0.0)
                if outcome.metrics is not None:
                    self._record_episode(outcome.metrics)

            observations.append(current_observations)
            raw_actions.append(raw_action)
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
            _mean, _log_std, next_value = self.model(current_observations)

        observation_tensor = torch.stack(observations)
        raw_action_tensor = torch.stack(raw_actions)
        old_log_probability_tensor = torch.stack(old_log_probabilities)
        reward_tensor = torch.stack(rewards)
        done_tensor = torch.stack(dones)
        value_tensor = torch.stack(values)

        advantages = torch.zeros_like(reward_tensor)
        gae = torch.zeros(ENVIRONMENT_COUNT, device=self.device)
        bootstrap = next_value
        for step_index in reversed(range(ROLLOUT_STEPS)):
            not_terminal = 1.0 - done_tensor[step_index]
            delta = (
                reward_tensor[step_index]
                + GAMMA * bootstrap * not_terminal
                - value_tensor[step_index]
            )
            gae = delta + GAMMA * GAE_LAMBDA * not_terminal * gae
            advantages[step_index] = gae
            bootstrap = value_tensor[step_index]
        returns = advantages + value_tensor

        batch_observations = observation_tensor.reshape(-1, OBSERVATION_SIZE)
        batch_raw_actions = raw_action_tensor.reshape(-1, ACTION_SIZE)
        batch_old_log_probabilities = old_log_probability_tensor.reshape(-1)
        batch_old_values = value_tensor.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_advantages = (
            batch_advantages - batch_advantages.mean()
        ) / (batch_advantages.std(unbiased=False) + 1e-8)

        batch_size = batch_observations.shape[0]
        entropy_coefficient = self.entropy_coefficient()
        sums = {key: 0.0 for key in ("policy", "value", "entropy", "kl", "clip", "gradient")}
        minibatch_count = 0
        stop_early = False

        for _epoch in range(PPO_EPOCHS):
            permutation = torch.randperm(
                batch_size, generator=self.shuffle_generator
            ).to(self.device)
            for start in range(0, batch_size, MINIBATCH_SIZE):
                indices = permutation[start : start + MINIBATCH_SIZE]
                mean, log_std, new_values = self.model(batch_observations[indices])
                distribution = self.distribution(mean, log_std)
                new_log_probabilities = distribution.log_prob(
                    batch_raw_actions[indices]
                ).sum(dim=-1)
                entropy = distribution.entropy().sum(dim=-1).mean()
                log_ratio = new_log_probabilities - batch_old_log_probabilities[indices]
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
                value_loss = 0.5 * torch.maximum(
                    (new_values - batch_returns[indices]).pow(2),
                    (clipped_values - batch_returns[indices]).pow(2),
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
            raise ValueError("Cette sauvegarde n'est pas une session 3D compatible.")
        self.model.load_state_dict(data["model_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        torch.set_rng_state(data["torch_rng_state"].cpu())
        self.action_generator.set_state(data["action_generator_state"])
        self.shuffle_generator.set_state(data["shuffle_generator_state"])
        self.worlds = [CellWorld3D.from_dict(item) for item in data["worlds"]]
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


class Evaluator3D:
    def __init__(self, trainer: PPOTrainer3D) -> None:
        self.trainer = trainer
        self.generator = torch.Generator(device=trainer.device)
        self.reset()

    def reset(self) -> None:
        self.world = CellWorld3D(seed=6060, curriculum_level=2)
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
        return sum(item.captured for item in self.episodes)

    def success_rate(self) -> float:
        return self.captures / len(self.episodes) if self.episodes else 0.0

    def average(self, attribute: str) -> float:
        return (
            sum(float(getattr(item, attribute)) for item in self.episodes)
            / len(self.episodes)
            if self.episodes
            else 0.0
        )


# ---------------------------------------------------------------------------
# Interface Tkinter et projection du cube
# ---------------------------------------------------------------------------


class Application:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.root = tk.Tk()
        self.root.title("Cellule 3D — PPO continu — flagelle à 3 segments")
        self.root.geometry(f"{CANVAS_WIDTH}x{CANVAS_HEIGHT + 56}")
        self.root.minsize(CANVAS_WIDTH, CANVAS_HEIGHT + 56)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.trainer = PPOTrainer3D(seed=1234)
        self.evaluator = Evaluator3D(self.trainer)
        self.demo_world = CellWorld3D(seed=9090, curriculum_level=0)
        self.demo_generator = torch.Generator(device=self.trainer.device)
        self.demo_generator.manual_seed(9091)
        self.learning_running = False
        self.evaluation_running = False
        self.evaluation_deterministic = False
        self.training_busy = False
        self.closed = False
        self.show_vectors = True
        self.capture_flash = 0
        self.view_index = 0
        self.views = ((-35.0, 24.0), (35.0, 24.0), (-125.0, 22.0), (-35.0, 52.0))
        self.camera_yaw, self.camera_pitch = self.views[self.view_index]
        self.camera_zoom = 1.0
        self.camera_drag_origin: tuple[int, int] | None = None
        self.camera_drag_angles: tuple[float, float] | None = None
        # L'entraînement PPO s'exécute sur le thread Tkinter. Une pause entre
        # deux mises à jour laisse plusieurs images à la démonstration visible.
        self.cadences = (1, 350, 1000)
        self.cadence_names = ("MAX", "NORMALE", "LENTE")
        self.cadence_index = 1

        toolbar = tk.Frame(self.root, bg="#18222f", padx=6, pady=7)
        toolbar.pack(fill="x")
        self.start_button = tk.Button(
            toolbar, text="Démarrer PPO", width=16, command=self.toggle_learning
        )
        self.start_button.pack(side="left", padx=2)
        self.update_button = tk.Button(
            toolbar, text="1 mise à jour", width=15, command=self.run_one_update
        )
        self.update_button.pack(side="left", padx=2)
        self.evaluation_button = tk.Button(
            toolbar, text="Évaluer RN", width=14, command=self.start_evaluation
        )
        self.evaluation_button.pack(side="left", padx=2)
        self.evaluation_mode_button = tk.Button(
            toolbar,
            text="Test : stochastique",
            width=19,
            command=self.toggle_evaluation_mode,
        )
        self.evaluation_mode_button.pack(side="left", padx=2)
        self.cadence_button = tk.Button(
            toolbar,
            text="Cadence : NORMALE",
            width=18,
            command=self.cycle_cadence,
        )
        self.cadence_button.pack(side="left", padx=2)
        tk.Button(
            toolbar, text="Vue prédéfinie", width=15, command=self.change_view
        ).pack(side="left", padx=2)
        self.vector_button = tk.Button(
            toolbar, text="Masquer vecteurs", width=16, command=self.toggle_vectors
        )
        self.vector_button.pack(side="left", padx=2)
        tk.Button(
            toolbar, text="Sauver session", width=15, command=self.save_session
        ).pack(side="left", padx=2)
        tk.Button(
            toolbar, text="Restaurer", width=13, command=self.load_session
        ).pack(side="left", padx=2)
        tk.Button(
            toolbar, text="Réinitialiser", width=13, command=self.reset_network
        ).pack(side="right", padx=2)

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="#07111b",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        # Les interactions sont limitées à la partie gauche réservée au cube.
        # Elles changent uniquement la caméra, jamais l'état physique du monde.
        self.canvas.bind("<ButtonPress-1>", self.begin_camera_drag)
        self.canvas.bind("<B1-Motion>", self.drag_camera)
        self.canvas.bind("<ButtonRelease-1>", self.end_camera_drag)
        self.canvas.bind("<MouseWheel>", self.zoom_camera)
        self.canvas.bind("<Button-4>", self.zoom_camera)
        self.canvas.bind("<Button-5>", self.zoom_camera)
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
        self.update_button.configure(
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
        self.update_button.configure(state="disabled")

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

    def change_view(self) -> None:
        self.view_index = (self.view_index + 1) % len(self.views)
        self.camera_yaw, self.camera_pitch = self.views[self.view_index]

    @staticmethod
    def pointer_in_3d_view(event: Any) -> bool:
        return 0 <= int(event.x) < 885 and 0 <= int(event.y) <= CANVAS_HEIGHT

    def begin_camera_drag(self, event: Any) -> None:
        if not self.pointer_in_3d_view(event):
            return
        self.camera_drag_origin = (int(event.x), int(event.y))
        self.camera_drag_angles = (self.camera_yaw, self.camera_pitch)
        self.canvas.configure(cursor="fleur")

    def drag_camera(self, event: Any) -> None:
        if self.camera_drag_origin is None or self.camera_drag_angles is None:
            return
        delta_x = int(event.x) - self.camera_drag_origin[0]
        delta_y = int(event.y) - self.camera_drag_origin[1]
        initial_yaw, initial_pitch = self.camera_drag_angles
        self.camera_yaw = (initial_yaw + 0.35 * delta_x) % 360.0
        # La limite évite de retourner la caméra et conserve les axes lisibles.
        self.camera_pitch = clamp(initial_pitch - 0.30 * delta_y, -78.0, 78.0)

    def end_camera_drag(self, _event: Any) -> None:
        self.camera_drag_origin = None
        self.camera_drag_angles = None
        self.canvas.configure(cursor="")

    def zoom_camera(self, event: Any) -> str | None:
        if not self.pointer_in_3d_view(event):
            return None
        # Windows/macOS transmettent delta ; Linux utilise Button-4/Button-5.
        zoom_in = getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
        factor = 1.10 if zoom_in else 1.0 / 1.10
        self.camera_zoom = clamp(self.camera_zoom * factor, 1.00, 3.00)
        return "break"

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
            self.update_button.configure(state="normal")
            self.messagebox.showerror("Erreur PPO 3D", str(exc))
        finally:
            self.training_busy = False

    def training_cycle(self) -> None:
        if self.closed:
            return
        if self.learning_running and not self.training_busy:
            self._perform_update()
        self.root.after(self.cadences[self.cadence_index], self.training_cycle)

    @staticmethod
    def session_root() -> Path:
        return Path(__file__).resolve().parent / "sauvegardes_pytorch_3d"

    def _history_rows(self) -> list[list[Any]]:
        history = self.trainer.history
        length = max(len(history.rewards), len(history.policy_losses), 1)
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
                    value(history.lateral_motions),
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
            checkpoint["evaluator_generator_state"] = self.evaluator.generator.get_state()
            checkpoint["application"] = {
                "cadence_index": self.cadence_index,
                "show_vectors": self.show_vectors,
                "evaluation_deterministic": self.evaluation_deterministic,
                "view_index": self.view_index,
                "camera_yaw": self.camera_yaw,
                "camera_pitch": self.camera_pitch,
                "camera_zoom": self.camera_zoom,
            }
            torch.save(checkpoint, directory / SESSION_FILENAME)
            configuration = {
                "format": SESSION_FORMAT,
                "saved_at": checkpoint["saved_at"],
                "architecture": [OBSERVATION_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, ACTION_SIZE, 1],
                "actor": "gaussien continu avec tanh",
                "algorithm": "PPO + GAE",
                "flagellum_segments": SEGMENT_COUNT,
                "segment_lengths": list(SEGMENT_LENGTHS),
                "segment_force_weights": list(SEGMENT_FORCE_WEIGHTS),
                "joint_limits_degrees": [
                    math.degrees(value) for value in JOINT_LIMITS
                ],
                "shape_propulsion": SHAPE_PROPULSION,
                "stagnation_speed": STAGNATION_SPEED,
                "stagnation_grace_steps": STAGNATION_GRACE_STEPS,
                "stagnation_penalty": STAGNATION_PENALTY,
                "joint_axes": 2,
                "environment_count": ENVIRONMENT_COUNT,
                "rollout_steps": ROLLOUT_STEPS,
                "ppo_epochs": PPO_EPOCHS,
                "learning_rate": LEARNING_RATE,
            }
            (directory / CONFIG_FILENAME).write_text(
                json.dumps(configuration, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            with (directory / HISTORY_FILENAME).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, delimiter=";")
                writer.writerow(
                    [
                        "index", "reward", "success", "path_efficiency",
                        "effective_speed", "lateral_motion", "policy_loss",
                        "value_loss", "entropy", "approximate_kl", "clip_fraction",
                    ]
                )
                writer.writerows(self._history_rows())
        except (OSError, RuntimeError, ValueError) as exc:
            self.messagebox.showerror("Sauvegarde impossible", str(exc))
            return
        self.messagebox.showinfo(
            "Session 3D sauvegardée",
            f"Checkpoint, configuration et historique :\n{directory}",
        )

    def load_session(self) -> None:
        root = self.session_root()
        root.mkdir(parents=True, exist_ok=True)
        selected = self.filedialog.askdirectory(
            title="Sélectionner un dossier de session PPO 3D",
            initialdir=str(root),
            mustexist=True,
        )
        if not selected:
            return
        path = Path(selected) / SESSION_FILENAME
        if not path.is_file():
            self.messagebox.showerror(
                "Dossier invalide", f"Le fichier {SESSION_FILENAME} est absent."
            )
            return
        self.learning_running = False
        self.evaluation_running = False
        try:
            try:
                checkpoint = torch.load(
                    path, map_location=self.trainer.device, weights_only=True
                )
            except TypeError:
                checkpoint = torch.load(path, map_location=self.trainer.device)
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint invalide.")
            self.trainer.restore_checkpoint(checkpoint)
            self.demo_world = CellWorld3D.from_dict(checkpoint["demo_world"])
            self.demo_generator.set_state(checkpoint["demo_generator_state"])
            self.evaluator = Evaluator3D(self.trainer)
            self.evaluator.world = CellWorld3D.from_dict(checkpoint["evaluator_world"])
            self.evaluator.episodes = [
                EpisodeMetrics(**item)
                for item in checkpoint.get("evaluator_episodes", [])
            ]
            self.evaluator.generator.set_state(checkpoint["evaluator_generator_state"])
            application = checkpoint.get("application", {})
            self.cadence_index = int(application.get("cadence_index", 1))
            self.show_vectors = bool(application.get("show_vectors", True))
            self.evaluation_deterministic = bool(
                application.get("evaluation_deterministic", False)
            )
            self.view_index = int(application.get("view_index", 0)) % len(self.views)
            default_yaw, default_pitch = self.views[self.view_index]
            self.camera_yaw = float(application.get("camera_yaw", default_yaw))
            self.camera_pitch = clamp(
                float(application.get("camera_pitch", default_pitch)), -78.0, 78.0
            )
            self.camera_zoom = clamp(
                float(application.get("camera_zoom", 1.0)), 1.00, 3.00
            )
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
            self.messagebox.showerror("Restauration impossible", str(exc))
            return

        self.start_button.configure(text="Démarrer PPO")
        self.update_button.configure(state="normal")
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
            "Session 3D restaurée",
            "Réseau, optimiseur, mondes et historiques restaurés.\n"
            "La session est volontairement en pause.",
        )

    def reset_network(self) -> None:
        if not self.messagebox.askyesno(
            "Réinitialiser PPO 3D",
            "Effacer les poids, l'optimiseur et toutes les courbes ?",
        ):
            return
        self.learning_running = False
        self.evaluation_running = False
        seed = random.randrange(1, 1_000_000)
        self.trainer = PPOTrainer3D(seed=seed)
        self.evaluator = Evaluator3D(self.trainer)
        self.demo_world = CellWorld3D(seed=seed + 2, curriculum_level=0)
        self.demo_generator = torch.Generator(device=self.trainer.device)
        self.demo_generator.manual_seed(seed + 3)
        self.start_button.configure(text="Démarrer PPO")
        self.update_button.configure(state="normal")
        self.evaluation_button.configure(text="Évaluer RN")

    def tick(self) -> None:
        if self.closed:
            return
        if self.learning_running and not self.training_busy:
            self.demo_world.set_curriculum_level(self.trainer.curriculum_level())
            action = self.trainer.infer(
                self.demo_world.observations(), self.demo_generator, False
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

    def visible_world(self) -> CellWorld3D:
        return self.evaluator.world if self.evaluation_running else self.demo_world

    def project(self, point: Vec3) -> tuple[float, float, float]:
        yaw = math.radians(self.camera_yaw)
        pitch = math.radians(self.camera_pitch)
        cosine_yaw, sine_yaw = math.cos(yaw), math.sin(yaw)
        horizontal_x = cosine_yaw * point.x - sine_yaw * point.y
        depth_axis = sine_yaw * point.x + cosine_yaw * point.y
        vertical = math.cos(pitch) * point.z - math.sin(pitch) * depth_axis
        depth = math.sin(pitch) * point.z + math.cos(pitch) * depth_axis
        scale = 0.72 * self.camera_zoom
        return 435.0 + scale * horizontal_x, 372.0 - scale * vertical, depth

    def draw_3d_line(
        self,
        start: Vec3,
        end: Vec3,
        color: str,
        width: int = 1,
        dash: tuple[int, int] | None = None,
        arrow: str | None = None,
    ) -> None:
        x1, y1, _depth1 = self.project(start)
        x2, y2, _depth2 = self.project(end)
        self.canvas.create_line(
            x1, y1, x2, y2, fill=color, width=width, dash=dash, arrow=arrow
        )

    def draw_cube(self) -> None:
        c = self.canvas
        half = WORLD_HALF_SIZE
        vertices = {
            (x, y, z): Vec3(x * half, y * half, z * half)
            for x in (-1, 1)
            for y in (-1, 1)
            for z in (-1, 1)
        }
        for x in (-1, 1):
            for y in (-1, 1):
                self.draw_3d_line(vertices[(x, y, -1)], vertices[(x, y, 1)], "#496476", 2)
        for z in (-1, 1):
            for x in (-1, 1):
                self.draw_3d_line(vertices[(x, -1, z)], vertices[(x, 1, z)], "#496476", 2)
            for y in (-1, 1):
                self.draw_3d_line(vertices[(-1, y, z)], vertices[(1, y, z)], "#496476", 2)

        # Deux plans quadrillés donnent la profondeur sans masquer la cellule.
        for fraction in (-0.5, 0.0, 0.5):
            coordinate = fraction * 2.0 * half
            self.draw_3d_line(Vec3(-half, coordinate, -half), Vec3(half, coordinate, -half), "#173247")
            self.draw_3d_line(Vec3(coordinate, -half, -half), Vec3(coordinate, half, -half), "#173247")
            self.draw_3d_line(Vec3(-half, half, coordinate), Vec3(half, half, coordinate), "#132b3d")
            self.draw_3d_line(Vec3(coordinate, half, -half), Vec3(coordinate, half, half), "#132b3d")

        origin = Vec3(-half, -half, -half)
        axes = ((Vec3(80, 0, 0), "X", "#ff7a7a"), (Vec3(0, 80, 0), "Y", "#76df9b"), (Vec3(0, 0, 80), "Z", "#71b7ff"))
        for vector, label, color in axes:
            endpoint = origin + vector
            self.draw_3d_line(origin, endpoint, color, 2, arrow="last")
            x, y, _depth = self.project(endpoint)
            c.create_text(x + 4, y - 4, text=label, fill=color, anchor="sw", font=("Segoe UI", 9, "bold"))

    def draw_vector(self, origin: Vec3, vector: Vec3, color: str, label: str, scale: float) -> None:
        magnitude = vector.length()
        if magnitude < 1e-9:
            return
        length = min(115.0, max(14.0, magnitude * scale))
        endpoint = origin + vector.normalized() * length
        self.draw_3d_line(origin, endpoint, color, 2, arrow="last")
        x, y, _depth = self.project(endpoint)
        self.canvas.create_text(x + 4, y - 4, text=label, fill=color, anchor="sw", font=("Segoe UI", 8, "bold"))

    def draw_world(self) -> None:
        c = self.canvas
        world = self.visible_world()
        self.draw_cube()

        # Projection verticale sur le plan inférieur : elle rend la coordonnée
        # z perceptible même lorsque la trajectoire passe devant les arêtes.
        for point, color in ((world.cell.position, "#416c78"), (world.target, "#806d39")):
            floor_point = Vec3(point.x, point.y, -WORLD_HALF_SIZE)
            self.draw_3d_line(point, floor_point, color, 1, dash=(4, 4))

        if len(world.trajectory) >= 2:
            projected: list[float] = []
            for point in world.trajectory:
                x, y, _depth = self.project(point)
                projected.extend((x, y))
            c.create_line(*projected, fill="#718792", width=2, smooth=True)

        target_x, target_y, target_depth = self.project(world.target)
        depth_scale = clamp(1.0 + target_depth / 1400.0, 0.72, 1.25)
        capture_pixels = CAPTURE_RADIUS * 0.72 * self.camera_zoom * depth_scale
        target_pixels = TARGET_RADIUS * 0.72 * self.camera_zoom * depth_scale
        c.create_oval(
            target_x - capture_pixels,
            target_y - capture_pixels,
            target_x + capture_pixels,
            target_y + capture_pixels,
            outline="#f7d154",
            dash=(4, 4),
            width=2,
        )
        c.create_oval(
            target_x - target_pixels,
            target_y - target_pixels,
            target_x + target_pixels,
            target_y + target_pixels,
            fill="#ffffff" if self.capture_flash else "#ffd54f",
            outline="#fff4b0",
            width=2,
        )

        geometry = world.flagellum_geometry()
        segment_colors = ("#73d7ff", "#4eb7e5", "#329acb")
        for index in range(SEGMENT_COUNT):
            self.draw_3d_line(
                geometry.points[index],
                geometry.points[index + 1],
                segment_colors[index],
                6 - index,
            )
        for joint in geometry.points[:-1]:
            x, y, depth = self.project(joint)
            radius = clamp(3.5 * (1.0 + depth / 1600.0), 2.5, 5.0)
            c.create_oval(x - radius, y - radius, x + radius, y + radius, fill="#d9f5ff", outline="#22789c")

        forward = world.body_forward()
        right = world.cell.orientation.rotate(Vec3(0.0, 1.0, 0.0)).normalized()
        up = world.cell.orientation.rotate(Vec3(0.0, 0.0, 1.0)).normalized()
        body_points: list[float] = []
        for index in range(28):
            angle = 2.0 * math.pi * index / 28
            point = (
                world.cell.position
                + forward * (BODY_HALF_LENGTH * math.cos(angle))
                + right * (BODY_RADIUS * math.sin(angle))
            )
            x, y, _depth = self.project(point)
            body_points.extend((x, y))
        c.create_polygon(*body_points, fill="#67d8a2", outline="#d8ffea", width=2, smooth=True)
        self.draw_3d_line(
            world.cell.position - forward * BODY_HALF_LENGTH,
            world.cell.position + forward * BODY_HALF_LENGTH,
            "#d8ffea",
            2,
        )
        self.draw_3d_line(
            world.cell.position - up * BODY_RADIUS,
            world.cell.position + up * BODY_RADIUS,
            "#3b8e6a",
            1,
        )

        telemetry = world.telemetry
        if self.show_vectors:
            force_colors = ("#ffad58", "#ff825e", "#ff5f79")
            for index, (middle, force) in enumerate(zip(geometry.middles, telemetry.segment_forces)):
                self.draw_vector(middle, force, force_colors[index], f"F{index + 1}", 480.0)
            self.draw_vector(world.cell.position, telemetry.propulsion, "#46e66d", "PROP", 90.0)
            self.draw_vector(world.cell.position, telemetry.velocity, "#48a9ff", "V", 48.0)
            self.draw_vector(world.cell.position, telemetry.target_direction, "#cf8cff", "CIBLE", 78.0)
            self.draw_vector(world.cell.position, telemetry.torque, "#ffcf67", "τ", 2.0)

        c.create_text(
            24,
            CANVAS_HEIGHT - 16,
            text="Glisser : rotation libre   Molette : zoom   Vue prédéfinie : cadrage   F1/F2/F3 : forces   τ : couple",
            anchor="sw",
            fill="#a9bbc6",
            font=("Segoe UI", 9),
        )

    def draw_statistics(self) -> None:
        c = self.canvas
        world = self.visible_world()
        telemetry = world.telemetry
        left, top, right, bottom = 895, 18, 1215, 720
        c.create_rectangle(left, top, right, bottom, fill="#0c1722", outline="#38566b", width=2)
        if self.learning_running:
            state, color = "APPRENTISSAGE PPO 3D", "#66e58a"
        elif self.evaluation_running:
            mode = "DÉTERMINISTE" if self.evaluation_deterministic else "STOCHASTIQUE"
            state, color = f"ÉVALUATION {mode}", "#62c8ff"
        else:
            state, color = "PAUSE — MONDE FIGÉ", "#ffca5c"
        c.create_text((left + right) * 0.5, top + 23, text="ACTOR-CRITIC CONTINU", fill="#e9f4fa", font=("Segoe UI", 11, "bold"))
        c.create_text((left + right) * 0.5, top + 49, text=state, fill=color, font=("Segoe UI", 9, "bold"))

        update = self.trainer.last_update
        level = self.trainer.curriculum_level()
        ppo_info = (
            f"PPO / GAE — 3D\n"
            f"Curriculum : {level + 1}/3\n"
            f"{CURRICULUM_NAMES[level]}\n"
            f"Mondes     : {ENVIRONMENT_COUNT}\n"
            f"Lot        : {ENVIRONMENT_COUNT * ROLLOUT_STEPS}\n"
            f"Mises à j. : {self.trainer.updates}\n"
            f"Pas simulés: {self.trainer.total_environment_steps}\n"
            f"Poursuites : {self.trainer.episodes}\n"
            f"Captures   : {self.trainer.captures}\n"
            f"Réussite20 : {100*self.trainer.success_rate():5.1f}%\n"
            f"Loss Actor : {update.policy_loss:+.5f}\n"
            f"Loss Critic: {update.value_loss:.5f}\n"
            f"Entropie   : {update.entropy:.4f}\n"
            f"KL approx. : {update.approximate_kl:.5f}"
        )
        c.create_text(left + 15, top + 75, text=ppo_info, anchor="nw", fill="#d8e7ef", font=("Consolas", 8))

        action = telemetry.action
        movement_info = (
            f"CELLULE AFFICHÉE\n"
            f"Position X : {world.cell.position.x:+7.2f}\n"
            f"Position Y : {world.cell.position.y:+7.2f}\n"
            f"Position Z : {world.cell.position.z:+7.2f}\n"
            f"Distance   : {telemetry.distance_to_target:7.2f}\n"
            f"Vitesse    : {world.cell.velocity.length():.4f}\n"
            f"Vers cible : {telemetry.useful_speed:+.4f}\n"
            f"Latérale   : {telemetry.lateral_speed:.4f}\n"
            f"Alignement : {telemetry.alignment:+.3f}\n"
            f"Rotation   : {world.cell.angular_velocity.length():.4f}\n"
            f"Phase      : {math.degrees(world.cell.gait_phase):6.1f}°\n"
            f"Vue        : {self.camera_yaw:5.0f}° {self.camera_pitch:+4.0f}° "
            f"x{self.camera_zoom:.2f}\n"
            f"Commandes  :\n"
            f"  S1 {action[0]:+4.2f} {action[1]:+4.2f}\n"
            f"  S2 {action[2]:+4.2f} {action[3]:+4.2f}\n"
            f"  S3 {action[4]:+4.2f} {action[5]:+4.2f}"
        )
        c.create_text(left + 15, top + 294, text=movement_info, anchor="nw", fill="#c5e3ef", font=("Consolas", 8))

        if self.evaluation_running:
            evaluation_info = (
                f"ÉVALUATION — POIDS FIGÉS\n"
                f"Poursuites : {len(self.evaluator.episodes)}\n"
                f"Réussites  : {self.evaluator.captures}\n"
                f"Taux       : {100*self.evaluator.success_rate():5.1f}%\n"
                f"Pas moyens : {self.evaluator.average('steps'):.1f}\n"
                f"V.eff. moy.: {self.evaluator.average('effective_speed'):.4f}\n"
                f"Eff. trajet: {self.evaluator.average('path_efficiency'):.3f}"
            )
        elif self.trainer.last_episode is None:
            evaluation_info = "DERNIÈRE POURSUITE\nEn attente"
        else:
            last = self.trainer.last_episode
            evaluation_info = (
                f"DERNIÈRE POURSUITE\n"
                f"Résultat   : {'SUCCÈS' if last.captured else 'ÉCHEC'}\n"
                f"Récompense : {last.reward:+.3f}\n"
                f"Pas        : {last.steps}\n"
                f"V.effective: {last.effective_speed:.4f}\n"
                f"Efficacité : {last.path_efficiency:.3f}"
            )
        c.create_text(left + 15, top + 545, text=evaluation_info, anchor="nw", fill="#d9c8f3", font=("Consolas", 8))
        c.create_text((left + right) * 0.5, bottom - 16, text="Évaluation : aucun optimizer.step().", fill="#8fa9b8", font=("Segoe UI", 8))

    def draw(self) -> None:
        self.canvas.delete("all")
        self.draw_world()
        self.draw_statistics()
        history = self.trainer.history
        self.draw_chart(
            1227, 18, 1485, 360, "RÉCOMPENSE",
            (("récompense", history.rewards, "#78909c"), ("moyenne 20", moving_average(history.rewards, 20), "#62df79")),
        )
        self.draw_chart(
            1495, 18, 1750, 360, "PERTES PPO",
            (("Actor", history.policy_losses, "#47b7ff"), ("Critic", history.value_losses, "#ffad4d")),
            include_zero=True,
        )
        self.draw_chart(
            1227, 372, 1485, 720, "EFFICACITÉ / SUCCÈS",
            (("trajet", history.path_efficiencies, "#d58cff"), ("succès 20", moving_average(history.successes, 20), "#ffe066")),
            fixed_minimum=0.0,
            fixed_maximum=1.0,
        )
        normalized_kl = [min(1.0, value / TARGET_KL) for value in history.approximate_kls]
        self.draw_chart(
            1495, 372, 1750, 720, "SANTÉ PPO",
            (("KL/cible", normalized_kl, "#ff7a90"), ("clip", history.clip_fractions, "#a98cff")),
            fixed_minimum=0.0,
            fixed_maximum=1.0,
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
        c.create_rectangle(left, top, right, bottom, fill="#0c1722", outline="#38566b", width=2)
        c.create_text((left + right) * 0.5, top + 17, text=title, fill="#e9f4fa", font=("Segoe UI", 9, "bold"))
        visible = [(name, values[-180:], color) for name, values, color in series]
        all_values = [value for _name, values, _color in visible for value in values]
        plot_left, plot_right = left + 34, right - 10
        plot_top, plot_bottom = top + 46, bottom - 30
        c.create_line(plot_left, plot_top, plot_left, plot_bottom, fill="#607d8b")
        c.create_line(plot_left, plot_bottom, plot_right, plot_bottom, fill="#607d8b")
        if not all_values:
            c.create_text((plot_left + plot_right) * 0.5, (plot_top + plot_bottom) * 0.5, text="En attente de données", fill="#68808d", font=("Segoe UI", 8))
            return
        minimum = fixed_minimum if fixed_minimum is not None else min(all_values)
        maximum = fixed_maximum if fixed_maximum is not None else max(all_values)
        if include_zero:
            minimum = min(minimum, 0.0)
            maximum = max(maximum, 0.0)
        if abs(maximum - minimum) < 1e-12:
            maximum += 0.5
            minimum -= 0.5
        c.create_text(plot_left - 4, plot_top, text=f"{maximum:.3g}", anchor="e", fill="#8fa9b8", font=("Consolas", 7))
        c.create_text(plot_left - 4, plot_bottom, text=f"{minimum:.3g}", anchor="e", fill="#8fa9b8", font=("Consolas", 7))
        legend_x = left + 8
        for name, values, color in visible:
            if len(values) == 1:
                x_values = [plot_left]
            else:
                x_values = [
                    plot_left + index * (plot_right - plot_left) / (len(values) - 1)
                    for index in range(len(values))
                ]
            points: list[float] = []
            for x_value, value in zip(x_values, values):
                y_value = plot_bottom - ((value - minimum) / (maximum - minimum)) * (plot_bottom - plot_top)
                points.extend((x_value, y_value))
            if len(points) >= 4:
                c.create_line(*points, fill=color, width=2, smooth=True)
            elif points:
                c.create_oval(points[0] - 2, points[1] - 2, points[0] + 2, points[1] + 2, fill=color, outline="")
            c.create_text(legend_x, bottom - 14, text=name, anchor="w", fill=color, font=("Segoe UI", 7))
            legend_x += max(58, len(name) * 6 + 12)


# ---------------------------------------------------------------------------
# Tests et exécution
# ---------------------------------------------------------------------------


def test_physics() -> None:
    world = CellWorld3D(seed=123, curriculum_level=2)
    rng = random.Random(456)
    origin = Vec3()
    for _ in range(8000):
        action = [rng.uniform(-1.0, 1.0) for _ in range(ACTION_SIZE)]
        world.step(action, remember_trajectory=False)
    observations = world.observations()
    assert len(observations) == OBSERVATION_SIZE
    assert all(math.isfinite(value) for value in observations)
    assert abs(world.cell.orientation.normalized().w - world.cell.orientation.w) < 1e-9
    for index, angles in enumerate(world.cell.joint_angles):
        assert all(abs(value) <= JOINT_LIMITS[index] + 1e-12 for value in angles)
    geometry = world.flagellum_geometry()
    assert len(geometry.points) == SEGMENT_COUNT + 1
    assert len(geometry.middles) == SEGMENT_COUNT
    for index, expected_length in enumerate(SEGMENT_LENGTHS):
        measured_length = (geometry.points[index + 1] - geometry.points[index]).length()
        assert abs(measured_length - expected_length) < 1e-9
    restored = CellWorld3D.from_dict(world.to_dict())
    assert restored.to_dict() == world.to_dict()

    # Régression : une commande constante finit en butée, mais ne doit plus
    # fabriquer une vitesse articulaire ni une propulsion fictives.
    static_world = CellWorld3D(seed=789, curriculum_level=0)
    static_world._apply_boundaries = lambda: (False, Vec3())  # type: ignore[method-assign]
    for _ in range(300):
        static_world.step([0.8] * ACTION_SIZE, remember_trajectory=False)
    assert all(
        abs(speed) < 1e-12
        for pair in static_world.cell.joint_speeds
        for speed in pair
    )
    assert static_world.telemetry.propulsion.length() < 1e-12
    assert static_world.cell.velocity.length() < 1e-12

    # Une onde déphasée doit en revanche conserver une propulsion nette.
    wave_world = CellWorld3D(seed=790, curriculum_level=0)
    wave_world._apply_boundaries = lambda: (False, Vec3())  # type: ignore[method-assign]
    wave_speeds: list[float] = []
    for step_index in range(500):
        wave_action: list[float] = []
        for segment_index in range(SEGMENT_COUNT):
            phase = (
                2.0 * math.pi * step_index / GAIT_PERIOD
                - segment_index * math.pi / 2.0
            )
            wave_action.extend((math.sin(phase), math.cos(phase)))
        wave_world.step(wave_action, remember_trajectory=False)
        if step_index >= 300:
            wave_speeds.append(wave_world.cell.velocity.length())
    mean_wave_speed = sum(wave_speeds) / len(wave_speeds)
    assert mean_wave_speed > 0.50

    displacement = (world.cell.position - origin).length()
    print(
        "PHYSIQUE 3D OK "
        f"pas={world.total_steps} déplacement={displacement:.2f} "
        f"onde={mean_wave_speed:.2f} "
        f"position=({world.cell.position.x:.1f}, {world.cell.position.y:.1f}, {world.cell.position.z:.1f})"
    )


def run_headless(update_count: int) -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch est requis : py -m pip install torch")
    trainer = PPOTrainer3D(seed=1234)
    for index in range(update_count):
        metrics = trainer.ppo_update()
        print(
            f"MAJ {index + 1:4d} niveau={trainer.curriculum_level() + 1}/3 "
            f"pas={trainer.total_environment_steps:8d} épisodes={trainer.episodes:5d} "
            f"succès20={100*trainer.success_rate():5.1f}% "
            f"actor={metrics.policy_loss:+.5f} critic={metrics.value_loss:.5f} "
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
        help="teste le moteur 3D sans exiger PyTorch",
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
