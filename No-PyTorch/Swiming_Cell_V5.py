"""
Cellule nageuse 2D — Actor-Critic à réseau unique.

Cette expérience utilise un seul réseau de neurones persistant :

    9 observations -> 16 neurones partagés -> Actor (9 actions)
                                           -> Critic (valeur V(s))

Le réseau apprend en ligne après chaque action avec une récompense. Il n'y a
ni population, ni sélection génétique, ni réseau concurrent. L'Actor choisit
les commandes des deux articulations ; le Critic estime la récompense future.

Le fichier n'emploie que la bibliothèque standard Python : tkinter, math,
random et json. Aucun paquet externe n'est nécessaire.

Lancement :
    py cellule_flagelle_apprentissage.py

Test sans interface :
    py cellule_flagelle_apprentissage.py --headless 50000

Les sessions complètes sont créées automatiquement dans :
    sauvegardes/AAAAMMJJ_HHMMSS/session_actor_critic.json

Pour restaurer une session, on sélectionne son dossier horodaté complet et
non le fichier JSON qu'il contient.

La physique reste volontairement pédagogique : elle représente une traînée
visqueuse anisotrope et un flagelle articulé, sans prétendre remplacer une
simulation hydrodynamique complète.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Monde, cellule et flagelle
# ---------------------------------------------------------------------------

WORLD_WIDTH = 820
WORLD_HEIGHT = 680
CANVAS_WIDTH = 1680
CANVAS_HEIGHT = 720
WORLD_MARGIN = 18.0

CELL_RADIUS = 14.0
BODY_HALF_LENGTH = 19.0
BODY_HALF_WIDTH = 12.0
SEGMENT_LENGTH = 36.0

TARGET_RADIUS = 10.0
CAPTURE_RADIUS = 2.0 * TARGET_RADIUS
MIN_TARGET_DISTANCE = 190.0
TARGET_SEQUENCE_LENGTH = 256

# Un dossier horodaté représente une session complète. Le nom et le format
# fixes permettent de refuser un dossier incomplet ou provenant d'un autre
# programme avant de modifier l'état courant de l'application.
SESSION_FILENAME = "session_actor_critic.json"
SESSION_FORMAT = "cellule-actor-critic-session-v2"

# Limites mécaniques distinctes : le premier segment reste derrière le corps
# et le second peut se courber un peu davantage sans repasser devant la cellule.
MAX_BASE_ANGLE = math.radians(40.0)
MAX_INTERSEGMENT_ANGLE = math.radians(55.0)
MAX_JOINT_SPEED = 0.22
MOTOR_ACCELERATION = 0.055
JOINT_DAMPING = 0.88

# Modèle de fluide : résistance latérale supérieure à la résistance axiale.
DRAG_PARALLEL = 0.022
DRAG_PERPENDICULAR = 0.085
FORCE_TO_SPEED = 2.40
TORQUE_TO_ROTATION = 0.0040
SHAPE_PROPULSION = 3.20
MAX_CELL_SPEED = 2.40
MAX_CELL_ROTATION = 0.085

# Une poursuite trop longue est arrêtée pour produire une mesure d'échec.
TIMEOUT_BASE = 400
TIMEOUT_PER_PIXEL = 4.0


# ---------------------------------------------------------------------------
# Récompense et Actor-Critic
# ---------------------------------------------------------------------------

# Rapprochement normalisé par la distance initiale de la poursuite.
PROGRESS_REWARD = 2.0
CAPTURE_REWARD = 3.0
TIME_PENALTY = 0.0010
ENERGY_PENALTY = 0.0004
WALL_PENALTY = 0.040

NETWORK_INPUTS = 9
NETWORK_HIDDEN = 16
NETWORK_ACTIONS = 9

GAMMA = 0.990
LEARNING_RATE = 0.0012
ENTROPY_COEFFICIENT = 0.012
CRITIC_COEFFICIENT = 0.50
GRADIENT_CLIP = 2.0
ADAM_BETA_1 = 0.90
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-8

# Deux commandes discrètes {-1, 0, +1}, donc neuf combinaisons.
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
# Vecteurs 2D sans dépendance scientifique
# ---------------------------------------------------------------------------


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

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        return self.x * other.y - self.y * other.x

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        magnitude = self.length()
        if magnitude < 1e-12:
            return Vec2()
        return self * (1.0 / magnitude)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


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
    if not values:
        return []
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        count = min(index + 1, window)
        result.append(running_sum / count)
    return result


# ---------------------------------------------------------------------------
# Un seul réseau avec tronc partagé, tête Actor et tête Critic
# ---------------------------------------------------------------------------


@dataclass
class LearningUpdate:
    actor_loss: float
    critic_loss: float
    entropy: float
    td_error: float
    value: float


class ActorCriticNetwork:
    """Petit Actor-Critic entraîné par rétropropagation et Adam."""

    def __init__(self, seed: int = 42) -> None:
        rng = random.Random(seed)
        hidden_limit = math.sqrt(6.0 / (NETWORK_INPUTS + NETWORK_HIDDEN))
        actor_limit = math.sqrt(6.0 / (NETWORK_HIDDEN + NETWORK_ACTIONS))
        critic_limit = math.sqrt(6.0 / (NETWORK_HIDDEN + 1))

        self.weights_hidden = [
            [rng.uniform(-hidden_limit, hidden_limit) for _ in range(NETWORK_INPUTS)]
            for _ in range(NETWORK_HIDDEN)
        ]
        self.biases_hidden = [0.0 for _ in range(NETWORK_HIDDEN)]
        self.weights_actor = [
            [rng.uniform(-actor_limit, actor_limit) for _ in range(NETWORK_HIDDEN)]
            for _ in range(NETWORK_ACTIONS)
        ]
        self.biases_actor = [0.0 for _ in range(NETWORK_ACTIONS)]
        self.weights_critic = [
            rng.uniform(-critic_limit, critic_limit) for _ in range(NETWORK_HIDDEN)
        ]
        self.bias_critic = 0.0
        self.reset_optimizer()

    @staticmethod
    def _zeros_matrix(rows: int, columns: int) -> list[list[float]]:
        return [[0.0 for _ in range(columns)] for _ in range(rows)]

    def reset_optimizer(self) -> None:
        """Réinitialise uniquement les moments Adam, pas les poids appris."""

        self.adam_step = 0
        self.m_hidden = self._zeros_matrix(NETWORK_HIDDEN, NETWORK_INPUTS)
        self.v_hidden = self._zeros_matrix(NETWORK_HIDDEN, NETWORK_INPUTS)
        self.m_bias_hidden = [0.0] * NETWORK_HIDDEN
        self.v_bias_hidden = [0.0] * NETWORK_HIDDEN
        self.m_actor = self._zeros_matrix(NETWORK_ACTIONS, NETWORK_HIDDEN)
        self.v_actor = self._zeros_matrix(NETWORK_ACTIONS, NETWORK_HIDDEN)
        self.m_bias_actor = [0.0] * NETWORK_ACTIONS
        self.v_bias_actor = [0.0] * NETWORK_ACTIONS
        self.m_critic = [0.0] * NETWORK_HIDDEN
        self.v_critic = [0.0] * NETWORK_HIDDEN
        self.m_bias_critic = 0.0
        self.v_bias_critic = 0.0

    def forward(self, inputs: list[float]) -> tuple[list[float], list[float], float]:
        hidden: list[float] = []
        for row, bias in zip(self.weights_hidden, self.biases_hidden):
            activation = bias + sum(weight * value for weight, value in zip(row, inputs))
            hidden.append(math.tanh(activation))

        logits = [
            bias + sum(weight * value for weight, value in zip(row, hidden))
            for row, bias in zip(self.weights_actor, self.biases_actor)
        ]
        maximum_logit = max(logits)
        exponentials = [math.exp(value - maximum_logit) for value in logits]
        denominator = sum(exponentials)
        probabilities = [value / denominator for value in exponentials]
        value = self.bias_critic + sum(
            weight * activation
            for weight, activation in zip(self.weights_critic, hidden)
        )
        return hidden, probabilities, value

    def choose_action(
        self, inputs: list[float], rng: random.Random, stochastic: bool
    ) -> int:
        _hidden, probabilities, _value = self.forward(inputs)
        if not stochastic:
            return max(range(NETWORK_ACTIONS), key=lambda index: probabilities[index])

        draw = rng.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if draw <= cumulative:
                return index
        return NETWORK_ACTIONS - 1

    @staticmethod
    def _adam_value(
        value: float,
        gradient: float,
        moment: float,
        velocity: float,
        correction_1: float,
        correction_2: float,
    ) -> tuple[float, float, float]:
        moment = ADAM_BETA_1 * moment + (1.0 - ADAM_BETA_1) * gradient
        velocity = ADAM_BETA_2 * velocity + (1.0 - ADAM_BETA_2) * gradient * gradient
        corrected_moment = moment / correction_1
        corrected_velocity = velocity / correction_2
        value -= LEARNING_RATE * corrected_moment / (
            math.sqrt(corrected_velocity) + ADAM_EPSILON
        )
        return value, moment, velocity

    def learn(
        self,
        inputs: list[float],
        action: int,
        reward: float,
        next_inputs: list[float],
        terminal: bool,
    ) -> LearningUpdate:
        """Effectue une mise à jour Actor-Critic sur une transition."""

        hidden, probabilities, value = self.forward(inputs)
        if terminal:
            next_value = 0.0
        else:
            _next_hidden, _next_probabilities, next_value = self.forward(next_inputs)

        target = reward + (0.0 if terminal else GAMMA * next_value)
        td_error = target - value
        advantage = clamp(td_error, -5.0, 5.0)
        entropy = -sum(
            probability * math.log(max(probability, 1e-12))
            for probability in probabilities
        )
        actor_loss = -advantage * math.log(max(probabilities[action], 1e-12))
        actor_loss -= ENTROPY_COEFFICIENT * entropy
        critic_loss = 0.5 * td_error * td_error

        # Gradient de -A log(pi(a|s)) et du bonus d'entropie.
        sum_p_log_p = sum(
            probability * math.log(max(probability, 1e-12))
            for probability in probabilities
        )
        gradient_logits: list[float] = []
        for index, probability in enumerate(probabilities):
            policy_gradient = advantage * (
                probability - (1.0 if index == action else 0.0)
            )
            entropy_gradient = ENTROPY_COEFFICIENT * probability * (
                math.log(max(probability, 1e-12)) - sum_p_log_p
            )
            gradient_logits.append(policy_gradient + entropy_gradient)

        gradient_value = CRITIC_COEFFICIENT * (value - target)
        gradient_actor = [
            [gradient_logits[action_index] * activation for activation in hidden]
            for action_index in range(NETWORK_ACTIONS)
        ]
        gradient_bias_actor = gradient_logits[:]
        gradient_critic = [gradient_value * activation for activation in hidden]
        gradient_bias_critic = gradient_value

        gradient_hidden: list[float] = []
        for hidden_index in range(NETWORK_HIDDEN):
            actor_contribution = sum(
                self.weights_actor[action_index][hidden_index]
                * gradient_logits[action_index]
                for action_index in range(NETWORK_ACTIONS)
            )
            critic_contribution = self.weights_critic[hidden_index] * gradient_value
            gradient_hidden.append(actor_contribution + critic_contribution)

        gradient_pre_activation = [
            gradient * (1.0 - activation * activation)
            for gradient, activation in zip(gradient_hidden, hidden)
        ]
        gradient_weights_hidden = [
            [gradient * input_value for input_value in inputs]
            for gradient in gradient_pre_activation
        ]
        gradient_bias_hidden = gradient_pre_activation[:]

        all_gradients: list[float] = []
        for matrix in (gradient_weights_hidden, gradient_actor):
            for row in matrix:
                all_gradients.extend(row)
        all_gradients.extend(gradient_bias_hidden)
        all_gradients.extend(gradient_bias_actor)
        all_gradients.extend(gradient_critic)
        all_gradients.append(gradient_bias_critic)
        gradient_norm = math.sqrt(sum(value_ * value_ for value_ in all_gradients))
        gradient_scale = min(1.0, GRADIENT_CLIP / max(gradient_norm, 1e-12))

        self.adam_step += 1
        correction_1 = 1.0 - ADAM_BETA_1**self.adam_step
        correction_2 = 1.0 - ADAM_BETA_2**self.adam_step

        for row_index in range(NETWORK_HIDDEN):
            for column_index in range(NETWORK_INPUTS):
                updated = self._adam_value(
                    self.weights_hidden[row_index][column_index],
                    gradient_weights_hidden[row_index][column_index] * gradient_scale,
                    self.m_hidden[row_index][column_index],
                    self.v_hidden[row_index][column_index],
                    correction_1,
                    correction_2,
                )
                self.weights_hidden[row_index][column_index] = updated[0]
                self.m_hidden[row_index][column_index] = updated[1]
                self.v_hidden[row_index][column_index] = updated[2]

            updated_bias = self._adam_value(
                self.biases_hidden[row_index],
                gradient_bias_hidden[row_index] * gradient_scale,
                self.m_bias_hidden[row_index],
                self.v_bias_hidden[row_index],
                correction_1,
                correction_2,
            )
            self.biases_hidden[row_index] = updated_bias[0]
            self.m_bias_hidden[row_index] = updated_bias[1]
            self.v_bias_hidden[row_index] = updated_bias[2]

        for action_index in range(NETWORK_ACTIONS):
            for hidden_index in range(NETWORK_HIDDEN):
                updated = self._adam_value(
                    self.weights_actor[action_index][hidden_index],
                    gradient_actor[action_index][hidden_index] * gradient_scale,
                    self.m_actor[action_index][hidden_index],
                    self.v_actor[action_index][hidden_index],
                    correction_1,
                    correction_2,
                )
                self.weights_actor[action_index][hidden_index] = updated[0]
                self.m_actor[action_index][hidden_index] = updated[1]
                self.v_actor[action_index][hidden_index] = updated[2]

            updated_bias = self._adam_value(
                self.biases_actor[action_index],
                gradient_bias_actor[action_index] * gradient_scale,
                self.m_bias_actor[action_index],
                self.v_bias_actor[action_index],
                correction_1,
                correction_2,
            )
            self.biases_actor[action_index] = updated_bias[0]
            self.m_bias_actor[action_index] = updated_bias[1]
            self.v_bias_actor[action_index] = updated_bias[2]

        for hidden_index in range(NETWORK_HIDDEN):
            updated = self._adam_value(
                self.weights_critic[hidden_index],
                gradient_critic[hidden_index] * gradient_scale,
                self.m_critic[hidden_index],
                self.v_critic[hidden_index],
                correction_1,
                correction_2,
            )
            self.weights_critic[hidden_index] = updated[0]
            self.m_critic[hidden_index] = updated[1]
            self.v_critic[hidden_index] = updated[2]

        updated_bias = self._adam_value(
            self.bias_critic,
            gradient_bias_critic * gradient_scale,
            self.m_bias_critic,
            self.v_bias_critic,
            correction_1,
            correction_2,
        )
        self.bias_critic = updated_bias[0]
        self.m_bias_critic = updated_bias[1]
        self.v_bias_critic = updated_bias[2]

        return LearningUpdate(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            entropy=entropy,
            td_error=abs(td_error),
            value=value,
        )

    def to_dict(self, include_optimizer: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "architecture": [NETWORK_INPUTS, NETWORK_HIDDEN, NETWORK_ACTIONS, 1],
            "weights_hidden": self.weights_hidden,
            "biases_hidden": self.biases_hidden,
            "weights_actor": self.weights_actor,
            "biases_actor": self.biases_actor,
            "weights_critic": self.weights_critic,
            "bias_critic": self.bias_critic,
        }
        if include_optimizer:
            # Les moments Adam font partie de l'apprentissage. Sans eux, les
            # poids seraient restaurés mais la dynamique de mise à jour changerait.
            data["optimizer"] = {
                "adam_step": self.adam_step,
                "m_hidden": self.m_hidden,
                "v_hidden": self.v_hidden,
                "m_bias_hidden": self.m_bias_hidden,
                "v_bias_hidden": self.v_bias_hidden,
                "m_actor": self.m_actor,
                "v_actor": self.v_actor,
                "m_bias_actor": self.m_bias_actor,
                "v_bias_actor": self.v_bias_actor,
                "m_critic": self.m_critic,
                "v_critic": self.v_critic,
                "m_bias_critic": self.m_bias_critic,
                "v_bias_critic": self.v_bias_critic,
            }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ActorCriticNetwork":
        expected = [NETWORK_INPUTS, NETWORK_HIDDEN, NETWORK_ACTIONS, 1]
        if data.get("architecture") != expected:
            raise ValueError(f"Architecture incompatible ; attendu : {expected}")
        network = cls(seed=0)
        network.weights_hidden = [
            [float(value) for value in row]
            for row in data["weights_hidden"]  # type: ignore[index]
        ]
        network.biases_hidden = [
            float(value) for value in data["biases_hidden"]  # type: ignore[arg-type]
        ]
        network.weights_actor = [
            [float(value) for value in row]
            for row in data["weights_actor"]  # type: ignore[index]
        ]
        network.biases_actor = [
            float(value) for value in data["biases_actor"]  # type: ignore[arg-type]
        ]
        network.weights_critic = [
            float(value) for value in data["weights_critic"]  # type: ignore[arg-type]
        ]
        network.bias_critic = float(data["bias_critic"])  # type: ignore[arg-type]
        network.reset_optimizer()
        optimizer = data.get("optimizer")
        if isinstance(optimizer, dict):
            network.adam_step = int(optimizer["adam_step"])
            network.m_hidden = optimizer["m_hidden"]  # type: ignore[assignment]
            network.v_hidden = optimizer["v_hidden"]  # type: ignore[assignment]
            network.m_bias_hidden = optimizer["m_bias_hidden"]  # type: ignore[assignment]
            network.v_bias_hidden = optimizer["v_bias_hidden"]  # type: ignore[assignment]
            network.m_actor = optimizer["m_actor"]  # type: ignore[assignment]
            network.v_actor = optimizer["v_actor"]  # type: ignore[assignment]
            network.m_bias_actor = optimizer["m_bias_actor"]  # type: ignore[assignment]
            network.v_bias_actor = optimizer["v_bias_actor"]  # type: ignore[assignment]
            network.m_critic = optimizer["m_critic"]  # type: ignore[assignment]
            network.v_critic = optimizer["v_critic"]  # type: ignore[assignment]
            network.m_bias_critic = float(optimizer["m_bias_critic"])
            network.v_bias_critic = float(optimizer["v_bias_critic"])
        return network


# ---------------------------------------------------------------------------
# Environnement physique et mesures d'une poursuite
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
    action_index: int = 4
    captured: bool = False
    timed_out: bool = False


@dataclass
class StepOutcome:
    reward: float
    terminal: bool
    metrics: EpisodeMetrics | None


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


class CellWorld:
    def __init__(self, targets: list[Vec2]) -> None:
        if not targets:
            raise ValueError("Une séquence de cibles est obligatoire.")
        self.targets = targets
        self.target_index = 0
        self.target = targets[0]
        self.cell = CellState()
        self.previous_distance = self.distance_to_target()
        self.initial_distance = self.previous_distance
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_path_length = 0.0
        self.episode_energy = 0.0
        self.episode_wall_hits = 0
        self.total_steps = 0
        self.total_captures = 0
        self.total_timeouts = 0
        self.trajectory: list[Vec2] = [Vec2(self.cell.position.x, self.cell.position.y)]
        self.telemetry = Telemetry(distance_to_target=self.previous_distance)

    def distance_to_target(self) -> float:
        return (self.target - self.cell.position).length()

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

    def step(self, action_index: int, remember_trajectory: bool) -> StepOutcome:
        command_1, command_2 = ACTIONS[action_index]
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
        normalized_progress = (
            self.previous_distance - current_distance
        ) / max(self.initial_distance, 1.0)
        reward = PROGRESS_REWARD * normalized_progress
        reward -= TIME_PENALTY
        reward -= ENERGY_PENALTY * energy
        if wall_hit:
            reward -= WALL_PENALTY

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
                self.initial_distance / max(self.episode_step, 1) if captured else 0.0
            )
            path_efficiency = (
                min(
                    1.0,
                    self.initial_distance / max(self.episode_path_length, 1e-9),
                )
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
            action_index=action_index,
            captured=captured,
            timed_out=timed_out and not captured,
        )
        return StepOutcome(reward=reward, terminal=terminal, metrics=metrics)

    def to_dict(self) -> dict[str, object]:
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
            },
            "previous_distance": self.previous_distance,
            "initial_distance": self.initial_distance,
            "episode_step": self.episode_step,
            "episode_reward": self.episode_reward,
            "episode_path_length": self.episode_path_length,
            "episode_energy": self.episode_energy,
            "episode_wall_hits": self.episode_wall_hits,
            "total_steps": self.total_steps,
            "total_captures": self.total_captures,
            "total_timeouts": self.total_timeouts,
            "trajectory": [vector_data(point) for point in self.trajectory],
            "telemetry": {
                "force_segment_1": vector_data(self.telemetry.force_segment_1),
                "force_segment_2": vector_data(self.telemetry.force_segment_2),
                "propulsion": vector_data(self.telemetry.propulsion),
                "velocity": vector_data(self.telemetry.velocity),
                "target_direction": vector_data(self.telemetry.target_direction),
                "wall_reaction": vector_data(self.telemetry.wall_reaction),
                "torque": self.telemetry.torque,
                "reward": self.telemetry.reward,
                "distance_to_target": self.telemetry.distance_to_target,
                "action_index": self.telemetry.action_index,
                "captured": self.telemetry.captured,
                "timed_out": self.telemetry.timed_out,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CellWorld":
        def vector(values: object) -> Vec2:
            if not isinstance(values, list) or len(values) != 2:
                raise ValueError("Vecteur de session invalide.")
            return Vec2(float(values[0]), float(values[1]))

        targets = [vector(values) for values in data["targets"]]  # type: ignore[index]
        world = cls(targets)
        world.target_index = int(data["target_index"])
        world.target = world.targets[world.target_index]
        cell_data = data["cell"]
        if not isinstance(cell_data, dict):
            raise ValueError("État de cellule invalide.")
        world.cell = CellState(
            position=vector(cell_data["position"]),
            angle=float(cell_data["angle"]),
            velocity=vector(cell_data["velocity"]),
            angular_velocity=float(cell_data["angular_velocity"]),
            joint_angle_1=float(cell_data["joint_angle_1"]),
            joint_angle_2=float(cell_data["joint_angle_2"]),
            joint_speed_1=float(cell_data["joint_speed_1"]),
            joint_speed_2=float(cell_data["joint_speed_2"]),
        )
        for name in (
            "previous_distance",
            "initial_distance",
            "episode_reward",
            "episode_path_length",
            "episode_energy",
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
        world.trajectory = [
            vector(values) for values in data.get("trajectory", [])  # type: ignore[arg-type]
        ] or [Vec2(world.cell.position.x, world.cell.position.y)]

        telemetry_data = data.get("telemetry")
        if isinstance(telemetry_data, dict):
            world.telemetry = Telemetry(
                force_segment_1=vector(telemetry_data["force_segment_1"]),
                force_segment_2=vector(telemetry_data["force_segment_2"]),
                propulsion=vector(telemetry_data["propulsion"]),
                velocity=vector(telemetry_data["velocity"]),
                target_direction=vector(telemetry_data["target_direction"]),
                wall_reaction=vector(telemetry_data["wall_reaction"]),
                torque=float(telemetry_data["torque"]),
                reward=float(telemetry_data["reward"]),
                distance_to_target=float(telemetry_data["distance_to_target"]),
                action_index=int(telemetry_data["action_index"]),
                captured=bool(telemetry_data["captured"]),
                timed_out=bool(telemetry_data["timed_out"]),
            )
        return world


# ---------------------------------------------------------------------------
# Entraîneur : même réseau, un environnement d'apprentissage
# ---------------------------------------------------------------------------


@dataclass
class MetricsHistory:
    rewards: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    critic_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    td_errors: list[float] = field(default_factory=list)
    effective_speeds: list[float] = field(default_factory=list)
    normalized_times: list[float] = field(default_factory=list)
    path_efficiencies: list[float] = field(default_factory=list)
    successes: list[float] = field(default_factory=list)


class Trainer:
    def __init__(self, network: ActorCriticNetwork, seed: int = 1234) -> None:
        self.network = network
        self.rng = random.Random(seed)
        self.world = CellWorld(generate_target_sequence(seed + 1, TARGET_SEQUENCE_LENGTH))
        self.history = MetricsHistory()
        self.total_updates = 0
        self.episodes = 0
        self.captures = 0
        self.last_episode: EpisodeMetrics | None = None
        self.last_update = LearningUpdate(0.0, 0.0, 0.0, 0.0, 0.0)
        self.sum_actor_loss = 0.0
        self.sum_critic_loss = 0.0
        self.sum_entropy = 0.0
        self.sum_td_error = 0.0
        self.updates_in_episode = 0

    def train_step(self) -> EpisodeMetrics | None:
        inputs = self.world.observations()
        action = self.network.choose_action(inputs, self.rng, stochastic=True)
        outcome = self.world.step(action, remember_trajectory=False)
        next_inputs = self.world.observations()
        update = self.network.learn(
            inputs=inputs,
            action=action,
            reward=outcome.reward,
            next_inputs=next_inputs,
            terminal=outcome.terminal,
        )
        self.last_update = update
        self.total_updates += 1
        self.sum_actor_loss += update.actor_loss
        self.sum_critic_loss += update.critic_loss
        self.sum_entropy += update.entropy
        self.sum_td_error += update.td_error
        self.updates_in_episode += 1

        if outcome.metrics is None:
            return None

        metrics = outcome.metrics
        count = max(1, self.updates_in_episode)
        self.episodes += 1
        if metrics.captured:
            self.captures += 1
        self.last_episode = metrics
        self.history.rewards.append(metrics.reward)
        self.history.actor_losses.append(self.sum_actor_loss / count)
        self.history.critic_losses.append(self.sum_critic_loss / count)
        self.history.entropies.append(self.sum_entropy / count)
        self.history.td_errors.append(self.sum_td_error / count)
        self.history.effective_speeds.append(metrics.effective_speed)
        self.history.normalized_times.append(metrics.normalized_time)
        self.history.path_efficiencies.append(metrics.path_efficiency)
        self.history.successes.append(1.0 if metrics.captured else 0.0)

        self.sum_actor_loss = 0.0
        self.sum_critic_loss = 0.0
        self.sum_entropy = 0.0
        self.sum_td_error = 0.0
        self.updates_in_episode = 0
        return metrics

    def train_steps(self, count: int) -> None:
        for _ in range(count):
            self.train_step()

    def train_one_episode(self) -> EpisodeMetrics:
        while True:
            metrics = self.train_step()
            if metrics is not None:
                return metrics

    def success_rate(self, window: int = 20) -> float:
        recent = self.history.successes[-window:]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def to_dict(self) -> dict[str, object]:
        return {
            "rng_state": repr(self.rng.getstate()),
            "world": self.world.to_dict(),
            "history": asdict(self.history),
            "total_updates": self.total_updates,
            "episodes": self.episodes,
            "captures": self.captures,
            "last_episode": asdict(self.last_episode) if self.last_episode else None,
            "last_update": asdict(self.last_update),
            "sum_actor_loss": self.sum_actor_loss,
            "sum_critic_loss": self.sum_critic_loss,
            "sum_entropy": self.sum_entropy,
            "sum_td_error": self.sum_td_error,
            "updates_in_episode": self.updates_in_episode,
        }

    @classmethod
    def from_dict(
        cls, network: ActorCriticNetwork, data: dict[str, object]
    ) -> "Trainer":
        trainer = cls(network, seed=0)
        trainer.rng.setstate(ast.literal_eval(str(data["rng_state"])))
        world_data = data["world"]
        if not isinstance(world_data, dict):
            raise ValueError("Monde d'entraînement invalide.")
        trainer.world = CellWorld.from_dict(world_data)

        history_data = data["history"]
        if not isinstance(history_data, dict):
            raise ValueError("Historique d'entraînement invalide.")
        trainer.history = MetricsHistory(
            rewards=[float(value) for value in history_data["rewards"]],
            actor_losses=[float(value) for value in history_data["actor_losses"]],
            critic_losses=[float(value) for value in history_data["critic_losses"]],
            entropies=[float(value) for value in history_data["entropies"]],
            td_errors=[float(value) for value in history_data["td_errors"]],
            effective_speeds=[
                float(value) for value in history_data["effective_speeds"]
            ],
            normalized_times=[
                float(value) for value in history_data["normalized_times"]
            ],
            path_efficiencies=[
                float(value) for value in history_data["path_efficiencies"]
            ],
            successes=[float(value) for value in history_data["successes"]],
        )
        trainer.total_updates = int(data["total_updates"])
        trainer.episodes = int(data["episodes"])
        trainer.captures = int(data["captures"])

        last_episode_data = data.get("last_episode")
        trainer.last_episode = (
            EpisodeMetrics(**last_episode_data)  # type: ignore[arg-type]
            if isinstance(last_episode_data, dict)
            else None
        )
        last_update_data = data.get("last_update")
        if isinstance(last_update_data, dict):
            trainer.last_update = LearningUpdate(**last_update_data)  # type: ignore[arg-type]
        trainer.sum_actor_loss = float(data["sum_actor_loss"])
        trainer.sum_critic_loss = float(data["sum_critic_loss"])
        trainer.sum_entropy = float(data["sum_entropy"])
        trainer.sum_td_error = float(data["sum_td_error"])
        trainer.updates_in_episode = int(data["updates_in_episode"])
        return trainer


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
        self.root.title("Cellule 2D — Actor-Critic à réseau unique")
        self.root.geometry(f"{CANVAS_WIDTH}x{CANVAS_HEIGHT + 54}")
        self.root.minsize(CANVAS_WIDTH, CANVAS_HEIGHT + 54)

        self.network = ActorCriticNetwork(seed=42)
        self.trainer = Trainer(self.network, seed=1234)
        self.demo_rng = random.Random(9876)
        self.demo_world = CellWorld(
            generate_target_sequence(9090, TARGET_SEQUENCE_LENGTH)
        )
        self.learning_running = False
        self.show_vectors = True
        self.training_speeds = (1, 10, 50, 200)
        self.speed_index = 2
        self.capture_flash = 0

        toolbar = tk.Frame(self.root, bg="#18222f", padx=8, pady=7)
        toolbar.pack(fill="x")
        self.start_button = tk.Button(
            toolbar,
            text="Démarrer apprentissage",
            width=20,
            command=self.toggle_learning,
        )
        self.start_button.pack(side="left", padx=3)
        tk.Button(
            toolbar,
            text="Entraîner 1 poursuite",
            width=19,
            command=self.train_one_episode,
        ).pack(side="left", padx=3)
        self.speed_button = tk.Button(
            toolbar,
            text="Calcul : 50 pas/image",
            width=20,
            command=self.cycle_speed,
        )
        self.speed_button.pack(side="left", padx=3)
        self.vector_button = tk.Button(
            toolbar,
            text="Masquer vecteurs",
            width=16,
            command=self.toggle_vectors,
        )
        self.vector_button.pack(side="left", padx=3)
        tk.Button(
            toolbar, text="Sauver session", width=15, command=self.save_session
        ).pack(side="left", padx=3)
        tk.Button(
            toolbar, text="Restaurer session", width=17, command=self.load_session
        ).pack(side="left", padx=3)
        tk.Button(
            toolbar,
            text="Réinitialiser RN",
            width=16,
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

    def toggle_learning(self) -> None:
        self.learning_running = not self.learning_running
        self.start_button.configure(
            text="Pause apprentissage"
            if self.learning_running
            else "Démarrer apprentissage"
        )

    def train_one_episode(self) -> None:
        self.trainer.train_one_episode()

    def cycle_speed(self) -> None:
        self.speed_index = (self.speed_index + 1) % len(self.training_speeds)
        speed = self.training_speeds[self.speed_index]
        self.speed_button.configure(text=f"Calcul : {speed} pas/image")

    def toggle_vectors(self) -> None:
        self.show_vectors = not self.show_vectors
        self.vector_button.configure(
            text="Masquer vecteurs" if self.show_vectors else "Afficher vecteurs"
        )

    @staticmethod
    def session_root() -> Path:
        """Dossier sauvegardes placé exactement à côté du script."""

        return Path(__file__).resolve().parent / "sauvegardes"

    def save_session(self) -> None:
        root = self.session_root()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_directory = root / timestamp
        suffix = 1
        while session_directory.exists():
            session_directory = root / f"{timestamp}_{suffix:02d}"
            suffix += 1
        path = session_directory / SESSION_FILENAME
        payload = {
            "format": SESSION_FORMAT,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "network": self.network.to_dict(include_optimizer=True),
            "trainer": self.trainer.to_dict(),
            "demo_world": self.demo_world.to_dict(),
            "demo_rng_state": repr(self.demo_rng.getstate()),
            "application": {
                "training_speed_index": self.speed_index,
                "show_vectors": self.show_vectors,
                "capture_flash": self.capture_flash,
                "was_running": self.learning_running,
            },
        }
        try:
            session_directory.mkdir(parents=True, exist_ok=False)
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            self.messagebox.showerror("Sauvegarde impossible", str(exc))
            return
        self.messagebox.showinfo(
            "Session sauvegardée",
            f"L'entraînement complet a été enregistré dans :\n{session_directory}",
        )

    def load_session(self) -> None:
        root = self.session_root()
        root.mkdir(parents=True, exist_ok=True)

        # La sauvegarde est considérée comme un ensemble indivisible. On fait
        # donc choisir son dossier horodaté plutôt qu'un fichier isolé.
        selected_directory = self.filedialog.askdirectory(
            title="Sélectionner le dossier de session à restaurer",
            initialdir=str(root),
            mustexist=True,
        )
        if not selected_directory:
            return

        session_directory = Path(selected_directory)
        path = session_directory / SESSION_FILENAME
        if not path.is_file():
            self.messagebox.showerror(
                "Dossier de session invalide",
                "Le dossier sélectionné ne contient pas le fichier attendu :\n"
                f"{SESSION_FILENAME}\n\n"
                "Sélectionnez directement un dossier horodaté créé par "
                "« Sauver session ».",
            )
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format") != SESSION_FORMAT:
                raise ValueError(
                    "Ce dossier ne contient pas une session complète compatible."
                )
            network = ActorCriticNetwork.from_dict(payload["network"])
            trainer_data = payload["trainer"]
            demo_world_data = payload["demo_world"]
            if not isinstance(trainer_data, dict) or not isinstance(demo_world_data, dict):
                raise ValueError("Structure de session invalide.")
            trainer = Trainer.from_dict(network, trainer_data)
            demo_world = CellWorld.from_dict(demo_world_data)
            demo_rng = random.Random()
            demo_rng.setstate(ast.literal_eval(str(payload["demo_rng_state"])))
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            SyntaxError,
            json.JSONDecodeError,
        ) as exc:
            self.messagebox.showerror("Chargement impossible", str(exc))
            return
        self.network = network
        self.trainer = trainer
        self.demo_world = demo_world
        self.demo_rng = demo_rng
        app_data = payload.get("application", {})
        if isinstance(app_data, dict):
            self.speed_index = clamp(
                int(app_data.get("training_speed_index", 2)),
                0,
                len(self.training_speeds) - 1,
            )  # type: ignore[assignment]
            self.speed_index = int(self.speed_index)
            self.show_vectors = bool(app_data.get("show_vectors", True))
            self.capture_flash = int(app_data.get("capture_flash", 0))
        self.speed_button.configure(
            text=f"Calcul : {self.training_speeds[self.speed_index]} pas/image"
        )
        self.vector_button.configure(
            text="Masquer vecteurs" if self.show_vectors else "Afficher vecteurs"
        )
        # Une session chargée est volontairement mise en pause pour permettre
        # de vérifier ses compteurs avant de reprendre manuellement.
        self.learning_running = False
        self.start_button.configure(text="Démarrer apprentissage")
        self.messagebox.showinfo(
            "Session restaurée",
            "Poids, optimiseur, courbes et environnement ont été restaurés.\n"
            f"Dossier : {session_directory}\n\n"
            "Cliquez sur Démarrer apprentissage pour continuer.",
        )

    def reset_network(self) -> None:
        if not self.messagebox.askyesno(
            "Réinitialiser le réseau",
            "Effacer les poids appris et toutes les courbes ?",
        ):
            return
        self.learning_running = False
        self.start_button.configure(text="Démarrer apprentissage")
        seed = random.randrange(1, 1_000_000)
        self.network = ActorCriticNetwork(seed=seed)
        self.trainer = Trainer(self.network, seed=seed + 1)
        self.demo_world = CellWorld(
            generate_target_sequence(seed + 2, TARGET_SEQUENCE_LENGTH)
        )

    def tick(self) -> None:
        if self.learning_running:
            self.trainer.train_steps(self.training_speeds[self.speed_index])
            # La démonstration utilise le même réseau. Elle ne progresse que
            # lorsque l'apprentissage est actif : avant Démarrer et en Pause,
            # cellule, articulations et trajectoire restent strictement figées.
            demo_inputs = self.demo_world.observations()
            demo_action = self.network.choose_action(
                demo_inputs, self.demo_rng, stochastic=False
            )
            outcome = self.demo_world.step(demo_action, remember_trajectory=True)
            if outcome.metrics is not None and outcome.metrics.captured:
                self.capture_flash = 12
            elif self.capture_flash > 0:
                self.capture_flash -= 1

        self.draw()
        self.root.after(33, self.tick)

    def draw(self) -> None:
        self.canvas.delete("all")
        self.draw_world()
        self.draw_statistics()
        history = self.trainer.history
        reward_average = moving_average(history.rewards, 20)
        self.draw_chart(
            1125,
            18,
            1392,
            342,
            "RÉCOMPENSE PAR POURSUITE",
            (
                ("récompense", history.rewards, "#78909c"),
                ("moyenne 20", reward_average, "#62df79"),
            ),
        )
        self.draw_chart(
            1404,
            18,
            1668,
            342,
            "LOSSES ACTOR / CRITIC",
            (
                ("actor", history.actor_losses, "#47b7ff"),
                ("critic", history.critic_losses, "#ffad4d"),
            ),
            include_zero=True,
        )
        self.draw_chart(
            1125,
            354,
            1392,
            680,
            "VITESSE EFFECTIVE VERS CIBLE",
            (("distance initiale / pas", history.effective_speeds, "#5de578"),),
            fixed_minimum=0.0,
        )
        self.draw_chart(
            1404,
            354,
            1668,
            680,
            "EFFICACITÉ ET RÉUSSITE",
            (
                ("trajet", history.path_efficiencies, "#d58cff"),
                ("réussite 20", moving_average(history.successes, 20), "#ffe066"),
            ),
            fixed_minimum=0.0,
            fixed_maximum=1.0,
        )

    def draw_world(self) -> None:
        c = self.canvas
        for x in range(20, WORLD_WIDTH, 40):
            c.create_line(x, 20, x, WORLD_HEIGHT - 20, fill="#102536")
        for y in range(20, WORLD_HEIGHT, 40):
            c.create_line(20, y, WORLD_WIDTH - 20, y, fill="#102536")
        boundary_color = (
            "#ff8a65"
            if self.demo_world.telemetry.wall_reaction.length() > 0
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

        if len(self.demo_world.trajectory) >= 2:
            coordinates: list[float] = []
            for point in self.demo_world.trajectory:
                coordinates.extend((point.x, point.y))
            c.create_line(*coordinates, fill="#718792", width=2, smooth=True)

        target = self.demo_world.target
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
        ) = self.demo_world.flagellum_geometry()
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
        cosine = math.cos(self.demo_world.cell.angle)
        sine = math.sin(self.demo_world.cell.angle)
        for index in range(24):
            phi = 2.0 * math.pi * index / 24
            local_x = BODY_HALF_LENGTH * math.cos(phi)
            local_y = BODY_HALF_WIDTH * math.sin(phi)
            body_points.extend(
                (
                    self.demo_world.cell.position.x + cosine * local_x - sine * local_y,
                    self.demo_world.cell.position.y + sine * local_x + cosine * local_y,
                )
            )
        c.create_polygon(
            *body_points, fill="#67d8a2", outline="#d8ffea", width=2, smooth=True
        )
        front = self.demo_world.cell.position + direction(self.demo_world.cell.angle) * 13
        c.create_oval(
            front.x - 3,
            front.y - 3,
            front.x + 3,
            front.y + 3,
            fill="#153d2d",
        )

        telemetry = self.demo_world.telemetry
        if self.show_vectors:
            self.draw_arrow(middle_1, telemetry.force_segment_1, 80, "#ff9f43", "F1")
            self.draw_arrow(middle_2, telemetry.force_segment_2, 80, "#ff7f50", "F2")
            self.draw_arrow(
                self.demo_world.cell.position,
                telemetry.propulsion,
                72,
                "#46e66d",
                "PROP",
            )
            self.draw_arrow(
                self.demo_world.cell.position,
                telemetry.velocity,
                78,
                "#48a9ff",
                "V",
            )
            self.draw_arrow(
                self.demo_world.cell.position,
                telemetry.target_direction,
                68,
                "#cf8cff",
                "CIBLE",
                dash=(5, 3),
            )
            if telemetry.wall_reaction.length() > 0:
                self.draw_arrow(
                    self.demo_world.cell.position,
                    telemetry.wall_reaction,
                    30,
                    "#ff4b4b",
                    "PAROI",
                )

        c.create_text(
            30,
            WORLD_HEIGHT - 10,
            text="F1/F2 : forces des segments   PROP : propulsion   V : vitesse",
            anchor="sw",
            fill="#a9bbc6",
            font=("Segoe UI", 9),
        )

    def draw_statistics(self) -> None:
        """Place les mesures hors du monde, entre la grille et les graphiques."""

        c = self.canvas
        left, top, right, bottom = 842, 18, 1112, 680
        c.create_rectangle(
            left, top, right, bottom, fill="#0c1722", outline="#38566b", width=2
        )
        state = "APPRENTISSAGE ACTIF" if self.learning_running else "PAUSE — MONDE FIGÉ"
        state_color = "#66e58a" if self.learning_running else "#ffca5c"
        c.create_text(
            (left + right) * 0.5,
            top + 24,
            text="MESURES DU RN UNIQUE",
            fill="#e9f4fa",
            font=("Segoe UI", 11, "bold"),
        )
        c.create_text(
            (left + right) * 0.5,
            top + 51,
            text=state,
            fill=state_color,
            font=("Segoe UI", 9, "bold"),
        )

        telemetry = self.demo_world.telemetry
        speed = self.demo_world.cell.velocity.length()
        local_velocity = rotate_into_cell_frame(
            self.demo_world.cell.velocity, self.demo_world.cell.angle
        )
        if speed < 0.025:
            movement = "ARRÊT"
        elif local_velocity.x < -0.025:
            movement = "RECUL"
        elif abs(local_velocity.y) > abs(local_velocity.x):
            movement = "DÉRIVE LATÉRALE"
        else:
            movement = "AVANCE"

        learning_info = (
            f"APPRENTISSAGE\n"
            f"Pas        : {self.trainer.total_updates}\n"
            f"Poursuites : {self.trainer.episodes}\n"
            f"Captures   : {self.trainer.captures}\n"
            f"Réussite20 : {100*self.trainer.success_rate():5.1f}%\n\n"
            f"Actor loss : {self.trainer.last_update.actor_loss:+.5f}\n"
            f"Critic loss: {self.trainer.last_update.critic_loss:.5f}\n"
            f"Entropie   : {self.trainer.last_update.entropy:.4f}\n"
            f"Erreur TD  : {self.trainer.last_update.td_error:.5f}"
        )
        c.create_text(
            left + 17,
            top + 82,
            text=learning_info,
            anchor="nw",
            fill="#d8e7ef",
            font=("Consolas", 9),
        )

        command = ACTIONS[telemetry.action_index]
        demo_info = (
            f"DÉMONSTRATION\n"
            f"Distance : {telemetry.distance_to_target:6.1f}\n"
            f"Vitesse  : {speed:.3f}\n"
            f"Régime   : {movement}\n"
            f"Rotation : {math.degrees(self.demo_world.cell.angular_velocity):+.2f}°/pas\n"
            f"Angle base: {math.degrees(self.demo_world.cell.joint_angle_1):+5.1f}° / ±40°\n"
            f"Angle seg.: {math.degrees(self.demo_world.cell.joint_angle_2):+5.1f}° / ±55°\n"
            f"Action   : {command}\n"
            f"Cibles   : {self.demo_world.total_captures}"
        )
        c.create_text(
            left + 17,
            top + 285,
            text=demo_info,
            anchor="nw",
            fill="#c5e3ef",
            font=("Consolas", 9),
        )

        last = self.trainer.last_episode
        if last is None:
            last_info = "DERNIÈRE POURSUITE\nEn attente"
        else:
            last_info = (
                f"DERNIÈRE POURSUITE\n"
                f"Résultat  : {'SUCCÈS' if last.captured else 'ÉCHEC'}\n"
                f"Récompense: {last.reward:+.3f}\n"
                f"Pas       : {last.steps}\n"
                f"Distance  : {last.initial_distance:.1f}\n"
                f"V.efficace: {last.effective_speed:.4f}\n"
                f"Temps/dist: {last.normalized_time:.3f}\n"
                f"Efficacité: {last.path_efficiency:.3f}"
            )
        c.create_text(
            left + 17,
            top + 438,
            text=last_info,
            anchor="nw",
            fill="#d9c8f3",
            font=("Consolas", 9),
        )
        c.create_text(
            (left + right) * 0.5,
            bottom - 22,
            text="Le recul n'est ni récompensé ni pénalisé.",
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
            arrowshape=(10, 12, 5),
            dash=dash,
        )
        self.canvas.create_text(
            endpoint.x + 4,
            endpoint.y - 5,
            text=label,
            anchor="w",
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
        fixed_minimum: float | None = None,
        fixed_maximum: float | None = None,
        include_zero: bool = False,
    ) -> None:
        c = self.canvas
        c.create_rectangle(
            left, top, right, bottom, fill="#0c1722", outline="#38566b", width=2
        )
        c.create_text(
            (left + right) * 0.5,
            top + 18,
            text=title,
            fill="#e9f4fa",
            font=("Segoe UI", 10, "bold"),
        )
        plot_left = left + 43
        plot_right = right - 14
        plot_top = top + 48
        plot_bottom = bottom - 48
        c.create_line(plot_left, plot_bottom, plot_right, plot_bottom, fill="#607886")
        c.create_line(plot_left, plot_top, plot_left, plot_bottom, fill="#607886")

        visible_series = [(name, values[-120:], color) for name, values, color in series]
        all_values = [
            value
            for _name, values, _color in visible_series
            for value in values
            if math.isfinite(value)
        ]
        if not all_values:
            c.create_text(
                (plot_left + plot_right) * 0.5,
                (plot_top + plot_bottom) * 0.5,
                text="En attente d'une poursuite terminée",
                fill="#718895",
                font=("Segoe UI", 8),
            )
            return

        minimum = fixed_minimum if fixed_minimum is not None else min(all_values)
        maximum = fixed_maximum if fixed_maximum is not None else max(all_values)
        if include_zero:
            minimum = min(minimum, 0.0)
            maximum = max(maximum, 0.0)
        if math.isclose(minimum, maximum):
            padding = max(0.1, abs(maximum) * 0.1)
            minimum -= padding
            maximum += padding

        for grid_index in range(5):
            y = plot_top + grid_index * (plot_bottom - plot_top) / 4
            c.create_line(plot_left, y, plot_right, y, fill="#172b39")
        c.create_text(
            plot_left - 5,
            plot_top,
            text=f"{maximum:.3g}",
            anchor="e",
            fill="#8fa6b3",
            font=("Consolas", 7),
        )
        c.create_text(
            plot_left - 5,
            plot_bottom,
            text=f"{minimum:.3g}",
            anchor="e",
            fill="#8fa6b3",
            font=("Consolas", 7),
        )

        for name, values, color in visible_series:
            if not values:
                continue
            points: list[float] = []
            for index, value in enumerate(values):
                fraction_x = index / max(1, len(values) - 1)
                fraction_y = (value - minimum) / (maximum - minimum)
                points.extend(
                    (
                        plot_left + fraction_x * (plot_right - plot_left),
                        plot_bottom - fraction_y * (plot_bottom - plot_top),
                    )
                )
            if len(points) >= 4:
                c.create_line(*points, fill=color, width=2)
            else:
                c.create_oval(
                    points[0] - 2,
                    points[1] - 2,
                    points[0] + 2,
                    points[1] + 2,
                    fill=color,
                )

        legend_x = left + 12
        for name, values, color in visible_series:
            if not values:
                continue
            c.create_text(
                legend_x,
                bottom - 23,
                text=f"{name}: {values[-1]:.4g}",
                anchor="w",
                fill=color,
                font=("Segoe UI", 8),
            )
            legend_x += max(105, len(name) * 6 + 62)

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Test sans interface
# ---------------------------------------------------------------------------


def run_headless(training_steps: int) -> None:
    network = ActorCriticNetwork(seed=42)
    trainer = Trainer(network, seed=1234)
    last_report = 0
    for _ in range(training_steps):
        metrics = trainer.train_step()
        if metrics is not None and trainer.episodes - last_report >= 10:
            last_report = trainer.episodes
            print(
                f"épisodes={trainer.episodes:4d} "
                f"captures={trainer.captures:4d} "
                f"réussite20={100*trainer.success_rate():5.1f}% "
                f"reward={metrics.reward:+8.3f} "
                f"actor={trainer.history.actor_losses[-1]:+8.5f} "
                f"critic={trainer.history.critic_losses[-1]:8.5f}"
            )
    print(
        f"FINAL pas={trainer.total_updates} épisodes={trainer.episodes} "
        f"captures={trainer.captures} réussite20={100*trainer.success_rate():.1f}%"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cellule 2D contrôlée par un Actor-Critic à réseau unique."
    )
    parser.add_argument(
        "--headless",
        type=int,
        metavar="PAS",
        help="entraîne ce nombre de pas sans ouvrir l'interface",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.headless is not None:
        run_headless(max(1, arguments.headless))
    else:
        Application().run()


if __name__ == "__main__":
    main()
