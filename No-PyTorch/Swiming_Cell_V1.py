"""
Cellule nageuse 2D : apprentissage visible par neuro-évolution.

Objectif pédagogique
--------------------
Une cellule possède une faible vitesse propre et un flagelle à deux segments.
Un petit réseau de neurones commande les deux articulations. Sa récompense
dépend du rapprochement vers une cible fixe. Quand la cellule atteint la zone
de capture (deux fois le rayon de la cible), une nouvelle cible apparaît.

Le programme utilise exclusivement la bibliothèque standard Python :
    - tkinter pour l'interface ;
    - math et random pour le calcul ;
    - json pour sauvegarder le meilleur réseau.

Lancement :
    python cellule_flagelle_apprentissage.py

Test sans interface (utile pour vérifier le moteur) :
    python cellule_flagelle_apprentissage.py --headless 3

Important : la physique est un modèle visqueux pédagogique, pas une simulation
hydrodynamique exhaustive. Les forces latérales et longitudinales sont
différenciées afin qu'un cycle non réciproque du flagelle puisse produire une
propulsion et un couple de rotation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Paramètres du monde et de la cellule
# ---------------------------------------------------------------------------

WORLD_WIDTH = 780
WORLD_HEIGHT = 680
CANVAS_WIDTH = 1220
CANVAS_HEIGHT = 720
WORLD_MARGIN = 18.0

CELL_RADIUS = 14.0
BODY_HALF_LENGTH = 19.0
BODY_HALF_WIDTH = 12.0
SEGMENT_LENGTH = 34.0

TARGET_RADIUS = 10.0
CAPTURE_RADIUS = 2.0 * TARGET_RADIUS
MIN_TARGET_DISTANCE = 210.0

# La cellule avance toujours un peu dans l'axe de son corps.
BASE_SPEED = 0.20

# Paramètres des deux articulations du flagelle.
MAX_JOINT_ANGLE = 1.15
MAX_JOINT_SPEED = 0.13
MOTOR_ACCELERATION = 0.024
JOINT_DAMPING = 0.91

# Résistance plus forte perpendiculairement au segment que longitudinalement.
DRAG_PARALLEL = 0.018
DRAG_PERPENDICULAR = 0.062
FORCE_TO_SPEED = 1.35
TORQUE_TO_ROTATION = 0.0035
SHAPE_PROPULSION = 1.50

# Apprentissage par population de petits réseaux.
NETWORK_INPUTS = 6
NETWORK_HIDDEN = 6
NETWORK_OUTPUTS = 2
POPULATION_SIZE = 18
ELITE_COUNT = 3
# 2 400 pas donnent à la vitesse propre (0,20 pixel/pas) le temps de parcourir
# environ 480 pixels. Une cellule correctement orientée peut donc atteindre une
# première cible éloignée, alors qu'une mauvaise politique reste pénalisée.
EVALUATION_STEPS = 2_400
TARGET_SEQUENCE_LENGTH = 64

# Récompenses et pénalités.
PROGRESS_REWARD = 0.12
CAPTURE_REWARD = 25.0
ENERGY_PENALTY = 0.0015
WALL_PENALTY = 0.7


# ---------------------------------------------------------------------------
# Petites fonctions vectorielles : aucun module scientifique externe requis
# ---------------------------------------------------------------------------


@dataclass
class Vec2:
    """Vecteur 2D minimal utilisé pour les positions, vitesses et forces."""

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
        """Produit vectoriel 2D : valeur scalaire selon l'axe z."""

        return self.x * other.y - self.y * other.x

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        magnitude = self.length()
        if magnitude < 1e-12:
            return Vec2()
        return self * (1.0 / magnitude)


def direction(angle: float) -> Vec2:
    """Vecteur unitaire orienté selon un angle en radians."""

    return Vec2(math.cos(angle), math.sin(angle))


def perpendicular(vector: Vec2) -> Vec2:
    """Tourne un vecteur de 90 degrés dans le sens trigonométrique."""

    return Vec2(-vector.y, vector.x)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def rotate_into_cell_frame(vector: Vec2, cell_angle: float) -> Vec2:
    """Exprime un vecteur mondial dans le repère propre de la cellule."""

    cosine = math.cos(cell_angle)
    sine = math.sin(cell_angle)
    return Vec2(
        cosine * vector.x + sine * vector.y,
        -sine * vector.x + cosine * vector.y,
    )


# ---------------------------------------------------------------------------
# Réseau de neurones entièrement écrit dans ce fichier
# ---------------------------------------------------------------------------


@dataclass
class NeuralNetwork:
    """Réseau 6 entrées -> 6 neurones cachés -> 2 sorties."""

    weights_input_hidden: list[list[float]]
    biases_hidden: list[float]
    weights_hidden_output: list[list[float]]
    biases_output: list[float]

    @classmethod
    def random_network(cls, rng: random.Random) -> "NeuralNetwork":
        """Crée un réseau aux poids aléatoires centrés autour de zéro."""

        return cls(
            weights_input_hidden=[
                [rng.uniform(-1.0, 1.0) for _ in range(NETWORK_INPUTS)]
                for _ in range(NETWORK_HIDDEN)
            ],
            biases_hidden=[rng.uniform(-0.35, 0.35) for _ in range(NETWORK_HIDDEN)],
            weights_hidden_output=[
                [rng.uniform(-1.0, 1.0) for _ in range(NETWORK_HIDDEN)]
                for _ in range(NETWORK_OUTPUTS)
            ],
            biases_output=[rng.uniform(-0.35, 0.35) for _ in range(NETWORK_OUTPUTS)],
        )

    def forward(self, inputs: list[float]) -> tuple[list[float], list[float]]:
        """Calcule les activations cachées puis les deux commandes motrices."""

        hidden: list[float] = []
        for neuron_weights, bias in zip(
            self.weights_input_hidden, self.biases_hidden
        ):
            total = bias + sum(
                weight * value for weight, value in zip(neuron_weights, inputs)
            )
            hidden.append(math.tanh(total))

        outputs: list[float] = []
        for neuron_weights, bias in zip(
            self.weights_hidden_output, self.biases_output
        ):
            total = bias + sum(
                weight * value for weight, value in zip(neuron_weights, hidden)
            )
            outputs.append(math.tanh(total))
        return outputs, hidden

    def copy(self) -> "NeuralNetwork":
        return NeuralNetwork.from_dict(self.to_dict())

    def mutated(
        self, rng: random.Random, sigma: float, mutation_probability: float = 0.32
    ) -> "NeuralNetwork":
        """Copie le réseau puis perturbe une partie de ses poids."""

        child = self.copy()

        def mutate_value(value: float) -> float:
            if rng.random() < mutation_probability:
                value += rng.gauss(0.0, sigma)
            return clamp(value, -4.0, 4.0)

        child.weights_input_hidden = [
            [mutate_value(value) for value in row]
            for row in child.weights_input_hidden
        ]
        child.biases_hidden = [mutate_value(value) for value in child.biases_hidden]
        child.weights_hidden_output = [
            [mutate_value(value) for value in row]
            for row in child.weights_hidden_output
        ]
        child.biases_output = [mutate_value(value) for value in child.biases_output]
        return child

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": [NETWORK_INPUTS, NETWORK_HIDDEN, NETWORK_OUTPUTS],
            "weights_input_hidden": self.weights_input_hidden,
            "biases_hidden": self.biases_hidden,
            "weights_hidden_output": self.weights_hidden_output,
            "biases_output": self.biases_output,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "NeuralNetwork":
        architecture = data.get("architecture")
        if architecture != [NETWORK_INPUTS, NETWORK_HIDDEN, NETWORK_OUTPUTS]:
            raise ValueError("Architecture de réseau incompatible.")
        return cls(
            weights_input_hidden=[
                [float(value) for value in row]
                for row in data["weights_input_hidden"]  # type: ignore[index]
            ],
            biases_hidden=[
                float(value) for value in data["biases_hidden"]  # type: ignore[arg-type]
            ],
            weights_hidden_output=[
                [float(value) for value in row]
                for row in data["weights_hidden_output"]  # type: ignore[index]
            ],
            biases_output=[
                float(value) for value in data["biases_output"]  # type: ignore[arg-type]
            ],
        )


# ---------------------------------------------------------------------------
# Monde physique simplifié
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
class Telemetry:
    """Valeurs physiques conservées pour les vecteurs et le tableau de bord."""

    inputs: list[float] = field(default_factory=lambda: [0.0] * NETWORK_INPUTS)
    hidden: list[float] = field(default_factory=lambda: [0.0] * NETWORK_HIDDEN)
    outputs: list[float] = field(default_factory=lambda: [0.0] * NETWORK_OUTPUTS)
    force_segment_1: Vec2 = field(default_factory=Vec2)
    force_segment_2: Vec2 = field(default_factory=Vec2)
    propulsion: Vec2 = field(default_factory=Vec2)
    velocity: Vec2 = field(default_factory=Vec2)
    target_direction: Vec2 = field(default_factory=Vec2)
    wall_reaction: Vec2 = field(default_factory=Vec2)
    torque: float = 0.0
    reward: float = 0.0
    distance_to_target: float = 0.0
    captured: bool = False


def generate_target_sequence(seed: int, count: int) -> list[Vec2]:
    """Produit une suite reproductible de cibles suffisamment espacées."""

    rng = random.Random(seed)
    targets: list[Vec2] = []
    previous = Vec2(WORLD_WIDTH * 0.5, WORLD_HEIGHT * 0.5)
    safe_margin = WORLD_MARGIN + CAPTURE_RADIUS + 8.0

    for _ in range(count):
        candidate = previous
        for _attempt in range(500):
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
    """Contient la cellule, la cible fixe et les équations de mouvement."""

    def __init__(self, targets: list[Vec2]) -> None:
        if not targets:
            raise ValueError("La séquence de cibles ne peut pas être vide.")
        self.targets = targets
        self.target_index = 0
        self.target = targets[0]
        self.cell = CellState()
        self.previous_distance = self.distance_to_target()
        self.total_reward = 0.0
        self.captures = 0
        self.wall_hits = 0
        self.energy = 0.0
        self.distance_travelled = 0.0
        self.steps = 0
        self.telemetry = Telemetry(distance_to_target=self.previous_distance)
        self.trajectory: list[Vec2] = [self.cell.position]

    def distance_to_target(self) -> float:
        return (self.target - self.cell.position).length()

    def neural_inputs(self) -> list[float]:
        """Construit six observations normalisées, vues depuis la cellule."""

        relative_world = self.target - self.cell.position
        relative_local = rotate_into_cell_frame(relative_world, self.cell.angle)
        half_width = WORLD_WIDTH * 0.5
        half_height = WORLD_HEIGHT * 0.5
        return [
            clamp(relative_local.x / half_width, -1.5, 1.5),
            clamp(relative_local.y / half_height, -1.5, 1.5),
            self.cell.joint_angle_1 / MAX_JOINT_ANGLE,
            self.cell.joint_angle_2 / MAX_JOINT_ANGLE,
            self.cell.joint_speed_1 / MAX_JOINT_SPEED,
            self.cell.joint_speed_2 / MAX_JOINT_SPEED,
        ]

    def flagellum_geometry(
        self,
    ) -> tuple[Vec2, Vec2, Vec2, Vec2, Vec2, Vec2, Vec2, Vec2]:
        """Retourne attaches, extrémités, milieux et directions des segments."""

        body_axis = direction(self.cell.angle)
        attachment = self.cell.position - body_axis * BODY_HALF_LENGTH

        segment_angle_1 = self.cell.angle + math.pi + self.cell.joint_angle_1
        tangent_1 = direction(segment_angle_1)
        end_1 = attachment + tangent_1 * SEGMENT_LENGTH
        middle_1 = attachment + tangent_1 * (SEGMENT_LENGTH * 0.5)

        segment_angle_2 = segment_angle_1 + self.cell.joint_angle_2
        tangent_2 = direction(segment_angle_2)
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
        """Force du fluide, opposée au déplacement relatif du segment."""

        normal = perpendicular(tangent)
        longitudinal = tangent * velocity.dot(tangent)
        lateral = normal * velocity.dot(normal)
        return longitudinal * (-DRAG_PARALLEL) + lateral * (-DRAG_PERPENDICULAR)

    def _advance_target(self) -> None:
        self.target_index = (self.target_index + 1) % len(self.targets)
        self.target = self.targets[self.target_index]
        self.previous_distance = self.distance_to_target()

    def _apply_boundaries(self) -> tuple[bool, Vec2]:
        """Maintient la cellule dans le rectangle et renvoie la réaction visible."""

        hit = False
        reaction = Vec2()
        minimum_x = WORLD_MARGIN + CELL_RADIUS
        maximum_x = WORLD_WIDTH - WORLD_MARGIN - CELL_RADIUS
        minimum_y = WORLD_MARGIN + CELL_RADIUS
        maximum_y = WORLD_HEIGHT - WORLD_MARGIN - CELL_RADIUS

        if self.cell.position.x < minimum_x:
            correction = minimum_x - self.cell.position.x
            self.cell.position.x = minimum_x
            self.cell.velocity.x = abs(self.cell.velocity.x) * 0.25
            reaction.x += correction + 0.3
            hit = True
        elif self.cell.position.x > maximum_x:
            correction = self.cell.position.x - maximum_x
            self.cell.position.x = maximum_x
            self.cell.velocity.x = -abs(self.cell.velocity.x) * 0.25
            reaction.x -= correction + 0.3
            hit = True

        if self.cell.position.y < minimum_y:
            correction = minimum_y - self.cell.position.y
            self.cell.position.y = minimum_y
            self.cell.velocity.y = abs(self.cell.velocity.y) * 0.25
            reaction.y += correction + 0.3
            hit = True
        elif self.cell.position.y > maximum_y:
            correction = self.cell.position.y - maximum_y
            self.cell.position.y = maximum_y
            self.cell.velocity.y = -abs(self.cell.velocity.y) * 0.25
            reaction.y -= correction + 0.3
            hit = True
        return hit, reaction

    def step(self, network: NeuralNetwork, remember_trajectory: bool = False) -> float:
        """Exécute une décision du réseau puis une étape de physique."""

        inputs = self.neural_inputs()
        outputs, hidden = network.forward(inputs)

        # Les sorties accélèrent les articulations ; l'amortissement évite les
        # oscillations numériques infinies.
        self.cell.joint_speed_1 = clamp(
            self.cell.joint_speed_1 * JOINT_DAMPING
            + outputs[0] * MOTOR_ACCELERATION,
            -MAX_JOINT_SPEED,
            MAX_JOINT_SPEED,
        )
        self.cell.joint_speed_2 = clamp(
            self.cell.joint_speed_2 * JOINT_DAMPING
            + outputs[1] * MOTOR_ACCELERATION,
            -MAX_JOINT_SPEED,
            MAX_JOINT_SPEED,
        )

        self.cell.joint_angle_1 += self.cell.joint_speed_1
        self.cell.joint_angle_2 += self.cell.joint_speed_2

        # Une butée absorbe une partie de la vitesse articulaire.
        if abs(self.cell.joint_angle_1) > MAX_JOINT_ANGLE:
            self.cell.joint_angle_1 = clamp(
                self.cell.joint_angle_1, -MAX_JOINT_ANGLE, MAX_JOINT_ANGLE
            )
            self.cell.joint_speed_1 *= -0.22
        if abs(self.cell.joint_angle_2) > MAX_JOINT_ANGLE:
            self.cell.joint_angle_2 = clamp(
                self.cell.joint_angle_2, -MAX_JOINT_ANGLE, MAX_JOINT_ANGLE
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

        # Vitesse relative des milieux de segments due aux articulations.
        velocity_middle_1 = (
            perpendicular(tangent_1)
            * (SEGMENT_LENGTH * 0.5 * self.cell.joint_speed_1)
        )
        velocity_joint_2 = (
            perpendicular(tangent_1)
            * (SEGMENT_LENGTH * self.cell.joint_speed_1)
        )
        velocity_middle_2 = velocity_joint_2 + perpendicular(tangent_2) * (
            SEGMENT_LENGTH
            * 0.5
            * (self.cell.joint_speed_1 + self.cell.joint_speed_2)
        )

        force_1 = self.segment_drag(velocity_middle_1, tangent_1)
        force_2 = self.segment_drag(velocity_middle_2, tangent_2)

        # Le terme d'aire alpha1*dalpha2-alpha2*dalpha1 favorise les cycles
        # non réciproques : plier/déplier exactement à l'envers ne suffit pas.
        shape_rate = (
            self.cell.joint_angle_1 * self.cell.joint_speed_2
            - self.cell.joint_angle_2 * self.cell.joint_speed_1
        )
        shape_force = body_axis * (SHAPE_PROPULSION * shape_rate)
        force_1 = force_1 + shape_force * 0.5
        force_2 = force_2 + shape_force * 0.5
        total_flagellar_force = force_1 + force_2

        # Les forces décentrées génèrent un couple et font pivoter la cellule.
        torque = (middle_1 - self.cell.position).cross(force_1)
        torque += (middle_2 - self.cell.position).cross(force_2)
        target_angular_velocity = torque * TORQUE_TO_ROTATION
        self.cell.angular_velocity = (
            0.68 * self.cell.angular_velocity + 0.32 * target_angular_velocity
        )
        self.cell.angle += self.cell.angular_velocity
        self.cell.angle = math.atan2(
            math.sin(self.cell.angle), math.cos(self.cell.angle)
        )

        # Vitesse propre faible + effet du flagelle. La cellule se déplace donc
        # toujours un peu, même avant que le réseau apprenne une bonne nage.
        base_velocity = direction(self.cell.angle) * BASE_SPEED
        propulsion = base_velocity + total_flagellar_force * FORCE_TO_SPEED
        old_position = Vec2(self.cell.position.x, self.cell.position.y)
        self.cell.velocity = self.cell.velocity * 0.55 + propulsion * 0.45
        self.cell.position = self.cell.position + self.cell.velocity
        wall_hit, wall_reaction = self._apply_boundaries()

        displacement = (self.cell.position - old_position).length()
        self.distance_travelled += displacement
        self.energy += outputs[0] ** 2 + outputs[1] ** 2
        self.steps += 1

        current_distance = self.distance_to_target()
        progress = self.previous_distance - current_distance
        reward = PROGRESS_REWARD * progress
        reward -= ENERGY_PENALTY * (outputs[0] ** 2 + outputs[1] ** 2)
        if wall_hit:
            reward -= WALL_PENALTY
            self.wall_hits += 1

        captured = current_distance <= CAPTURE_RADIUS
        if captured:
            reward += CAPTURE_REWARD
            self.captures += 1
            self._advance_target()
            current_distance = self.previous_distance
        else:
            self.previous_distance = current_distance

        self.total_reward += reward
        if remember_trajectory:
            self.trajectory.append(Vec2(self.cell.position.x, self.cell.position.y))
            if len(self.trajectory) > 600:
                del self.trajectory[:100]

        self.telemetry = Telemetry(
            inputs=inputs,
            hidden=hidden,
            outputs=outputs,
            force_segment_1=force_1,
            force_segment_2=force_2,
            propulsion=propulsion,
            velocity=self.cell.velocity,
            target_direction=(self.target - self.cell.position).normalized(),
            wall_reaction=wall_reaction,
            torque=torque,
            reward=reward,
            distance_to_target=current_distance,
            captured=captured,
        )
        return reward


# ---------------------------------------------------------------------------
# Neuro-évolution : sélection des meilleurs puis mutation de leurs poids
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    fitness: float
    captures: int
    wall_hits: int
    energy: float
    distance_travelled: float


def evaluate_network(
    network: NeuralNetwork, targets: list[Vec2], steps: int = EVALUATION_STEPS
) -> EvaluationResult:
    world = CellWorld(targets)
    for _ in range(steps):
        world.step(network, remember_trajectory=False)
    return EvaluationResult(
        fitness=world.total_reward,
        captures=world.captures,
        wall_hits=world.wall_hits,
        energy=world.energy,
        distance_travelled=world.distance_travelled,
    )


class Evolution:
    """Population, évaluation, sélection et mutation des réseaux."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.targets = generate_target_sequence(1001, TARGET_SEQUENCE_LENGTH)
        self.population = [
            NeuralNetwork.random_network(self.rng) for _ in range(POPULATION_SIZE)
        ]
        self.generation = 0
        self.best_network = self.population[0].copy()
        self.best_ever_fitness = -math.inf
        self.best_result = EvaluationResult(-math.inf, 0, 0, 0.0, 0.0)
        self.history_best: list[float] = []
        self.history_average: list[float] = []
        self.last_champion_changed = False

    def run_generation(self) -> EvaluationResult:
        scored = [
            (evaluate_network(network, self.targets), network)
            for network in self.population
        ]
        scored.sort(key=lambda item: item[0].fitness, reverse=True)
        results = [result for result, _network in scored]
        generation_best_result, generation_best_network = scored[0]
        average = sum(result.fitness for result in results) / len(results)

        self.generation += 1
        self.history_best.append(generation_best_result.fitness)
        self.history_average.append(average)
        self.last_champion_changed = False

        if generation_best_result.fitness > self.best_ever_fitness:
            self.best_ever_fitness = generation_best_result.fitness
            self.best_network = generation_best_network.copy()
            self.best_result = generation_best_result
            self.last_champion_changed = True

        # L'amplitude des mutations décroît progressivement sans disparaître.
        sigma = max(0.075, 0.30 * (0.985 ** self.generation))
        elites = [network.copy() for _result, network in scored[:ELITE_COUNT]]
        parent_pool = [network for _result, network in scored[:8]]
        next_population = elites

        while len(next_population) < POPULATION_SIZE - 1:
            # Les meilleurs parents sont légèrement plus souvent sélectionnés.
            rank = int((self.rng.random() ** 1.8) * len(parent_pool))
            parent = parent_pool[min(rank, len(parent_pool) - 1)]
            next_population.append(parent.mutated(self.rng, sigma=sigma))

        # Un nouveau réseau aléatoire maintient une petite diversité génétique.
        next_population.append(NeuralNetwork.random_network(self.rng))
        self.population = next_population
        return generation_best_result


# ---------------------------------------------------------------------------
# Interface Tkinter et visualisations
# ---------------------------------------------------------------------------


class Application:
    INPUT_LABELS = ("cible X", "cible Y", "angle 1", "angle 2", "vit. 1", "vit. 2")
    OUTPUT_LABELS = ("moteur 1", "moteur 2")

    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.root = tk.Tk()
        self.root.title("Cellule 2D — apprentissage visible du flagelle")
        self.root.geometry(f"{CANVAS_WIDTH}x{CANVAS_HEIGHT + 52}")
        self.root.minsize(CANVAS_WIDTH, CANVAS_HEIGHT + 52)

        self.evolution = Evolution(seed=42)
        demo_targets = generate_target_sequence(9090, TARGET_SEQUENCE_LENGTH)
        self.demo_world = CellWorld(demo_targets)
        self.demo_network = self.evolution.best_network.copy()

        self.running = False
        self.show_vectors = True
        self.training_speeds = (1, 2, 4)
        self.training_speed_index = 0
        self.frames_since_training = 0
        self.capture_flash = 0

        toolbar = tk.Frame(self.root, bg="#18222f", padx=8, pady=7)
        toolbar.pack(fill="x")

        self.start_button = tk.Button(
            toolbar, text="Démarrer", width=13, command=self.toggle_running
        )
        self.start_button.pack(side="left", padx=3)
        tk.Button(
            toolbar, text="1 génération", width=13, command=self.single_generation
        ).pack(side="left", padx=3)
        self.speed_button = tk.Button(
            toolbar, text="Apprentissage ×1", width=17, command=self.cycle_speed
        )
        self.speed_button.pack(side="left", padx=3)
        self.vector_button = tk.Button(
            toolbar, text="Masquer vecteurs", width=16, command=self.toggle_vectors
        )
        self.vector_button.pack(side="left", padx=3)
        tk.Button(
            toolbar, text="Sauver le réseau", width=16, command=self.save_network
        ).pack(side="left", padx=3)
        tk.Button(
            toolbar, text="Charger un réseau", width=16, command=self.load_network
        ).pack(side="left", padx=3)
        tk.Button(
            toolbar, text="Nouvelle expérience", width=18, command=self.reset_experiment
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

    def toggle_running(self) -> None:
        self.running = not self.running
        self.start_button.configure(text="Pause" if self.running else "Démarrer")

    def single_generation(self) -> None:
        self.train_generations(1)

    def cycle_speed(self) -> None:
        self.training_speed_index = (self.training_speed_index + 1) % len(
            self.training_speeds
        )
        speed = self.training_speeds[self.training_speed_index]
        self.speed_button.configure(text=f"Apprentissage ×{speed}")

    def toggle_vectors(self) -> None:
        self.show_vectors = not self.show_vectors
        self.vector_button.configure(
            text="Masquer vecteurs" if self.show_vectors else "Afficher vecteurs"
        )

    def reset_experiment(self) -> None:
        if not self.messagebox.askyesno(
            "Nouvelle expérience",
            "Effacer l'apprentissage actuel et recréer la population ?",
        ):
            return
        self.running = False
        self.start_button.configure(text="Démarrer")
        self.evolution = Evolution(seed=random.randrange(1, 1_000_000))
        self.demo_network = self.evolution.best_network.copy()
        self.demo_world = CellWorld(
            generate_target_sequence(random.randrange(1, 1_000_000), TARGET_SEQUENCE_LENGTH)
        )

    def save_network(self) -> None:
        path = self.filedialog.asksaveasfilename(
            title="Sauvegarder le meilleur réseau",
            defaultextension=".json",
            filetypes=[("Réseau JSON", "*.json")],
            initialfile="meilleur_reseau_cellule.json",
        )
        if not path:
            return
        payload = {
            "generation": self.evolution.generation,
            "fitness": self.evolution.best_ever_fitness,
            "network": self.evolution.best_network.to_dict(),
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_network(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Charger un réseau",
            filetypes=[("Réseau JSON", "*.json"), ("Tous les fichiers", "*.*")],
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            network = NeuralNetwork.from_dict(payload["network"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.messagebox.showerror("Chargement impossible", str(exc))
            return
        self.demo_network = network
        self.evolution.best_network = network.copy()
        self.demo_world = CellWorld(generate_target_sequence(9090, TARGET_SEQUENCE_LENGTH))

    def train_generations(self, count: int) -> None:
        for _ in range(count):
            self.evolution.run_generation()
        # Le meilleur réseau contrôle immédiatement la cellule visible, sans
        # replacer celle-ci au centre : on observe le changement de comportement.
        self.demo_network = self.evolution.best_network.copy()

    def tick(self) -> None:
        # Animation de la meilleure politique actuellement connue.
        self.demo_world.step(self.demo_network, remember_trajectory=True)
        if self.demo_world.telemetry.captured:
            self.capture_flash = 12
        elif self.capture_flash > 0:
            self.capture_flash -= 1

        # L'entraînement se déroule par paquets entre deux séquences d'images.
        if self.running:
            self.frames_since_training += 1
            if self.frames_since_training >= 50:
                self.frames_since_training = 0
                speed = self.training_speeds[self.training_speed_index]
                self.train_generations(speed)

        self.draw()
        self.root.after(33, self.tick)

    def draw(self) -> None:
        self.canvas.delete("all")
        self.draw_world()
        self.draw_network()
        self.draw_learning_chart()

    def draw_world(self) -> None:
        c = self.canvas
        # Grille et limites du monde.
        for x in range(20, WORLD_WIDTH, 40):
            c.create_line(x, 20, x, WORLD_HEIGHT - 20, fill="#102536")
        for y in range(20, WORLD_HEIGHT, 40):
            c.create_line(20, y, WORLD_WIDTH - 20, y, fill="#102536")
        boundary_color = "#ff8a65" if self.demo_world.telemetry.wall_reaction.length() else "#4b6475"
        c.create_rectangle(
            WORLD_MARGIN,
            WORLD_MARGIN,
            WORLD_WIDTH - WORLD_MARGIN,
            WORLD_HEIGHT - WORLD_MARGIN,
            outline=boundary_color,
            width=3,
        )

        # Trajectoire récente de la cellule.
        if len(self.demo_world.trajectory) >= 2:
            coordinates: list[float] = []
            for point in self.demo_world.trajectory:
                coordinates.extend((point.x, point.y))
            c.create_line(*coordinates, fill="#718792", width=2, smooth=True)

        # Cible fixe et zone où elle est considérée comme atteinte.
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
        fill = "#ffffff" if self.capture_flash else "#ffd54f"
        c.create_oval(
            target.x - TARGET_RADIUS,
            target.y - TARGET_RADIUS,
            target.x + TARGET_RADIUS,
            target.y + TARGET_RADIUS,
            fill=fill,
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

        # Deux segments et leurs articulations.
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

        # Corps elliptique réellement orienté, dessiné comme un polygone fin.
        body_points: list[float] = []
        cosine = math.cos(self.demo_world.cell.angle)
        sine = math.sin(self.demo_world.cell.angle)
        for index in range(24):
            phi = 2.0 * math.pi * index / 24
            local_x = BODY_HALF_LENGTH * math.cos(phi)
            local_y = BODY_HALF_WIDTH * math.sin(phi)
            world_x = self.demo_world.cell.position.x + cosine * local_x - sine * local_y
            world_y = self.demo_world.cell.position.y + sine * local_x + cosine * local_y
            body_points.extend((world_x, world_y))
        c.create_polygon(
            *body_points, fill="#67d8a2", outline="#d8ffea", width=2, smooth=True
        )

        # Petit repère à l'avant : il distingue orientation et trajectoire.
        front = self.demo_world.cell.position + direction(self.demo_world.cell.angle) * 13
        c.create_oval(front.x - 3, front.y - 3, front.x + 3, front.y + 3, fill="#153d2d")

        if self.show_vectors:
            telemetry = self.demo_world.telemetry
            self.draw_arrow(middle_1, telemetry.force_segment_1, 95, "#ff9f43", "F1")
            self.draw_arrow(middle_2, telemetry.force_segment_2, 95, "#ff7f50", "F2")
            self.draw_arrow(
                self.demo_world.cell.position,
                telemetry.propulsion,
                100,
                "#46e66d",
                "PROP",
            )
            self.draw_arrow(
                self.demo_world.cell.position,
                telemetry.velocity,
                115,
                "#48a9ff",
                "V",
            )
            self.draw_arrow(
                self.demo_world.cell.position,
                telemetry.target_direction,
                70,
                "#cf8cff",
                "CIBLE",
                dash=(5, 3),
            )
            if telemetry.wall_reaction.length() > 0:
                self.draw_arrow(
                    self.demo_world.cell.position,
                    telemetry.wall_reaction,
                    35,
                    "#ff4b4b",
                    "PAROI",
                )

        speed = self.demo_world.cell.velocity.length()
        info = (
            f"Génération : {self.evolution.generation}\n"
            f"Cibles atteintes (démonstration) : {self.demo_world.captures}\n"
            f"Distance cible : {self.demo_world.telemetry.distance_to_target:6.1f}\n"
            f"Vitesse propre minimale : {BASE_SPEED:.2f}\n"
            f"Vitesse actuelle : {speed:.3f}\n"
            f"Orientation : {math.degrees(self.demo_world.cell.angle):6.1f}°\n"
            f"Récompense instantanée : {self.demo_world.telemetry.reward:+.3f}"
        )
        c.create_rectangle(29, 29, 300, 173, fill="#091722", outline="#315268")
        c.create_text(
            42,
            40,
            text=info,
            anchor="nw",
            fill="#d8e7ef",
            font=("Consolas", 10),
        )

        # Légende courte pour distinguer les vecteurs.
        legend = "F1/F2 : segments   PROP : propulsion   V : vitesse   CIBLE : direction"
        c.create_text(
            30,
            WORLD_HEIGHT - 10,
            text=legend,
            anchor="sw",
            fill="#a9bbc6",
            font=("Segoe UI", 9),
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
        length = vector.length()
        if length < 1e-7:
            return
        # La saturation évite qu'une pointe de force traverse tout l'écran.
        display_length = min(95.0, max(7.0, length * scale))
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

    @staticmethod
    def activity_color(value: float) -> str:
        value = clamp(value, -1.0, 1.0)
        if value >= 0:
            red = int(55 + 75 * value)
            green = int(90 + 115 * value)
            blue = int(115 + 140 * value)
        else:
            magnitude = -value
            red = int(90 + 150 * magnitude)
            green = int(85 - 35 * magnitude)
            blue = int(105 - 45 * magnitude)
        return f"#{red:02x}{green:02x}{blue:02x}"

    def draw_network(self) -> None:
        c = self.canvas
        panel_left = 800
        panel_right = CANVAS_WIDTH - 12
        panel_top = 18
        panel_bottom = 482
        c.create_rectangle(
            panel_left,
            panel_top,
            panel_right,
            panel_bottom,
            fill="#0c1722",
            outline="#38566b",
            width=2,
        )
        c.create_text(
            (panel_left + panel_right) / 2,
            38,
            text="RÉSEAU DU MEILLEUR INDIVIDU",
            fill="#e9f4fa",
            font=("Segoe UI", 12, "bold"),
        )
        champion = "nouveau champion" if self.evolution.last_champion_changed else "champion conservé"
        c.create_text(
            (panel_left + panel_right) / 2,
            59,
            text=f"6 entrées → 6 neurones → 2 moteurs — {champion}",
            fill="#82a6bc",
            font=("Segoe UI", 9),
        )

        input_x, hidden_x, output_x = 842, 1010, 1162
        input_y = [105 + index * 57 for index in range(NETWORK_INPUTS)]
        hidden_y = [105 + index * 57 for index in range(NETWORK_HIDDEN)]
        output_y = [218, 335]
        network = self.demo_network
        telemetry = self.demo_world.telemetry

        # Connexions d'abord, afin que les neurones restent visibles au-dessus.
        for hidden_index, row in enumerate(network.weights_input_hidden):
            for input_index, weight in enumerate(row):
                self.draw_connection(
                    input_x,
                    input_y[input_index],
                    hidden_x,
                    hidden_y[hidden_index],
                    weight,
                )
        for output_index, row in enumerate(network.weights_hidden_output):
            for hidden_index, weight in enumerate(row):
                self.draw_connection(
                    hidden_x,
                    hidden_y[hidden_index],
                    output_x,
                    output_y[output_index],
                    weight,
                )

        for index, y in enumerate(input_y):
            value = telemetry.inputs[index]
            self.draw_neuron(input_x, y, value, self.INPUT_LABELS[index], label_side="left")
        for index, y in enumerate(hidden_y):
            value = telemetry.hidden[index]
            self.draw_neuron(hidden_x, y, value, f"H{index + 1}", label_side="below")
        for index, y in enumerate(output_y):
            value = telemetry.outputs[index]
            self.draw_neuron(
                output_x, y, value, self.OUTPUT_LABELS[index], label_side="right"
            )

        c.create_text(
            (panel_left + panel_right) / 2,
            461,
            text="Bleu : poids positif   Rouge : poids négatif   Épaisseur : intensité",
            fill="#90a9b7",
            font=("Segoe UI", 8),
        )

    def draw_connection(
        self, x1: float, y1: float, x2: float, y2: float, weight: float
    ) -> None:
        if abs(weight) < 0.035:
            color = "#34424c"
        elif weight > 0:
            color = "#2d86d8"
        else:
            color = "#d64b4b"
        width = 0.5 + min(3.5, abs(weight) * 1.25)
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def draw_neuron(
        self,
        x: float,
        y: float,
        value: float,
        label: str,
        label_side: str,
    ) -> None:
        radius = 12
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=self.activity_color(value),
            outline="#e5f4fb",
            width=2,
        )
        self.canvas.create_text(
            x,
            y,
            text=f"{value:+.1f}",
            fill="white",
            font=("Consolas", 7, "bold"),
        )
        if label_side == "left":
            self.canvas.create_text(
                x - 17, y, text=label, anchor="e", fill="#c3d3dd", font=("Segoe UI", 8)
            )
        elif label_side == "right":
            self.canvas.create_text(
                x + 17, y, text=label, anchor="w", fill="#c3d3dd", font=("Segoe UI", 8)
            )
        else:
            self.canvas.create_text(
                x, y + 19, text=label, anchor="n", fill="#aabfcb", font=("Segoe UI", 7)
            )

    def draw_learning_chart(self) -> None:
        c = self.canvas
        left, top, right, bottom = 800, 495, CANVAS_WIDTH - 12, WORLD_HEIGHT
        c.create_rectangle(left, top, right, bottom, fill="#0c1722", outline="#38566b", width=2)
        c.create_text(
            (left + right) / 2,
            top + 19,
            text="ÉVOLUTION DE LA RÉCOMPENSE",
            fill="#e9f4fa",
            font=("Segoe UI", 11, "bold"),
        )

        history_best = self.evolution.history_best
        history_average = self.evolution.history_average
        chart_left, chart_top = left + 44, top + 48
        chart_right, chart_bottom = right - 18, bottom - 34
        c.create_line(chart_left, chart_bottom, chart_right, chart_bottom, fill="#607886")
        c.create_line(chart_left, chart_top, chart_left, chart_bottom, fill="#607886")

        if history_best:
            values = history_best + history_average
            minimum = min(values)
            maximum = max(values)
            if math.isclose(minimum, maximum):
                minimum -= 1.0
                maximum += 1.0

            def points(series: list[float]) -> list[float]:
                result: list[float] = []
                count = len(series)
                for index, value in enumerate(series):
                    fraction_x = index / max(1, count - 1)
                    fraction_y = (value - minimum) / (maximum - minimum)
                    result.extend(
                        (
                            chart_left + fraction_x * (chart_right - chart_left),
                            chart_bottom - fraction_y * (chart_bottom - chart_top),
                        )
                    )
                return result

            best_points = points(history_best)
            average_points = points(history_average)
            if len(best_points) >= 4:
                c.create_line(*best_points, fill="#5de578", width=2)
                c.create_line(*average_points, fill="#f1b34c", width=2)
            else:
                c.create_oval(
                    best_points[0] - 2,
                    best_points[1] - 2,
                    best_points[0] + 2,
                    best_points[1] + 2,
                    fill="#5de578",
                )
            c.create_text(
                chart_left - 5,
                chart_top,
                text=f"{maximum:.1f}",
                anchor="e",
                fill="#8fa6b3",
                font=("Consolas", 8),
            )
            c.create_text(
                chart_left - 5,
                chart_bottom,
                text=f"{minimum:.1f}",
                anchor="e",
                fill="#8fa6b3",
                font=("Consolas", 8),
            )

        best_text = (
            f"Meilleur : {self.evolution.best_ever_fitness:.2f}"
            if math.isfinite(self.evolution.best_ever_fitness)
            else "Meilleur : en attente"
        )
        c.create_text(
            chart_left,
            bottom - 17,
            text=best_text,
            anchor="w",
            fill="#5de578",
            font=("Segoe UI", 8, "bold"),
        )
        c.create_text(
            chart_right,
            bottom - 17,
            text="vert : meilleur   orange : moyenne",
            anchor="e",
            fill="#b5c5ce",
            font=("Segoe UI", 8),
        )

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Point d'entrée et test sans interface
# ---------------------------------------------------------------------------


def run_headless(generations: int) -> None:
    """Teste le moteur et l'apprentissage sans ouvrir de fenêtre graphique."""

    evolution = Evolution(seed=42)
    for _ in range(generations):
        result = evolution.run_generation()
        print(
            f"génération={evolution.generation:3d} "
            f"meilleur_génération={result.fitness:9.3f} "
            f"meilleur_global={evolution.best_ever_fitness:9.3f} "
            f"cibles={result.captures:2d} murs={result.wall_hits:3d}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cellule 2D à flagelle commandée par un réseau neuro-évolutif."
    )
    parser.add_argument(
        "--headless",
        type=int,
        metavar="GENERATIONS",
        help="teste ce nombre de générations sans interface graphique",
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