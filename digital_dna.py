"""
Astrobe — DigitalDNA Module
============================
A biologically-grounded machine learning model that encodes planetary
features as DNA sequences, applies epigenetic modulation for environmental
context, and uses evolutionary algorithms to discover optimal biosignature
profiles.

Core concepts:
  - Each of A, C, T, G carries a learnable weight (analogous to neuron weights)
  - Epigenetic factors shift the output based on planetary context
  - L-BFGS-B optimization learns both sets of weights from training data
  - Genetic algorithm evolves populations of sequences toward habitability
"""

from __future__ import annotations
import random
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from typing import Optional

from astrobe.models import CandidatePlanet


BASES = ["A", "C", "T", "G"]

HABITABILITY_CLASSES = [
    "Uninhabitable",
    "Marginal",
    "Potentially Habitable",
    "Habitable",
]


# ─── DNA Encoding ──────────────────────────────────────────────────────────────

def encode_planet_to_dna(planet: CandidatePlanet, length: int = 20) -> str:
    """
    Encode a planet's key parameters as a DNA sequence.

    Each parameter maps to a segment of the sequence:
      - Temperature zone  → A/T ratio
      - Water presence    → G content
      - Atmosphere        → C content
      - Geothermal        → G/A mix
    """
    seq = []

    # Temperature segment (6 bases): hot → more G, cold → more A
    temp_norm = max(0.0, min(1.0, (planet.temperature_c + 20) / 70.0))
    for _ in range(6):
        r = random.random()
        if r < temp_norm * 0.5:
            seq.append("G")
        elif r < temp_norm:
            seq.append("C")
        elif r < temp_norm + 0.3:
            seq.append("T")
        else:
            seq.append("A")

    # Water segment (4 bases)
    water_bases = ["G", "G", "C", "T"] if planet.has_liquid_water else ["A", "A", "T", "A"]
    seq.extend(water_bases)

    # Atmosphere segment (4 bases)
    atm_bases = ["C", "G", "C", "T"] if planet.has_atmosphere else ["A", "T", "A", "A"]
    seq.extend(atm_bases)

    # Geothermal segment (remaining)
    geo_norm = planet.geothermal / 10.0
    remaining = length - len(seq)
    for _ in range(remaining):
        seq.append("G" if random.random() < geo_norm else "A")

    return "".join(seq[:length])


# ─── Forward Pass ──────────────────────────────────────────────────────────────

def process_base(base: str, weights: dict) -> float:
    return weights.get(base.upper(), 0.0)


def process_sequence(sequence: str, weights: dict) -> float:
    """Sum weighted contributions of all bases — the DigitalDNA forward pass."""
    return sum(process_base(b, weights) for b in sequence.upper())


def apply_epigenetics(raw: float, factors: dict, epi_weights: dict) -> float:
    """Apply learnable environmental modifiers to the raw sequence score."""
    modifier = sum(epi_weights.get(k, 0.0) * v for k, v in factors.items())
    return raw + modifier


def predict_probability(sequence: str, dna_weights: dict,
                         epi_factors: dict, epi_weights: dict) -> float:
    adjusted = apply_epigenetics(
        process_sequence(sequence, dna_weights), epi_factors, epi_weights
    )
    return float(1.0 / (1.0 + np.exp(-adjusted)))


def predict_class_probabilities(sequence: str, dna_weights: dict,
                                 epi_factors: dict, epi_weights: dict) -> dict:
    """Return probability distribution across 4 habitability classes."""
    adjusted = apply_epigenetics(
        process_sequence(sequence, dna_weights), epi_factors, epi_weights
    )
    raw_scores = np.array([
        max(0, 4.0 - adjusted),
        max(0, 3.0 - abs(adjusted - 1.5)),
        max(0, 3.0 - abs(adjusted - 2.5)),
        max(0, adjusted - 2.5),
    ])
    total = raw_scores.sum()
    probs = raw_scores / total if total > 0 else np.ones(4) / 4
    return dict(zip(HABITABILITY_CLASSES, probs.tolist()))


# ─── Training ──────────────────────────────────────────────────────────────────

def _cost_function(all_weights, sequences, expected, epi_list, epi_keys):
    dna_w = {"A": all_weights[0], "C": all_weights[1],
              "T": all_weights[2], "G": all_weights[3]}
    epi_w = {k: all_weights[4 + i] for i, k in enumerate(epi_keys)}
    return sum(
        (predict_probability(seq, dna_w, epi, epi_w) - exp) ** 2
        for seq, exp, epi in zip(sequences, expected, epi_list)
    )


def train(sequences: list[str], expected_outputs: list[float],
          epi_factors_list: list[dict], seed: int = 42):
    """
    Train DigitalDNA weights using L-BFGS-B optimisation.

    Returns (dna_weights, epigenetic_weights).
    """
    np.random.seed(seed)
    epi_keys = sorted({k for epi in epi_factors_list for k in epi})
    n_params = 4 + len(epi_keys)
    x0 = np.random.uniform(-1, 1, n_params)

    result = minimize(
        fun=_cost_function,
        x0=x0,
        args=(sequences, expected_outputs, epi_factors_list, epi_keys),
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )

    dna_w = {"A": result.x[0], "C": result.x[1],
             "T": result.x[2], "G": result.x[3]}
    epi_w = {k: result.x[4 + i] for i, k in enumerate(epi_keys)}
    return dna_w, epi_w, result.fun


# ─── Evolution ─────────────────────────────────────────────────────────────────

def reproduce(parent: str, mutation_rate: float = 0.05) -> str:
    """
    Produce a child sequence from a parent via point mutation.
    Each base is independently mutated with probability mutation_rate.
    """
    return "".join(
        random.choice(BASES) if random.random() < mutation_rate else b
        for b in parent
    )


def run_evolution(pop_size: int, seq_length: int, generations: int,
                  epi_factors: dict, dna_weights: dict, epi_weights: dict,
                  mutation_rate: float = 0.05,
                  verbose: bool = False) -> tuple[list[str], list[float]]:
    """
    Evolve a population of DNA sequences toward maximum habitability.

    Returns the final population and their fitness scores.
    """
    population = [
        "".join(random.choice(BASES) for _ in range(seq_length))
        for _ in range(pop_size)
    ]

    for gen in range(generations):
        fitness = [
            predict_probability(seq, dna_weights, epi_factors, epi_weights)
            for seq in population
        ]
        top_idx = np.argsort(fitness)[::-1][:pop_size // 2]
        top_seqs = [population[i] for i in top_idx]

        new_pop = []
        while len(new_pop) < pop_size:
            parent = random.choice(top_seqs)
            new_pop.append(reproduce(parent, mutation_rate))
        population = new_pop

        if verbose:
            print(f"  Gen {gen+1:>3}  best fitness: {max(fitness):.4f}")

    final_fitness = [
        predict_probability(seq, dna_weights, epi_factors, epi_weights)
        for seq in population
    ]
    return population, final_fitness


# ─── High-level wrapper ────────────────────────────────────────────────────────

@dataclass
class DigitalDNAResult:
    sequence: str
    class_probabilities: dict[str, float]
    top_class: str
    top_probability: float
    evolution_best_fitness: Optional[float] = None
    evolution_best_sequence: Optional[str] = None


def run_digital_dna(planet: CandidatePlanet,
                    dna_weights: Optional[dict] = None,
                    epi_weights: Optional[dict] = None,
                    run_evo: bool = True,
                    evo_generations: int = 20,
                    evo_pop_size: int = 50) -> DigitalDNAResult:
    """
    Run DigitalDNA analysis on a CandidatePlanet.

    If no pre-trained weights are provided, uses defaults derived from
    a small built-in training set of known habitable/uninhabitable worlds.
    """
    # Default weights if none provided (pre-tuned on inner solar system analogs)
    if dna_weights is None:
        dna_weights = {"A": -0.31, "C": 0.28, "T": -0.15, "G": 0.42}
    if epi_weights is None:
        epi_weights = {
            "temperature":    0.35,
            "atmosphere":     0.60,
            "liquid_water":   0.80,
            "geothermal":     0.20,
            "tectonics":      0.15,
            "silica":         0.10,
            "radiation":     -0.45,
            "water_activity": 0.50,
            "ph_neutral":     0.25,
        }

    # Encode planet if not already encoded
    if planet.dna_sequence is None:
        planet.dna_sequence = encode_planet_to_dna(planet)

    epi = planet.to_epigenetic_factors()
    class_probs = predict_class_probabilities(planet.dna_sequence, dna_weights, epi, epi_weights)
    top_class = max(class_probs, key=class_probs.get)

    best_fitness = None
    best_seq = None
    if run_evo:
        pop, fitness = run_evolution(
            pop_size=evo_pop_size,
            seq_length=len(planet.dna_sequence),
            generations=evo_generations,
            epi_factors=epi,
            dna_weights=dna_weights,
            epi_weights=epi_weights,
        )
        best_idx = int(np.argmax(fitness))
        best_fitness = fitness[best_idx]
        best_seq = pop[best_idx]

    return DigitalDNAResult(
        sequence=planet.dna_sequence,
        class_probabilities=class_probs,
        top_class=top_class,
        top_probability=class_probs[top_class],
        evolution_best_fitness=best_fitness,
        evolution_best_sequence=best_seq,
    )
