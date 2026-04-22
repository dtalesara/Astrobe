"""
Astrobe — HOPE Module
======================
Holistic Overview of Persistent Extremophiles.

Takes a CandidatePlanet's environment parameters and returns a ranked
list of extremophile organisms that could survive there.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from astrobe.models import CandidatePlanet


# ─── Survival Range ────────────────────────────────────────────────────────────

@dataclass
class SurvivalRange:
    min_val: float
    max_val: float
    optimal_min: Optional[float] = None
    optimal_max: Optional[float] = None

    def contains(self, value: float) -> bool:
        return self.min_val <= value <= self.max_val

    def is_optimal(self, value: float) -> bool:
        if self.optimal_min is not None and self.optimal_max is not None:
            return self.optimal_min <= value <= self.optimal_max
        return self.contains(value)


# ─── Extremophile ──────────────────────────────────────────────────────────────

@dataclass
class Extremophile:
    name: str
    classification: str
    domain: str
    habitat_type: str
    description: str
    real_examples: list[str]
    required_adaptations: list[str]
    temperature_c: SurvivalRange
    pressure_atm: SurvivalRange
    ph: SurvivalRange
    salinity_ppt: SurvivalRange
    radiation_gy: SurvivalRange
    water_activity: SurvivalRange
    requires_oxygen: Optional[bool] = None
    requires_liquid_water: bool = True
    notes: str = ""


# ─── Database ──────────────────────────────────────────────────────────────────

EXTREMOPHILE_DB: list[Extremophile] = [

    Extremophile(
        name="Piezophile (Deep-Sea Barophile)",
        classification="Piezophile", domain="Bacteria", habitat_type="ocean",
        description="Thrives under extreme hydrostatic pressure in deep ocean trenches.",
        real_examples=["Halomonas titanicae", "Shewanella benthica", "Moritella yayanosii"],
        required_adaptations=[
            "High-pressure-adapted enzymes (piezozymes)",
            "Unsaturated/branched-chain fatty acids in cell membranes",
            "Pressure-regulated gene expression systems",
            "Compact, pressure-stable ribosomal structures",
        ],
        temperature_c =SurvivalRange(-2,  10,   2,   6),
        pressure_atm  =SurvivalRange(100, 1100, 400, 800),
        ph            =SurvivalRange(6.0, 8.5,  7.0, 8.0),
        salinity_ppt  =SurvivalRange(30,  40,   34,  36),
        radiation_gy  =SurvivalRange(0,   10),
        water_activity=SurvivalRange(0.98,1.0),
        requires_oxygen=None,
        notes="Found in Mariana Trench and other hadal zones (>6,000 m depth).",
    ),

    Extremophile(
        name="Hydrothermal Vent Chemolithotroph",
        classification="Thermophile / Chemolithotroph", domain="Bacteria", habitat_type="ocean",
        description="Oxidises inorganic compounds at deep-sea vents. Foundation of vent ecosystems.",
        real_examples=["Thermus aquaticus", "Aquifex aeolicus", "Nautilia profundicola"],
        required_adaptations=[
            "Chemosynthesis from H₂S, H₂, or Fe²⁺",
            "Thermostable proteins and heat-shock chaperones",
            "Sulphur metabolism pathways",
            "High-pressure-tolerant enzyme complexes",
        ],
        temperature_c =SurvivalRange(45,  121, 80,  105),
        pressure_atm  =SurvivalRange(150, 500, 200, 400),
        ph            =SurvivalRange(4.0, 9.0, 5.5, 7.5),
        salinity_ppt  =SurvivalRange(25,  45,  33,  38),
        radiation_gy  =SurvivalRange(0,   50),
        water_activity=SurvivalRange(0.97,1.0),
        requires_oxygen=False,
        notes="Key model for life on icy moons like Europa.",
    ),

    Extremophile(
        name="Halophile (Marine Extreme)",
        classification="Halophile", domain="Archaea", habitat_type="ocean",
        description="Requires high salt concentrations; uses compatible solutes to manage osmotic stress.",
        real_examples=["Halobacterium salinarum", "Haloarcula marismortui", "Dunaliella salina"],
        required_adaptations=[
            "Acidic proteome stable in high-salt environments",
            "Accumulation of compatible solutes (ectoine, betaine)",
            "Salt-in strategy: high intracellular KCl",
            "Specialised ion-transport proteins (Na⁺/H⁺ antiporters)",
        ],
        temperature_c =SurvivalRange(15,  55,  35,  50),
        pressure_atm  =SurvivalRange(1,   200, 1,   50),
        ph            =SurvivalRange(6.0, 9.0, 7.0, 8.5),
        salinity_ppt  =SurvivalRange(150, 380, 200, 350),
        radiation_gy  =SurvivalRange(0,   5000),
        water_activity=SurvivalRange(0.60,0.90),
        requires_oxygen=None,
        notes="Dead Sea and salt flats are primary habitats.",
    ),

    Extremophile(
        name="Hyperthermophile",
        classification="Hyperthermophile", domain="Archaea", habitat_type="volcano",
        description="Grows optimally above 80 °C; some survive past 121 °C.",
        real_examples=["Pyrolobus fumarii", "Methanopyrus kandleri", "Ignicoccus hospitalis"],
        required_adaptations=[
            "Reverse gyrase supercoils DNA for thermal stability",
            "Thermosome chaperones for correct protein folding",
            "Ether-linked lipids (archaeol) in membranes",
            "Heat-stable DNA polymerases",
        ],
        temperature_c =SurvivalRange(80,  122, 95,  115),
        pressure_atm  =SurvivalRange(1,   400, 1,   50),
        ph            =SurvivalRange(3.0, 9.0, 5.0, 8.0),
        salinity_ppt  =SurvivalRange(0,   50,  10,  35),
        radiation_gy  =SurvivalRange(0,   80),
        water_activity=SurvivalRange(0.90,1.0),
        requires_oxygen=None,
        notes="Found in volcanic vents, hot springs, and geothermal pools.",
    ),

    Extremophile(
        name="Acidophile",
        classification="Acidophile", domain="Bacteria / Archaea", habitat_type="volcano",
        description="Thrives at very low pH; maintains near-neutral internal pH.",
        real_examples=["Acidithiobacillus ferrooxidans", "Picrophilus torridus", "Sulfolobus acidocaldarius"],
        required_adaptations=[
            "Proton-pumping ATPases maintain internal pH near neutral",
            "Highly impermeable cell membranes prevent proton influx",
            "Acid-stable surface proteins with extra disulfide bonds",
            "Cytoplasm buffering using glutamate and lysine",
        ],
        temperature_c =SurvivalRange(0,   80,  45,  65),
        pressure_atm  =SurvivalRange(1,   50,  1,   5),
        ph            =SurvivalRange(0.0, 5.0, 1.0, 3.5),
        salinity_ppt  =SurvivalRange(0,   30,  0,   10),
        radiation_gy  =SurvivalRange(0,   50),
        water_activity=SurvivalRange(0.85,1.0),
        requires_oxygen=None,
        notes="Acid mine drainage and volcanic sulphur springs are typical habitats.",
    ),

    Extremophile(
        name="Alkaliphile",
        classification="Alkaliphile", domain="Bacteria", habitat_type="volcano",
        description="Grows optimally at pH 9–11; found in soda lakes and alkaline hydrothermal systems.",
        real_examples=["Natronobacterium gregoryi", "Bacillus alcalophilus", "Thioalkalivibrio"],
        required_adaptations=[
            "Acidic cell surface to buffer alkaline environment",
            "Na⁺/H⁺ antiporters for internal pH homeostasis",
            "ATP synthase adapted to sodium-motive force",
            "Alkaline-stable enzymes",
        ],
        temperature_c =SurvivalRange(4,   65,  30,  45),
        pressure_atm  =SurvivalRange(1,   50,  1,   5),
        ph            =SurvivalRange(8.5, 12.5,9.5, 11.0),
        salinity_ppt  =SurvivalRange(0,   200, 20,  100),
        radiation_gy  =SurvivalRange(0,   20),
        water_activity=SurvivalRange(0.75,1.0),
        requires_oxygen=None,
        notes="Lake Natron (Tanzania) and Mono Lake (California) are classic habitats.",
    ),

    Extremophile(
        name="Psychrophile (Cryophile)",
        classification="Psychrophile", domain="Bacteria / Archaea / Eukarya", habitat_type="ice",
        description="Grows optimally below 15 °C; many active near −20 °C in brine channels.",
        real_examples=["Polaromonas vacuolata", "Chlamydomonas nivalis", "Psychrobacter arcticus"],
        required_adaptations=[
            "Cold-active enzymes with flexible active sites",
            "Antifreeze proteins prevent ice crystal formation",
            "High unsaturated fatty acid content keeps membranes fluid",
            "Cryoprotectants (glycerol, trehalose) lower freezing point",
        ],
        temperature_c =SurvivalRange(-20, 15,  -5,  10),
        pressure_atm  =SurvivalRange(1,   400, 1,   50),
        ph            =SurvivalRange(5.0, 9.0, 6.5, 8.0),
        salinity_ppt  =SurvivalRange(0,   200, 30,  80),
        radiation_gy  =SurvivalRange(0,   200),
        water_activity=SurvivalRange(0.70,1.0),
        requires_oxygen=None,
        notes="Arctic/Antarctic sea ice brine channels; subglacial lakes like Lake Vostok.",
    ),

    Extremophile(
        name="Subglacial Lithoautotroph",
        classification="Psychrophile / Chemolithotroph", domain="Bacteria", habitat_type="ice",
        description="Survives under kilometres of ice in total darkness, oxidising bedrock minerals.",
        real_examples=["Candidatus Methylobacter", "Desulfosporosinus lacus", "Lake Whillans microbiome"],
        required_adaptations=[
            "Chemolithotrophy: energy from Fe²⁺, Mn²⁺, or H₂ from rock-water interactions",
            "Ultralow metabolic rates — near-dormancy for long periods",
            "Efficient nutrient scavenging in near-zero nutrient environments",
            "Cold-active enzymes functional near 0 °C",
        ],
        temperature_c =SurvivalRange(-5,  4,   0,   2),
        pressure_atm  =SurvivalRange(200, 600, 300, 400),
        ph            =SurvivalRange(6.0, 9.0, 7.0, 8.5),
        salinity_ppt  =SurvivalRange(0,   50,  0,   20),
        radiation_gy  =SurvivalRange(0,   5),
        water_activity=SurvivalRange(0.97,1.0),
        requires_oxygen=False,
        notes="Critical analogue for potential life in Europa's subsurface ocean.",
    ),

    Extremophile(
        name="Xerophile (Desert Desiccation Specialist)",
        classification="Xerophile", domain="Bacteria / Fungi", habitat_type="desert",
        description="Thrives in extremely dry conditions using cryptobiosis — near complete metabolic suspension.",
        real_examples=["Chroococcidiopsis sp.", "Xeromyces bisporus", "Artemia salina"],
        required_adaptations=[
            "Trehalose accumulation forms a protective glass around biomolecules",
            "Cryptobiosis: suspended animation with near-zero metabolic rate",
            "Desiccation-tolerant LEA proteins protect membranes",
            "Efficient DNA repair machinery for desiccation-caused strand breaks",
        ],
        temperature_c =SurvivalRange(-15, 65,  20,  45),
        pressure_atm  =SurvivalRange(1,   10,  1,   2),
        ph            =SurvivalRange(4.0, 10.0,6.0, 8.5),
        salinity_ppt  =SurvivalRange(0,   300, 0,   50),
        radiation_gy  =SurvivalRange(0,   15000),
        water_activity=SurvivalRange(0.50,0.85),
        requires_oxygen=None,
        requires_liquid_water=False,
        notes="Atacama Desert is the best terrestrial analogue for Martian surface.",
    ),

    Extremophile(
        name="Endolithic Cyanobacterium",
        classification="Photolithotroph / UV-tolerant", domain="Bacteria", habitat_type="desert",
        description="Lives inside translucent rocks, sheltered from UV while accessing scattered light.",
        real_examples=["Chroococcidiopsis thermalis", "Gloeocapsa sp.", "Nostoc commune"],
        required_adaptations=[
            "Rock-dwelling lifestyle for UV shielding and moisture retention",
            "Scytonemin UV-absorbing sunscreens",
            "Photosystem adapted to low-light penetrating translucent rock",
            "Nitrogen fixation for nutrient-poor environments",
        ],
        temperature_c =SurvivalRange(-10, 60,  15,  35),
        pressure_atm  =SurvivalRange(1,   5,   1,   2),
        ph            =SurvivalRange(4.0, 10.0,6.5, 8.0),
        salinity_ppt  =SurvivalRange(0,   50,  0,   20),
        radiation_gy  =SurvivalRange(0,   12000),
        water_activity=SurvivalRange(0.55,0.95),
        requires_oxygen=True,
        notes="Found in Antarctic dry valleys and Atacama Desert.",
    ),

    Extremophile(
        name="Radioresistant Extremophile",
        classification="Radiophile / Polyextremophile", domain="Bacteria", habitat_type="desert",
        description="Survives radiation doses thousands of times the lethal human dose.",
        real_examples=["Deinococcus radiodurans", "Rubrobacter radiotolerans", "Thermococcus gammatolerans"],
        required_adaptations=[
            "Hyper-efficient DNA repair: reassembles shattered chromosomes within hours",
            "High intracellular Mn²⁺:Fe²⁺ ratio protects proteins from oxidative damage",
            "Dense nucleoid structure limits DNA strand break propagation",
            "Redundant genome copies allow cross-template repair",
        ],
        temperature_c =SurvivalRange(-15, 50,  20,  37),
        pressure_atm  =SurvivalRange(1,   20,  1,   5),
        ph            =SurvivalRange(4.0, 10.0,6.5, 8.5),
        salinity_ppt  =SurvivalRange(0,   80,  0,   20),
        radiation_gy  =SurvivalRange(0,   30000),
        water_activity=SurvivalRange(0.60,1.0),
        requires_oxygen=None,
        requires_liquid_water=False,
        notes="Also survives vacuum and desiccation — relevant for space panspermia models.",
    ),
]


# ─── Analyser ──────────────────────────────────────────────────────────────────

@dataclass
class HOPEResult:
    organism: Extremophile
    score: float
    status: str           # "viable" | "marginal" | "unlikely"
    matched_conditions: list[str]
    limiting_factors: list[str]


def _score_param(value: float, sr: SurvivalRange, label: str):
    """
    Score a single parameter. Returns (score, matched_note, limiting_note, fatal).
    fatal=True means the value is outside the absolute survival range — hard disqualifier.
    """
    if sr.contains(value):
        note = f"{label}: {value} ({'optimal' if sr.is_optimal(value) else 'within tolerance'})"
        return (1.0 if sr.is_optimal(value) else 0.7), note, None, False
    # Outside survival range — this is a hard biological limit
    return 0.0, None, f"{label}: {value} (survival range {sr.min_val}–{sr.max_val})", True


def run_hope(planet: CandidatePlanet,
             db: list[Extremophile] = None) -> list[HOPEResult]:
    """
    Match a planet's environment against all extremophile organisms.

    An organism is 'unlikely' if ANY parameter falls outside its absolute
    survival range — biology does not average out fatal conditions.
    Status thresholds apply only to organisms with no fatal parameters:
      viable   >= 0.75 composite
      marginal >= 0.45 composite
    """
    if db is None:
        db = EXTREMOPHILE_DB

    results = []
    for org in db:
        scores, matched, limiting = [], [], []
        fatal_params = []

        # Hard disqualifier: liquid water requirement
        if org.requires_liquid_water and not planet.has_liquid_water:
            limiting.append("No liquid water (organism strictly requires it)")
            fatal_params.append("liquid_water")

        params = [
            (planet.temperature_c,  org.temperature_c,  "Temperature (°C)"),
            (planet.pressure_atm,   org.pressure_atm,   "Pressure (atm)"),
            (planet.ph,             org.ph,             "pH"),
            (planet.salinity_ppt,   org.salinity_ppt,   "Salinity (ppt)"),
            (planet.radiation_gy,   org.radiation_gy,   "Radiation (Gy/yr)"),
            (planet.water_activity, org.water_activity, "Water activity"),
        ]

        for val, sr, label in params:
            s, m, l, fatal = _score_param(val, sr, label)
            scores.append(s)
            if m: matched.append(m)
            if l: limiting.append(l)
            if fatal: fatal_params.append(label)

        # If any parameter is outside survival range → cannot be viable or marginal
        if fatal_params:
            status = "unlikely"
            # Score reflects how many parameters are within range, but capped low
            composite = min(sum(scores) / len(scores), 0.40)
        else:
            composite = sum(scores) / len(scores)
            status = "viable" if composite >= 0.75 else \
                     "marginal" if composite >= 0.45 else "unlikely"

        results.append(HOPEResult(
            organism=org,
            score=round(composite, 3),
            status=status,
            matched_conditions=matched,
            limiting_factors=limiting,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
