# 🌌 Astrobe — Unified Astrobiology Analysis Pipeline  
**Astrobe** is a Python pipeline for assessing the possibility of life on any planet or moon — real, hypothetical, or exoplanetary. It combines four independent analysis modules into a single composite score and verdict.

```
Your Planet
    │
    ├─ [1] Astrobe Core   — astrophysical habitability + Random Forest life prediction
    ├─ [2] BioSearch      — planetary emergence, biosignature phase & detectability
    ├─ [3] HOPE           — which extremophile organisms could actually survive there
    └─ [4] DigitalDNA     — nucleotide-encoded ML model + evolutionary simulation
           │
           └──▶  Composite Score  +  Verdict
```

---

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Command-Line Usage](#command-line-usage)
- [Using Astrobe in Python](#using-astrobe-in-python)
- [Planet Parameters — Full Reference](#planet-parameters--full-reference)
- [Understanding the Output](#understanding-the-output)
- [The Four Modules](#the-four-modules)
- [Built-in Presets](#built-in-presets)
- [Adding Your Own Planet to Presets](#adding-your-own-planet-to-presets)
- [Project Structure](#project-structure)

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/your-username/astrobe.git
cd astrobe

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package (editable mode — recommended for development)
pip install -e .
```

> No internet connection is required to run the pipeline. All modules run locally.

---

## Quickstart

### Run a built-in preset

```bash
python -m astrobe --preset earth
python -m astrobe --preset mars
python -m astrobe --preset europa_ocean
python -m astrobe --preset enceladus
```

### Run interactively — enter your own planet step by step

```bash
python -m astrobe --interactive
```

You will be prompted for each parameter with sensible defaults.

### List all built-in planets

```bash
python -m astrobe --list-presets
```

---

## Command-Line Usage

```
python -m astrobe [options]
```

### Mode flags (choose one)

| Flag | Description |
|------|-------------|
| `--preset NAME` | Run a built-in planet preset |
| `--interactive` | Build a custom planet interactively |
| `--list-presets` | Print all available presets and exit |
| `--name "My Planet"` | Define a custom planet via flags (see below) |

### Custom planet flags

These are used together with `--name`:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--name` | str | — | Planet name (**required** for custom mode) |
| `--location` | str | Unknown | Location or system description |
| `--description` | str | — | Brief description |
| `--temp` | float °C | 15.0 | Mean surface temperature |
| `--has-atmosphere` | flag | True | Planet has an atmosphere |
| `--no-atmosphere` | flag | — | Planet has no atmosphere |
| `--has-liquid-water` | flag | False | Confirmed liquid water present |
| `--no-liquid-water` | flag | — | No confirmed liquid water |
| `--atm-type` | str | neutral | `reducing`, `neutral`, or `oxidizing` |
| `--age` | float Ga | 4.5 | Planet age in billions of years |
| `--geothermal` | float 0–10 | 5.0 | Geothermal activity index |
| `--tectonics` | float 0–10 | 5.0 | Tectonic activity index |
| `--silica` | float 0–10 | 5.0 | Silica content index |
| `--pressure` | float atm | 1.0 | Surface atmospheric pressure |
| `--ph` | float 0–14 | 7.0 | Surface/ocean pH |
| `--salinity` | float ppt | 35.0 | Salinity in parts per thousand |
| `--radiation-dose` | float Gy/yr | 0.003 | Annual radiation dose (Grays) |
| `--water-activity` | float 0–1 | 0.99 | Water activity (1.0 = pure water) |
| `--star-type` | str | G | Host star type: M, K, G, or F |
| `--star-luminosity` | float | 1.0 | Host star luminosity (solar = 1.0) |
| `--star-radiation` | float 0–10 | 3.0 | Stellar radiation/flare activity index |
| `--orbital-distance` | float AU | 1.0 | Distance from host star in AU |

### Pipeline options

| Flag | Description |
|------|-------------|
| `--no-evolution` | Skip DigitalDNA evolutionary simulation (runs faster) |
| `--evo-generations N` | Number of evolutionary generations (default: 20) |
| `--show-all-organisms` | Show 'unlikely' organisms in HOPE results |

### Examples

```bash
# Kepler-22b with known parameters
python -m astrobe --name "Kepler-22b" \
                  --location "Kepler-22 system, ~620 ly" \
                  --temp -11 \
                  --has-atmosphere \
                  --age 4.0 \
                  --geothermal 5.0 --tectonics 4.5 --silica 5.0 \
                  --pressure 1.0 --ph 7.0 --salinity 35 \
                  --radiation-dose 0.005 --water-activity 0.80 \
                  --star-type G --star-luminosity 0.79 \
                  --star-radiation 3.5 --orbital-distance 0.849

# A hypothetical hot super-Earth
python -m astrobe --name "Hot Super-Earth" \
                  --temp 85 --has-atmosphere --no-liquid-water \
                  --geothermal 8 --tectonics 7 \
                  --star-luminosity 1.2 --orbital-distance 0.6

# Fast run without evolution
python -m astrobe --preset europa_ocean --no-evolution

# Show all extremophile results including unlikely ones
python -m astrobe --preset enceladus --show-all-organisms
```

---

## Using Astrobe in Python

For scripting, notebooks, or integration with your own code:

```python
from astrobe import CandidatePlanet, Star, analyse_planet
from astrobe.reporter import print_report

# Define your planet
planet = CandidatePlanet(
    name="My Exoplanet",
    location="HD 40307 system",
    star=Star(star_type="K", luminosity=0.23, radiation=4.0),
    orbital_distance_au=0.13,
    temperature_c=18,
    has_atmosphere=True,
    has_liquid_water=True,
    atmosphere_type="neutral",
    age_ga=6.0,
    geothermal=6.0,
    tectonics=5.0,
    silica=5.5,
    pressure_atm=1.2,
    ph=7.3,
    salinity_ppt=40.0,
    radiation_gy=0.01,
    water_activity=0.97,
)

# Run the full pipeline
report = analyse_planet(planet, verbose=True)

# Print formatted results
print_report(report)

# Access individual results
print(f"Composite score : {report.composite_score}")
print(f"Verdict         : {report.verdict}")
print(f"Habitability    : {report.astrobe.habitability_score}")
print(f"In habitable zone: {report.astrobe.in_habitable_zone}")
print(f"BioSearch phase : {report.biosearch.phase_label}")
print(f"DigitalDNA class: {report.digital_dna.top_class}")
print(f"Viable organisms: {len(report.viable_organisms)}")

# Iterate over viable extremophile organisms
for result in report.viable_organisms:
    print(f"  {result.organism.name}: {result.score:.1%}")
```

### Using a built-in preset in Python

```python
from astrobe.presets import PRESETS
from astrobe import analyse_planet
from astrobe.reporter import print_report

report = analyse_planet(PRESETS["europa_ocean"])
print_report(report)
```

### Comparing multiple planets

```python
from astrobe.presets import PRESETS
from astrobe import analyse_planet
from astrobe.models import CandidatePlanet, Star

planets = [
    PRESETS["earth"],
    PRESETS["mars"],
    PRESETS["europa_ocean"],
    CandidatePlanet(
        name="My Custom Planet",
        temperature_c=5,
        has_atmosphere=True,
        has_liquid_water=True,
        age_ga=3.0,
        star=Star("G", 0.9, 3.0),
        orbital_distance_au=0.95,
    ),
]

results = [(p.name, analyse_planet(p, run_evolution=False)) for p in planets]
results.sort(key=lambda x: x[1].composite_score, reverse=True)

print(f"\n{'Planet':<25} {'Score':>7}  Verdict")
print("-" * 50)
for name, r in results:
    print(f"{name:<25} {r.composite_score:>7.3f}  {r.verdict}")
```

---

## Planet Parameters — Full Reference

### Star

| Field | Type | Description | Example values |
|-------|------|-------------|----------------|
| `star_type` | str | Spectral class | `"M"`, `"K"`, `"G"`, `"F"` |
| `luminosity` | float | Solar luminosity units (Sun = 1.0) | `0.04` (M-dwarf), `1.0` (Sun), `2.5` (F-star) |
| `radiation` | float 0–10 | Flare/particle radiation activity index | `8.0` (active M-dwarf), `3.0` (Sun-like) |

### CandidatePlanet

#### Astrophysical

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | — | Planet name |
| `location` | str | `"Unknown"` | System or location description |
| `description` | str | `""` | Free-text description |
| `star` | Star | None | Host star (see above) |
| `orbital_distance_au` | float | None | Distance from star in Astronomical Units |
| `near_agn` | bool | None | Near an Active Galactic Nucleus (auto-detected if None) |

#### Climate / Surface

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature_c` | float | `15.0` | Mean surface temperature in °C |
| `has_atmosphere` | bool | `True` | Whether the planet has an atmosphere |
| `has_liquid_water` | bool | `True` | Whether liquid water is confirmed |
| `atmosphere_type` | str | `"neutral"` | `"reducing"`, `"neutral"`, or `"oxidizing"` |
| `composition` | ChemicalComposition | None | Atmospheric gas percentages (optional) |

#### Geological (all 0–10 index)

| Field | Type | Default | Description | Low end | High end |
|-------|------|---------|-------------|---------|----------|
| `age_ga` | float | `4.5` | Planet age in billions of years (Ga) | Young (< 1 Ga) | Old (> 8 Ga) |
| `geothermal` | float | `5.0` | Geothermal / internal heat activity | Cold, dead core | Highly active (like Io) |
| `tectonics` | float | `5.0` | Tectonic plate activity | Stagnant lid | Very active |
| `silica` | float | `5.0` | Silica mineral content | Silica-poor | Silica-rich |

#### Environmental (used by HOPE extremophile matching)

| Field | Type | Default | Description | Typical values |
|-------|------|---------|-------------|----------------|
| `pressure_atm` | float | `1.0` | Atmospheric/surface pressure in atm | Mars: 0.006 · Earth: 1.0 · Europa ocean: 1300 |
| `ph` | float | `7.0` | pH of surface liquid or ocean | Acid: 2–5 · Neutral: 6–8 · Alkaline: 9–12 |
| `salinity_ppt` | float | `35.0` | Salinity in parts per thousand | Fresh: 0–5 · Ocean: 35 · Dead Sea: 340 |
| `radiation_gy` | float | `0.003` | Annual radiation dose in Grays | Earth surface: 0.003 · Mars: 0.3 · Europa surface: 5400 |
| `water_activity` | float | `0.99` | Available water (0=bone dry, 1=pure water) | Pure water: 1.0 · Earth ocean: 0.98 · Brine: 0.5 |

### Adding atmospheric composition (optional)

```python
from astrobe.models import ChemicalComposition, CandidatePlanet

planet = CandidatePlanet(
    name="Earth-like",
    composition=ChemicalComposition({
        "N2": 78.0,
        "O2": 21.0,
        "CO2": 0.04,
        "H2O": 0.4,
        "Ar":  0.56,
    }),
    # ... other fields
)
```

> **Note:** Percentages must sum to approximately 100%.

---

## Understanding the Output

### Composite Score and Verdict

The final composite score (0–1) combines all four modules:

| Module | Weight | What it measures |
|--------|--------|-----------------|
| Astrobe Core | 30% | Physical habitability, orbital position, RF life prediction |
| BioSearch | 25% | Geological life emergence potential |
| DigitalDNA | 25% | DNA-encoded ML habitability classification |
| HOPE | 20% | Fraction of extremophile organisms that could survive |

**Verdict thresholds:**

| Score | Verdict |
|-------|---------|
| ≥ 0.70 | 🟢 STRONG CANDIDATE |
| ≥ 0.50 | 🟢 PROMISING |
| ≥ 0.30 | 🟡 MARGINAL |
| ≥ 0.15 | 🟡 UNLIKELY |
| < 0.15 | 🔴 INHOSPITABLE |

### BioSearch Life Phases

| Phase | Label | What it means |
|-------|-------|---------------|
| 0 | Prebiotic | Conditions too hostile; no life expected |
| 1 | Hydrothermal Biosphere | Early chemosynthetic life possible (like early Earth) |
| 2 | Marine Biosphere | Aquatic life possible; oceans present |
| 3 | Advanced / Oxygen-influenced Biosphere | Complex life possible; oxidising atmosphere |

### DigitalDNA Classes

| Class | Meaning |
|-------|---------|
| Uninhabitable | Strong prediction against life |
| Marginal | Some conditions present, most missing |
| Potentially Habitable | Several key conditions met |
| Habitable | Strong prediction for life |

### HOPE Organism Status

| Status | Meaning |
|--------|---------|
| ✔ Viable | All survival parameters within the organism's absolute range |
| ~ Marginal | Most parameters within range; some outside optimal zone |
| ✘ Unlikely | One or more parameters outside absolute survival limits |

> An organism cannot be viable if even one parameter (e.g., temperature, pressure, radiation) falls outside its known survival range. Biology does not average out lethal conditions.

---

## The Four Modules

### Module 1 — Astrobe Core (`core.py`)

The astrophysical backbone. Takes the planet's physical properties and outputs:
- **Habitability score** (0–1): weighted combination of atmosphere, temperature, liquid water, and orbital position
- **Random Forest prediction**: trained on atmosphere / temperature / liquid water data; outputs a life probability
- **Stellar habitable zone**: calculates inner and outer HZ bounds in AU from the host star's luminosity
- **AGN proximity flag**: flags if the radiation environment suggests proximity to an Active Galactic Nucleus
- **Future temperature projections**: linear warming projection at +10 and +50 years

### Module 2 — BioSearch (`biosearch.py`)

Geological and biosignature pipeline. Evaluates:
- **Emergence score**: how strongly geothermal activity, ocean presence, tectonic index, and silica content favour life's origin
- **Life phase** (0–3): based on emergence score and planet age-driven atmosphere evolution
- **Preservation probability**: whether biosignatures would have survived to be detectable
- **Detectability**: signal strength relative to stellar noise
- **Atmosphere evolution**: reducing → neutral → oxidizing, driven by `age_ga`

### Module 3 — HOPE (`hope.py`)

Extremophile compatibility engine. Tests eleven real organism types against six environmental parameters:

| Organism | Type | Key environment |
|----------|------|-----------------|
| Piezophile | Pressure specialist | Deep ocean, > 100 atm |
| Hydrothermal Vent Chemolithotroph | Heat + chemistry | 45–121°C, deep sea vents |
| Halophile | Salt specialist | 150–380 ppt salinity |
| Hyperthermophile | Extreme heat | 80–122°C |
| Acidophile | Low pH | pH 0–5 |
| Alkaliphile | High pH | pH 8.5–12.5 |
| Psychrophile | Cold specialist | −20 to +15°C |
| Subglacial Lithoautotroph | Under-ice chemist | −5 to +4°C, high pressure |
| Xerophile | Desiccation specialist | Water activity 0.50–0.85 |
| Endolithic Cyanobacterium | Rock-dwelling | UV-shielded, near-surface |
| Radioresistant Extremophile | Radiation tolerant | Up to 30,000 Gy/yr |

### Module 4 — DigitalDNA (`digital_dna.py`)

The most novel module. Encodes planetary parameters as a DNA sequence (A, C, T, G), then:
1. **Forward pass**: each nucleotide base has a learned numerical weight; the sequence score is a weighted sum — equivalent to a linear neuron
2. **Epigenetic modulation**: nine environmental factors (temperature, water, atmosphere, radiation, etc.) are applied as learnable multipliers that shift the prediction — the same sequence produces different outputs in different planetary contexts
3. **Four-class prediction**: Uninhabitable → Marginal → Potentially Habitable → Habitable
4. **Evolutionary simulation**: a genetic algorithm evolves a population of sequences toward maximum habitability, discovering which nucleotide compositions are optimal for the given environment

---

## Built-in Presets

| Key | Planet | System | Notes |
|-----|--------|--------|-------|
| `earth` | Earth | Solar System | Reference standard |
| `mars` | Mars | Solar System | Cold, thin atmosphere, no surface water |
| `europa_ocean` | Europa subsurface ocean | Jupiter system | Tidal heating, possible vents |
| `enceladus` | Enceladus subsurface ocean | Saturn system | Confirmed organics and H₂ in plumes |
| `kepler_442b` | Kepler-442b (analog) | ~1,200 ly | Super-Earth in HZ of K-type star |
| `proxima_b` | Proxima Centauri b (analog) | 4.2 ly | Closest known exoplanet; M-dwarf flare risk |

---

## Adding Your Own Planet to Presets

Open `astrobe/presets.py` and add an entry to the `PRESETS` dictionary:

```python
PRESETS["my_planet"] = CandidatePlanet(
    name="My Planet",
    location="XYZ system",
    description="A planet I want to test.",
    star=Star("K", luminosity=0.4, radiation=4.0),
    orbital_distance_au=0.55,
    temperature_c=8,
    has_atmosphere=True,
    has_liquid_water=True,
    atmosphere_type="neutral",
    age_ga=3.5,
    geothermal=6.0,
    tectonics=5.0,
    silica=5.5,
    pressure_atm=1.1,
    ph=7.2,
    salinity_ppt=32.0,
    radiation_gy=0.008,
    water_activity=0.98,
)
```

Then run it with:

```bash
python -m astrobe --preset my_planet
```

---

## Project Structure

```
astrobe/
├── astrobe/
│   ├── __init__.py        — Public API
│   ├── __main__.py        — CLI entry point (python -m astrobe)
│   ├── models.py          — CandidatePlanet, Star, ChemicalComposition
│   ├── core.py            — Astrobe Core: RF predictor, habitability, stellar zone
│   ├── biosearch.py       — BioSearch: emergence, phase, detectability
│   ├── hope.py            — HOPE: extremophile database and compatibility scorer
│   ├── digital_dna.py     — DigitalDNA: nucleotide-encoded ML + evolution
│   ├── pipeline.py        — Unified pipeline: runs all modules, returns AstrobeReport
│   ├── presets.py         — Built-in planet definitions
│   └── reporter.py        — Terminal output formatter
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Citation

If you use Astrobe in academic work, please cite:

```
Astrobe: Unified Astrobiology Analysis Pipeline v1.0
DigitalDNA: A Nucleotide-Encoded Machine Learning Framework
for Multi-Dimensional Astrobiological Habitability Assessment
Preprint, 2025.
```

---

## License

MIT License — free to use, modify, and distribute with attribution.
