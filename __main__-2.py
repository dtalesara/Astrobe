#!/usr/bin/env python3
"""
ASTROBE — Unified Astrobiology Analysis Pipeline
=================================================
Run any planet through the full Astrobe pipeline from the command line.

Usage examples:

  # Run a built-in preset
  python -m astrobe --preset earth
  python -m astrobe --preset kepler_442b

  # List all built-in presets
  python -m astrobe --list-presets

  # Define a custom planet interactively
  python -m astrobe --interactive

  # Define a custom planet via flags (minimum required fields)
  python -m astrobe --name "My Exoplanet" \\
                    --temp 12 \\
                    --has-atmosphere \\
                    --has-liquid-water \\
                    --age 3.5 \\
                    --star-luminosity 0.6 \\
                    --star-radiation 4.0 \\
                    --orbital-distance 0.7

  # Full custom planet with all parameters
  python -m astrobe --name "Kepler-22b" \\
                    --location "620 ly" \\
                    --temp -11 \\
                    --has-atmosphere \\
                    --age 4.0 \\
                    --geothermal 5.0 \\
                    --tectonics 4.5 \\
                    --silica 5.0 \\
                    --pressure 1.0 \\
                    --ph 7.0 \\
                    --salinity 35 \\
                    --radiation-dose 0.005 \\
                    --water-activity 0.80 \\
                    --star-type G \\
                    --star-luminosity 0.79 \\
                    --star-radiation 3.5 \\
                    --orbital-distance 0.849
"""

import argparse
import sys

from astrobe.models import CandidatePlanet, Star
from astrobe.pipeline import analyse_planet
from astrobe.presets import PRESETS
from astrobe.reporter import print_report


# ── ANSI colours ──────────────────────────────────────────────────────────────
R = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
BLUE = "\033[94m"


def _ask(prompt, default=None, cast=str, required=False):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {BOLD}{prompt}{R}{suffix}: ").strip()
        if not raw:
            if default is not None:
                return default
            if required:
                print(f"  {RED}This field is required.{R}")
                continue
            return None
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"  {RED}Invalid input — expected {cast.__name__}.{R}")


def _ask_bool(prompt, default=False):
    options = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"  {BOLD}{prompt}{R} {options}: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print(f"  {RED}Please enter y or n.{R}")


def interactive_build() -> CandidatePlanet:
    print()
    print(CYAN + BOLD + "  ── Define Your Planet ──────────────────────────────" + R)
    print(GREY + "  Press Enter to accept defaults shown in [brackets]." + R)
    print()

    name     = _ask("Planet name", required=True)
    location = _ask("Location / system", default="Unknown")
    desc     = _ask("Brief description", default="")

    print()
    print(BOLD + "  — Stellar Parameters —" + R)
    star_type  = _ask("Star type  (M / K / G / F)", default="G")
    luminosity = _ask("Star luminosity (solar units, e.g. 1.0 = Sun)", default=1.0, cast=float)
    radiation  = _ask("Stellar radiation index (0–10, higher = more flare activity)", default=3.0, cast=float)
    orb_dist   = _ask("Orbital distance (AU)", default=1.0, cast=float)

    print()
    print(BOLD + "  — Climate / Surface —" + R)
    temp     = _ask("Surface temperature (°C)", default=15.0, cast=float)
    has_atm  = _ask_bool("Does it have an atmosphere?", default=True)
    has_water= _ask_bool("Does it have confirmed liquid water?", default=False)
    atm_type = _ask("Atmosphere type  (reducing / neutral / oxidizing)", default="neutral")

    print()
    print(BOLD + "  — Geological Parameters (all 0–10) —" + R)
    age       = _ask("Planet age (billion years, Ga)", default=4.5, cast=float)
    geothermal= _ask("Geothermal activity (0=cold/dead, 10=highly active)", default=5.0, cast=float)
    tectonics = _ask("Tectonic activity  (0=none, 10=very active)", default=5.0, cast=float)
    silica    = _ask("Silica content     (0=low, 10=high)", default=5.0, cast=float)

    print()
    print(BOLD + "  — Environmental Parameters (for HOPE extremophile matching) —" + R)
    pressure  = _ask("Surface pressure (atm, Earth=1.0)", default=1.0, cast=float)
    ph        = _ask("pH (0–14, neutral=7)", default=7.0, cast=float)
    salinity  = _ask("Salinity (ppt, ocean=35)", default=35.0, cast=float)
    rad_dose  = _ask("Radiation dose (Gy/yr, Earth surface≈0.003)", default=0.003, cast=float)
    water_act = _ask("Water activity (0.0–1.0, pure water=1.0)", default=0.99, cast=float)

    return CandidatePlanet(
        name=name, location=location, description=desc,
        star=Star(star_type=star_type, luminosity=luminosity, radiation=radiation),
        orbital_distance_au=orb_dist,
        temperature_c=temp,
        has_atmosphere=has_atm,
        has_liquid_water=has_water,
        atmosphere_type=atm_type,
        age_ga=age,
        geothermal=geothermal,
        tectonics=tectonics,
        silica=silica,
        pressure_atm=pressure,
        ph=ph,
        salinity_ppt=salinity,
        radiation_gy=rad_dose,
        water_activity=water_act,
    )


def list_presets():
    print()
    print(BOLD + CYAN + "  Built-in Planet Presets" + R)
    print()
    categories = {
        "Solar System": ["earth", "mars", "europa_ocean", "enceladus"],
        "Exoplanets":   ["kepler_442b", "proxima_b"],
    }
    for cat, keys in categories.items():
        print(f"  {BOLD}{cat}{R}")
        for k in keys:
            p = PRESETS.get(k)
            if p:
                print(f"    {GREEN}{k:<20}{R}  {p.name}  {GREY}({p.location}){R}")
        print()
    print(GREY + "  Use: python -m astrobe --preset <name>" + R)
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="python -m astrobe",
        description="ASTROBE — Unified Astrobiology Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preset",      metavar="NAME", help="Run a built-in planet preset")
    mode.add_argument("--interactive", action="store_true", help="Build a custom planet interactively")
    mode.add_argument("--list-presets",action="store_true", help="List available presets and exit")

    # Custom planet flags
    g = parser.add_argument_group("Custom planet parameters")
    g.add_argument("--name",              help="Planet name (required for custom)")
    g.add_argument("--location",          default="Unknown")
    g.add_argument("--description",       default="")
    g.add_argument("--temp",              type=float, default=15.0,  metavar="°C")
    g.add_argument("--has-atmosphere",    action="store_true", dest="has_atmosphere")
    g.add_argument("--no-atmosphere",     action="store_false", dest="has_atmosphere")
    g.add_argument("--has-liquid-water",  action="store_true", dest="has_liquid_water")
    g.add_argument("--no-liquid-water",   action="store_false", dest="has_liquid_water")
    g.add_argument("--atm-type",          default="neutral", choices=["reducing","neutral","oxidizing"])
    g.add_argument("--age",               type=float, default=4.5,   metavar="Ga")
    g.add_argument("--geothermal",        type=float, default=5.0)
    g.add_argument("--tectonics",         type=float, default=5.0)
    g.add_argument("--silica",            type=float, default=5.0)
    g.add_argument("--pressure",          type=float, default=1.0,   metavar="atm")
    g.add_argument("--ph",                type=float, default=7.0)
    g.add_argument("--salinity",          type=float, default=35.0,  metavar="ppt")
    g.add_argument("--radiation-dose",    type=float, default=0.003, metavar="Gy/yr", dest="radiation_dose")
    g.add_argument("--water-activity",    type=float, default=0.99,  dest="water_activity")
    g.add_argument("--orbital-distance",  type=float, default=1.0,   metavar="AU", dest="orbital_distance")
    g.add_argument("--star-type",         default="G", dest="star_type")
    g.add_argument("--star-luminosity",   type=float, default=1.0,   dest="star_luminosity")
    g.add_argument("--star-radiation",    type=float, default=3.0,   dest="star_radiation")

    # Pipeline options
    parser.add_argument("--no-evolution", action="store_false", dest="run_evo",
                        help="Skip DigitalDNA evolutionary simulation (faster)")
    parser.add_argument("--evo-generations", type=int, default=20, dest="evo_gen",
                        help="Number of evolutionary generations (default: 20)")
    parser.add_argument("--show-all-organisms", action="store_true", dest="show_all",
                        help="Show unlikely organisms in HOPE results too")

    parser.set_defaults(has_atmosphere=True, has_liquid_water=False, run_evo=True)
    args = parser.parse_args()

    # Header
    print()
    print(CYAN + BOLD + "╔══════════════════════════════════════════════════════════╗")
    print("║  A S T R O B E  — Astrobiology Analysis Pipeline        ║")
    print("║  Core · BioSearch · HOPE · DigitalDNA                   ║")
    print("╚══════════════════════════════════════════════════════════╝" + R)

    if args.list_presets:
        list_presets()
        sys.exit(0)

    # Build the planet
    if args.preset:
        key = args.preset.lower()
        if key not in PRESETS:
            print(f"\n  {RED}Unknown preset '{key}'. Run --list-presets to see options.{R}\n")
            sys.exit(1)
        planet = PRESETS[key]

    elif args.interactive:
        planet = interactive_build()

    elif args.name:
        planet = CandidatePlanet(
            name=args.name, location=args.location, description=args.description,
            star=Star(star_type=args.star_type, luminosity=args.star_luminosity,
                      radiation=args.star_radiation),
            orbital_distance_au=args.orbital_distance,
            temperature_c=args.temp,
            has_atmosphere=args.has_atmosphere,
            has_liquid_water=args.has_liquid_water,
            atmosphere_type=args.atm_type,
            age_ga=args.age,
            geothermal=args.geothermal,
            tectonics=args.tectonics,
            silica=args.silica,
            pressure_atm=args.pressure,
            ph=args.ph,
            salinity_ppt=args.salinity,
            radiation_gy=args.radiation_dose,
            water_activity=args.water_activity,
        )
    else:
        print(f"\n  {YELLOW}No planet specified. Use --preset, --interactive, or --name.{R}")
        print(f"  Run {BOLD}python -m astrobe --help{R} for usage.\n")
        sys.exit(1)

    # Run pipeline
    print(f"\n  Running pipeline for: {BOLD}{planet.name}{R}\n")
    report = analyse_planet(
        planet,
        run_evolution=args.run_evo,
        evo_generations=args.evo_gen,
        verbose=True,
    )

    print_report(report, show_all_organisms=args.show_all)


if __name__ == "__main__":
    main()
