"""Math and physics solver CodeAgent with a numeric final-answer check.

Source shape:
- docs/source/en/guided_tour.md "Customizing agent termination conditions": a CodeAgent with
  `final_answer_checks=[is_integer]` that keeps working until the answer passes the check.
- examples/smolagents_benchmark/run.py: CodeAgent for math benchmarks with
  additional_authorized_imports=["numpy", "sympy"].
Adaptation: the check validates that the answer carries a number (and units for physics questions).
Two small local tools give CODATA constants and unit conversions so physics tasks have tool spans.
Code runs in the smolagents local executor with only math, fractions, decimal, statistics, itertools,
cmath and sympy authorized.
"""
from __future__ import annotations

import re
import uuid

import smol_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T
from smolagents import CodeAgent, tool

AGENT = AgentMeta(
    id="smol-math-solver",
    framework="smolagents",
    source_url="https://github.com/huggingface/smolagents/blob/main/docs/source/en/guided_tour.md",
    description="CodeAgent that solves math and physics problems in sandboxed Python with sympy, constants and unit tools, and a numeric answer check.",
    complexity="simple",
    model="claude-haiku-5-5",
    tracing="openinference",
)

AUTHORIZED_IMPORTS = ["math", "cmath", "fractions", "decimal", "statistics", "itertools", "sympy"]

CONSTANTS = {
    "speed_of_light": (299_792_458.0, "m/s"),
    "gravitational_constant": (6.67430e-11, "m^3 kg^-1 s^-2"),
    "planck_constant": (6.62607015e-34, "J s"),
    "reduced_planck_constant": (1.054571817e-34, "J s"),
    "boltzmann_constant": (1.380649e-23, "J/K"),
    "avogadro_constant": (6.02214076e23, "1/mol"),
    "gas_constant": (8.314462618, "J/(mol K)"),
    "elementary_charge": (1.602176634e-19, "C"),
    "electron_mass": (9.1093837015e-31, "kg"),
    "proton_mass": (1.67262192369e-27, "kg"),
    "vacuum_permittivity": (8.8541878128e-12, "F/m"),
    "vacuum_permeability": (1.25663706212e-6, "N/A^2"),
    "standard_gravity": (9.80665, "m/s^2"),
    "stefan_boltzmann_constant": (5.670374419e-8, "W m^-2 K^-4"),
    "earth_mass": (5.9722e24, "kg"),
    "earth_radius": (6.371e6, "m"),
    "sun_mass": (1.98847e30, "kg"),
    "astronomical_unit": (1.495978707e11, "m"),
    "standard_atmosphere": (101_325.0, "Pa"),
    "speed_of_sound_air_20c": (343.0, "m/s"),
}

# Factors to SI base units.
UNITS = {
    "m": 1.0, "km": 1e3, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9, "mi": 1609.344, "ft": 0.3048, "in": 0.0254, "yd": 0.9144,
    "ly": 9.4607e15, "au": 1.495978707e11,
    "kg": 1.0, "g": 1e-3, "mg": 1e-6, "lb": 0.45359237, "oz": 0.028349523125, "t": 1e3,
    "s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0, "day": 86400.0, "yr": 3.15576e7,
    "j": 1.0, "kj": 1e3, "cal": 4.184, "kcal": 4184.0, "ev": 1.602176634e-19, "kwh": 3.6e6, "btu": 1055.06,
    "w": 1.0, "kw": 1e3, "hp": 745.699872,
    "pa": 1.0, "kpa": 1e3, "bar": 1e5, "atm": 101325.0, "psi": 6894.757, "mmhg": 133.322,
    "n": 1.0, "kn": 1e3, "lbf": 4.4482216152605,
    "m/s": 1.0, "km/h": 1 / 3.6, "mph": 0.44704, "knot": 0.514444,
    "l": 1e-3, "ml": 1e-6, "m3": 1.0, "gal": 3.785411784,
}
DIMENSION = {
    **{u: "length" for u in ("m", "km", "cm", "mm", "um", "nm", "mi", "ft", "in", "yd", "ly", "au")},
    **{u: "mass" for u in ("kg", "g", "mg", "lb", "oz", "t")},
    **{u: "time" for u in ("s", "ms", "min", "h", "day", "yr")},
    **{u: "energy" for u in ("j", "kj", "cal", "kcal", "ev", "kwh", "btu")},
    **{u: "power" for u in ("w", "kw", "hp")},
    **{u: "pressure" for u in ("pa", "kpa", "bar", "atm", "psi", "mmhg")},
    **{u: "force" for u in ("n", "kn", "lbf")},
    **{u: "speed" for u in ("m/s", "km/h", "mph", "knot")},
    **{u: "volume" for u in ("l", "ml", "m3", "gal")},
}


@tool
def physical_constant(name: str) -> str:
    """Returns the CODATA value and SI unit of a physical constant.

    Args:
        name: Constant name, for example 'speed_of_light', 'planck_constant', 'boltzmann_constant', 'earth_mass'.
    """
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in CONSTANTS:
        close = [k for k in CONSTANTS if any(w in k for w in key.split("_") if len(w) > 2)]
        return f"Unknown constant '{name}'. Known: {', '.join(close or CONSTANTS)}"
    v, u = CONSTANTS[key]
    return f"{key} = {v!r} {u}"


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Converts a value between units of the same dimension (length, mass, time, energy, power, pressure,
    force, speed, volume). Temperatures are supported with 'C', 'F' and 'K'.

    Args:
        value: The numeric value to convert.
        from_unit: Source unit, for example 'mi', 'kWh', 'psi', 'km/h', 'C'.
        to_unit: Target unit, for example 'km', 'J', 'kPa', 'm/s', 'K'.
    """
    f, t = from_unit.strip(), to_unit.strip()
    temps = {"C", "F", "K"}
    if f.upper() in temps and t.upper() in temps:
        k = {"C": lambda x: x + 273.15, "F": lambda x: (x - 32) * 5 / 9 + 273.15, "K": lambda x: x}[f.upper()](value)
        out = {"C": k - 273.15, "F": (k - 273.15) * 9 / 5 + 32, "K": k}[t.upper()]
        return f"{value} {f.upper()} = {out:.6g} {t.upper()}"
    fl, tl = f.lower(), t.lower()
    if fl not in UNITS or tl not in UNITS:
        return f"Unsupported unit: {f if fl not in UNITS else t}. Supported: {', '.join(sorted(UNITS))}"
    if DIMENSION[fl] != DIMENSION[tl]:
        return f"Cannot convert {DIMENSION[fl]} ({f}) to {DIMENSION[tl]} ({t})."
    return f"{value} {f} = {value * UNITS[fl] / UNITS[tl]:.6g} {t}"


_PHYSICS = re.compile(r"\b(velocity|speed|force|energy|mass|acceleration|momentum|pressure|wavelength|frequency|"
                      r"temperature|power|voltage|current|charge|distance|height|joules?|newtons?|watts?)\b", re.I)


def answer_has_number(final_answer, memory, agent=None) -> bool:
    """Variant of the guided-tour `is_integer` check: the answer must contain a number, unless the agent
    explains that the problem is ill-posed."""
    text = str(final_answer)
    if re.search(r"\d", text):
        return True
    if re.search(r"(no (real )?solution|undefined|ill[- ]posed|cannot be determined|insufficient|contradict|impossible)", text, re.I):
        return True
    raise ValueError("The final answer must contain the numeric result (with units for physical quantities), "
                     "or state explicitly why the problem has no well-defined answer.")


def setup() -> None:
    C.setup_tracing()


def run(task: dict) -> str:
    T.set_run_id(f"{AGENT.id}:{task['id']}:{uuid.uuid4().hex[:8]}")
    text = task["input"]
    agent = CodeAgent(
        tools=[physical_constant, convert_units],
        model=C.make_model(AGENT.model),
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        max_steps=5,
        verbosity_level=0,
        final_answer_checks=[answer_has_number],
        instructions=(
            "Solve the problem by computing in Python, not by mental arithmetic. Verify the result once "
            "(substitute back, check limits or units). Give the final answer with a one-line justification"
            + (" and SI units." if _PHYSICS.search(text) else ".")
        ),
    )
    return C.last_text(agent.run(text))
