#!/usr/bin/env python3
"""Compatibility wrapper for the moved vocab32 G/D/T init sweep launcher."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parents[3] / "linear_net" / "lin_lang_model" / "vocab32" / "linear_resistor" / "run_gdt_init_sweep.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
