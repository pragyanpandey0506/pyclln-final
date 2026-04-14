#!/usr/bin/env python3
"""Compatibility wrapper for the moved vocab32 linear-resistor remodel trainer."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parents[2] / "linear_net" / "lin_lang_model" / "vocab32" / "clln_lang_trainer_embed4_onehot_ce_linear_resistor_remodel.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
