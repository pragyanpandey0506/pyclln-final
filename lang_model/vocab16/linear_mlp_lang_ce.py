#!/usr/bin/env python3
"""Compatibility wrapper for the moved vocab16 linear baseline."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parents[2] / "linear_net" / "lin_lang_model" / "vocab16" / "linear_mlp_lang_ce.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
