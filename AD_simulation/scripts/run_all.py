# run_all.py — execute the full AD simulation pipeline in order.
# Usage:  python scripts/run_all.py   (from AD_simulation/)

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for step in ["generate_simulation.py", "ranking_comparison.py",
             "sensitivity_analysis.py", "figure_5.py"]:
    print("=== %s ===" % step, flush=True)
    r = subprocess.run([sys.executable, str(HERE / step)], capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit("FAILED: %s" % step)
print("PIPELINE_OK", flush=True)
