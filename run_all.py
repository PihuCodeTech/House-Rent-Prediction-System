"""Run the whole project end to end, from any folder:

    python run_all.py                 # data exploration -> 12 model notebooks -> comparison -> final selection
                                      #   -> feature importance -> prediction demo -> tests
    python run_all.py --tuning        # also re-run the grid + Bayesian hyperparameter search first (20-40 min)
    python run_all.py --only random_forest xgboost_model   # just these notebooks (keeps other results)
    python run_all.py --no-tests

Before running it checks the Python packages and the dataset, and it uses THIS Python (your activated .venv)
as the notebook kernel. Each notebook is executed top to bottom and saved with its outputs. If anything fails it
stops, names the notebook and the error, and saves that notebook with the error shown inside it.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
BEFORE = ["data_exploration"]
MODEL_NOTEBOOKS = [
    "baseline_mean",
    "baseline_median",
    "linear_regression",
    "ridge_regression",
    "lasso_regression",
    "elastic_net",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost_model",
    "lightgbm_model",
    "catboost_model",
]
AFTER = ["model_comparison", "final_selection", "feature_importance", "predict_demo"]
REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "nbformat": "nbformat",
    "nbclient": "nbclient",
    "ipykernel": "ipykernel",
    "pytest": "pytest",
}


def fail(msg):
    """Print an error and stop with exit code 1."""
    print(f"\n✗ {msg}", flush=True)
    sys.exit(1)


def preflight(need_optuna):
    """Check packages and dataset before running anything; return the project config."""
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    if sys.prefix == sys.base_prefix:
        print("  (note: no virtual environment is active — run `source .venv/bin/activate` first if you use one)")
    missing = []
    for mod, pip_name in {**REQUIRED, **({"optuna": "optuna"} if need_optuna else {})}.items():
        try:
            importlib.import_module(mod)
        except ImportError as e:
            hint = (
                "  (on macOS also run: brew install libomp)"
                if mod in ("xgboost", "lightgbm") and "lib" in str(e)
                else ""
            )
            missing.append(f"{pip_name}: {e}{hint}")
    if missing:
        fail("Missing or broken packages:\n  " + "\n  ".join(missing) + "\nFix with: pip install -r requirements.txt")
    sys.path.insert(0, str(ROOT))
    from src import config

    if not config.DATA_RAW.exists():
        fail(f"Dataset not found at {config.DATA_RAW.relative_to(ROOT)} — see data/README.md for where to download it.")
    print(f"✓ packages OK, dataset found, seeds {config.SEEDS}")
    return config


def kernel_for_this_python():
    """Register a temporary kernel that runs THIS interpreter, so notebooks never pick up another Python."""
    kdir = Path(tempfile.mkdtemp(prefix="house_rent_kernel_"))
    spec = kdir / "kernels" / "house-rent-runner"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}", "--log-level=ERROR"],
                "display_name": "house-rent-runner",
                "language": "python",
            }
        )
    )
    os.environ["JUPYTER_PATH"] = str(kdir) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    return "house-rent-runner"


def run_notebook(name, kernel):
    """Execute one notebook top to bottom and save it with its outputs; stop on the first error."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    path = NB_DIR / f"{name}.ipynb"
    if not path.exists():
        fail(f"{path.relative_to(ROOT)} does not exist.")
    nb = nbformat.read(path, as_version=4)
    original_spec = nb.metadata.get("kernelspec")
    client = NotebookClient(
        nb, timeout=None, kernel_name=kernel, allow_errors=False, resources={"metadata": {"path": str(NB_DIR)}}
    )
    t0 = time.time()
    print(f"▶ {name:<22}", end="", flush=True)
    try:
        client.execute()
        ok, err = True, None
    except CellExecutionError as e:
        ok, err = False, e
    finally:
        if original_spec is not None:
            nb.metadata["kernelspec"] = original_spec
        nbformat.write(nb, path)
    if not ok:
        import re

        text = re.sub(r"\x1b\[[0-9;]*m", "", str(err))  # drop terminal colour codes
        lines = [line for line in text.splitlines() if line.strip()]
        print(" FAILED")
        fail(f"{name}.ipynb failed (the error is saved inside the notebook):\n  " + "\n  ".join(lines[-6:]))
    print(f" ✓ {time.time() - t0:5.0f}s", flush=True)


def main():
    """Parse arguments, run the notebooks in order, run the tests, print the results."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tuning", action="store_true", help="also run notebooks/hyperparameter_tuning.ipynb first")
    ap.add_argument("--only", nargs="+", metavar="NOTEBOOK", help="run only these notebooks (names without .ipynb)")
    ap.add_argument("--no-tests", action="store_true", help="skip the pytest checks at the end")
    args = ap.parse_args()

    config = preflight(need_optuna=args.tuning)
    kernel = kernel_for_this_python()

    if args.only:
        order = [n.removesuffix(".ipynb") for n in args.only]
    else:
        order = BEFORE + (["hyperparameter_tuning"] if args.tuning else []) + MODEL_NOTEBOOKS + AFTER
        # fresh run: clear outputs that the notebooks regenerate, so nothing stale survives a failed run
        stale = [
            config.RUN_REFERENCE,
            config.RESULTS_CSV,
            *(f for f in config.MODELS_DIR.glob("*") if f.is_file() and f.name != ".gitkeep"),
        ]
        for f in stale:
            if f.exists():
                f.unlink()
        print("✓ cleared previous results.csv, run reference and saved models")

    t0 = time.time()
    print(f"\nRunning {len(order)} notebooks\n")
    for name in order:
        run_notebook(name, kernel)
    print(f"\n✓ all notebooks finished in {(time.time() - t0) / 60:.1f} min")

    if not args.no_tests:
        print("\nRunning tests (tests/test_pipeline.py)")
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-W", "ignore", str(ROOT / "tests")],
            cwd=ROOT,
        )
        if r.returncode != 0:
            fail("tests failed — see the output above.")

    if config.RESULTS_CSV.exists():
        import pandas as pd

        table = pd.read_csv(config.RESULTS_CSV, encoding="utf-8-sig", dtype=str)
        print(
            f"\nRESULTS — mean ± std over seeds {', '.join(map(str, config.SEEDS))} ({config.RESULTS_CSV.relative_to(ROOT)})"
        )
        print(table.to_string(index=False))
    saved = sorted(p.name for p in config.MODELS_DIR.glob("*.joblib"))
    print(f"\nSaved models in models/: {', '.join(saved) if saved else 'none'}")
    print("\n✓ done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("interrupted.")
