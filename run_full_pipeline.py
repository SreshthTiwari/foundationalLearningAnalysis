"""run the whole pipeline (scripts 01-07 in order) and record how it was run

Usage: .venv/bin/python run_full_pipeline.py [--k N]
  --k N   force the number of clusters in 03 (default: auto)

Writes outputs/provenance/run_manifest.json: python and package versions, git commit, every
setting in scripts/config.py (including random seeds), input data hash, per-step timings,
and a hash of every output file. Stops at the first step that fails.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import config  # noqa: E402

STEPS = ["01_data_audit.py", "02_build_dataset.py", "03_profiles.py", "04_explain_and_outcomes.py",
         "05_survey_inference.py", "06_compare_k.py", "07_explainable_ml.py"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def write_manifest(started, steps, args):
    outputs = {str(p.relative_to(ROOT)): sha256(p)
               for p in sorted(config.OUTPUT_ROOT.rglob("*"))
               if p.is_file() and config.OUT_PROVENANCE not in p.parents and p.name != ".DS_Store"}
    manifest = dict(
        started_utc=started.isoformat(timespec="seconds"),
        finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        command=" ".join([Path(sys.executable).name, *sys.argv]),
        python=dict(version=platform.python_version(), implementation=platform.python_implementation(),
                    executable=sys.executable),
        platform=dict(system=platform.system(), release=platform.release(), machine=platform.machine()),
        git=dict(commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                 uncommitted_changes=bool(git("status", "--porcelain"))),
        settings=config.settings_snapshot(),
        cli_args=vars(args),
        input_data={str(config.FS_SAV.relative_to(ROOT)): dict(sha256=sha256(config.FS_SAV),
                                                                bytes=config.FS_SAV.stat().st_size)},
        code_sha256={str(p.relative_to(ROOT)): sha256(p)
                     for p in [ROOT / "run_full_pipeline.py", *sorted(SCRIPTS.glob("*.py"))]},
        steps=steps,
        installed_packages=dict(sorted((d.metadata["Name"], d.version) for d in metadata.distributions())),
        output_sha256=outputs,
    )
    path = config.ensure(config.OUT_PROVENANCE) / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    started = datetime.now(timezone.utc)
    steps = []
    for name in STEPS:
        cmd = [sys.executable, str(SCRIPTS / name)] + (["--k", str(args.k)] if args.k and name.startswith("03") else [])
        print(f"\n>>> {name}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        steps.append(dict(script=name, args=cmd[2:], return_code=rc, seconds=round(time.time() - t0, 1)))
        if rc != 0:
            write_manifest(started, steps, args)
            sys.exit(f"{name} failed (exit code {rc}); partial manifest written, stopping")
    path = write_manifest(started, steps, args)
    print(f"\nall {len(STEPS)} steps finished; manifest: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
