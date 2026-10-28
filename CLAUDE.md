# Jamaica MICS6 child-profiles project

Read `README.md` first: it has the goal, every step done, verified decisions, results, caveats and the to-do list. Detailed results: `outputs/results_summary.md`.

- Use the project venv: `.venv/bin/python` (never system Python). Code is in `scripts/` (`config.py` holds paths, seeds, codes, settings; `01`–`07` run in order and are deterministic); `run_full_pipeline.py` in the project root runs them all and writes `outputs/provenance/run_manifest.json`. Data is in `data/`, PDFs in `docs/`, outputs in `outputs/<step folder>/`.
- Never use reading or numeracy to build, predict or explain clusters. Report cluster n beside every estimate.
- Use absolute paths in shell commands (`cd` has previously changed the working directory).
- Keep new files minimal; update `README.md` whenever a step or result changes.
