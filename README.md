# HealthSense-RCP

HealthSense-RCP is an offline, public-data, non-clinical ML reliability lab for wearable physiological-state research. It tests whether an evaluation candidate may advance to **controlled research validation**. It is not a medical device, diagnostic, treatment system, or consumer alert product.

## What it implements

| Candidate | Role |
|---|---|
| C0 | majority/no-alert sanity bound |
| C1 | quality-aware gradient-boosted operational baseline |
| C2 | mask- and quality-gated temporal Transformer |
| C3 | PHT-derived Reference-Conditioned Physiological Phase residual model |
| C4 | optional ABC-inspired temporal boundary correction |
| C5 | calibrated selective routing surface |

The canonical output combines participant-disjoint scoring, calibration, threshold selection on development participants only, robustness scenarios, latency, release decisions, figures, model cards, and checksum-verified artifacts.

## Quickstart: deterministic CPU smoke

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_project_dependencies.py
python -m pip check
python -m healthsense.cli smoke --output-root outputs --run-id smoke-local --overwrite
python scripts/verify_artifacts.py outputs/smoke-local
```

On Linux/macOS, activate with `source .venv/bin/activate`.

The default packaged configuration is CPU-only by design, allowing the same smoke path to run locally and on GitHub-hosted CPU runners. It is a synthetic engineering check, not WESAD or PhysioNet evidence.

## Local verification

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_local_validation.ps1
```

or, where `make` is available:

```bash
make ci-local
```

`make clean-generated` removes only generated caches, build outputs, test reports, smoke outputs, and editable-install metadata. `make deps-check-full` runs a full `pip check` in a clean virtual environment. CI evidence artifacts are retained for seven days to limit repository storage while preserving recent failure diagnostics.

## Real-data boundaries

- Raw WESAD and PhysioNet data are excluded from version control and release bundles.
- WESAD pickle parsing requires explicit `--trust-pickle` acknowledgement.
- Prepared PhysioNet windows remain unlabelled until a reviewed tag map is supplied.
- External validation requires a shared feature schema and cannot select a champion or advance a release.
- C3's reference anchors are built from training data only. A training row cannot reference itself as an anchor.

## Device configuration

The packaged smoke profile uses CPU intentionally. For an explicit GPU research run, pass the device at the CLI boundary so the resolved value is included in the manifest configuration hash:

```powershell
python -m healthsense.cli evaluate-canonical `
  --npz "data\processed\wesad\wesad_canonical_windows.npz" `
  --metadata "data\processed\wesad\wesad_window_metadata.csv" `
  --output-root outputs `
  --run-id wesad-development `
  --device cuda
```

A requested `cuda` run fails closed when CUDA is unavailable. Run manifests record the requested configuration, resolved device, hardware snapshot, synchronized inference p50/p95/p99 latency, and GPU peak memory where applicable.

## Claims

Allowed only when corresponding evidence exists: `implemented`, `locally executed`, `public-data validated`, `GPU validated`, and `research-derived`.

Not allowed: `clinical`, `diagnostic`, `treatment`, `production-ready`, `Apple Watch validated`, or `cross-device generalization` based only on wrist-to-wrist external transfer.

See `docs/experiment_protocol.md`, `docs/operational_runbook.md`, `docs/architecture.md`, and `docs/validation_scope.md`.
