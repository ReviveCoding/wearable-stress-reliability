# CI Scope

GitHub Actions CI for HealthSense-RCP is intentionally scoped to repository integrity:

- package installation
- formatting
- linting
- Python syntax/bytecode compilation
- fixture-level unit/smoke tests
- package build

The CI workflow does **not** execute the full WESAD benchmark, GPU evaluation, cross-subject evidence runs, local failure-analysis artifacts, or strict full-source mypy validation. Full WESAD benchmark evidence is generated separately in a controlled local environment because the WESAD data and GPU runtime are not bundled with this repository.

This repository should be described as CI-qualified for code/package integrity, not as CI-reproducing the full benchmark or as strict-mypy-qualified.