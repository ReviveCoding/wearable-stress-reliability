from __future__ import annotations

import argparse
from pathlib import Path

from healthsense.datasets.physionet import inspect_physionet, prepare_physionet
from healthsense.datasets.wesad import prepare_wesad
from healthsense.pipeline import (
    run_canonical,
    run_cross_subject_confirmation,
    run_external_validation,
    run_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="healthsense", description="HealthSense-RCP non-clinical reliability lab"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="run deterministic synthetic end-to-end artifact smoke")
    smoke.add_argument("--config", default=None)
    smoke.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    smoke.add_argument("--output-root", default=None)
    smoke.add_argument("--run-id", default=None)
    smoke.add_argument("--overwrite", action="store_true", help="replace an existing run directory")

    wesad = sub.add_parser(
        "prepare-wesad", help="convert official WESAD pickle files to canonical windows"
    )
    wesad.add_argument("--source-root", required=True)
    wesad.add_argument("--output-dir", required=True)
    wesad.add_argument("--sensor-config", default="wrist", choices=["wrist", "chest"])
    wesad.add_argument(
        "--representation",
        default="native",
        choices=["native", "shared_eda_temp_acc"],
        help="native signals or conservative shared chest/wrist EDA+TEMP+ACC representation",
    )
    wesad.add_argument("--window-seconds", type=int, default=30)
    wesad.add_argument("--hop-seconds", type=int, default=5)
    wesad.add_argument("--sequence-length", type=int, default=60)
    wesad.add_argument("--min-label-purity", type=float, default=0.90)
    wesad.add_argument("--trust-pickle", action="store_true")

    physio = sub.add_parser(
        "inspect-physionet", help="inspect downloaded PhysioNet structure and source constraints"
    )
    physio.add_argument("--source-root", required=True)
    physio.add_argument("--output", default="physionet_source_manifest.json")

    prep_physio = sub.add_parser(
        "prepare-physionet", help="convert E4 sessions with an optional reviewed tag map"
    )
    prep_physio.add_argument("--source-root", required=True)
    prep_physio.add_argument("--output-dir", required=True)
    prep_physio.add_argument("--tag-map", default=None)
    prep_physio.add_argument("--window-seconds", type=int, default=30)
    prep_physio.add_argument("--hop-seconds", type=int, default=5)
    prep_physio.add_argument("--sequence-length", type=int, default=60)

    canonical = sub.add_parser(
        "evaluate-canonical",
        help="run participant-disjoint evaluation from verified canonical data",
    )
    canonical.add_argument("--npz", required=True)
    canonical.add_argument("--metadata", required=True)
    canonical.add_argument("--config", default=None)
    canonical.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    canonical.add_argument("--output-root", default=None)
    canonical.add_argument("--run-id", default=None)
    canonical.add_argument(
        "--split-manifest",
        default=None,
        help="reuse a previously published immutable split_manifest.json",
    )
    canonical.add_argument(
        "--overwrite", action="store_true", help="replace an existing run directory"
    )

    external = sub.add_parser(
        "evaluate-external",
        help="run locked binary cross-protocol or exercise-confound confirmation",
    )
    external.add_argument("--source-npz", required=True)
    external.add_argument("--source-metadata", required=True)
    external.add_argument("--external-npz", required=True)
    external.add_argument("--external-metadata", required=True)
    external.add_argument(
        "--external-mode", choices=["stress_rest", "exercise_confound"], default="stress_rest"
    )
    external.add_argument("--config", default=None)
    external.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    external.add_argument("--output-root", default=None)
    external.add_argument("--run-id", default=None)
    external.add_argument(
        "--overwrite", action="store_true", help="replace an existing run directory"
    )

    cross = sub.add_parser(
        "cross-validate-canonical",
        help="run deterministic nested participant-held-out confirmation folds",
    )
    cross.add_argument("--npz", required=True)
    cross.add_argument("--metadata", required=True)
    cross.add_argument("--folds", type=int, default=5)
    cross.add_argument("--config", default=None)
    cross.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    cross.add_argument("--output-root", default=None)
    cross.add_argument("--run-id", default=None)
    cross.add_argument("--overwrite", action="store_true", help="replace an existing run directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        out = run_smoke(
            args.config, args.output_root, args.run_id, device=args.device, overwrite=args.overwrite
        )
        print(out)
        return 0
    if args.command == "prepare-wesad":
        manifest = prepare_wesad(
            args.source_root,
            args.output_dir,
            trust_pickle=args.trust_pickle,
            sensor_config=args.sensor_config,
            representation=args.representation,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            sequence_length=args.sequence_length,
            min_label_purity=args.min_label_purity,
        )
        print(Path(args.output_dir) / "wesad_source_manifest.json")
        print(manifest["batch"])
        return 0
    if args.command == "inspect-physionet":
        manifest = inspect_physionet(args.source_root, args.output)
        print(args.output)
        print({"csv_count": manifest["csv_count"], "txt_count": manifest["txt_count"]})
        return 0
    if args.command == "prepare-physionet":
        manifest = prepare_physionet(
            args.source_root,
            args.output_dir,
            tag_map_path=args.tag_map,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            sequence_length=args.sequence_length,
        )
        print(Path(args.output_dir) / "physionet_source_manifest.json")
        print({"n_windows": manifest["n_windows"], "labeled_windows": manifest["labeled_windows"]})
        return 0
    if args.command == "evaluate-canonical":
        out = run_canonical(
            args.npz,
            args.metadata,
            args.config,
            args.output_root,
            args.run_id,
            args.split_manifest,
            device=args.device,
            overwrite=args.overwrite,
        )
        print(out)
        return 0
    if args.command == "evaluate-external":
        out = run_external_validation(
            args.source_npz,
            args.source_metadata,
            args.external_npz,
            args.external_metadata,
            external_mode=args.external_mode,
            config_path=args.config,
            output_root=args.output_root,
            run_id=args.run_id,
            device=args.device,
            overwrite=args.overwrite,
        )
        print(out)
        return 0
    if args.command == "cross-validate-canonical":
        out = run_cross_subject_confirmation(
            args.npz,
            args.metadata,
            n_folds=args.folds,
            config_path=args.config,
            output_root=args.output_root,
            run_id=args.run_id,
            device=args.device,
            overwrite=args.overwrite,
        )
        print(out)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
