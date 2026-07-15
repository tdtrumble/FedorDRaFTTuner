"""Command-line entry point for local-only Krea 2 DRaFT training."""

from __future__ import annotations

import argparse
import os
import sys

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply DRaFT reward optimization to an existing Krea 2 LoRA/LoKr."
    )
    parser.add_argument(
        "config_file_list",
        nargs="+",
        help="One or more Krea 2 DRaFT YAML/JSON config files",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate config structure and all required local files, then exit",
    )
    parser.add_argument(
        "-r",
        "--recover",
        action="store_true",
        help="Continue to the next config if a job fails",
    )
    parser.add_argument(
        "-n", "--name", default=None, help="Replace the config's [name] placeholder"
    )
    parser.add_argument("-l", "--log", default=None, help="Write console output to this file")
    return parser


def _configure_offline_runtime() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    # Defense in depth: retained third-party libraries must never contact a hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DISABLE_TELEMETRY"] = "YES"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def _validate_configs(paths: list[str], name: str | None) -> None:
    from toolkit.config import get_config
    from toolkit.draft_config import validate_public_config

    for path in paths:
        validate_public_config(get_config(path, name))
        print(f"OK: {path}")


def _seed_everything() -> None:
    raw_seed = os.environ.get("SEED")
    if raw_seed is None:
        return
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise ValueError(f"SEED must be an integer, got {raw_seed!r}") from exc

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _print_end_message(print_acc, jobs_completed: int, jobs_failed: int) -> None:
    print_acc("")
    print_acc("========================================")
    print_acc(f"Result: {jobs_completed} completed, {jobs_failed} failed")
    print_acc("========================================")


def main() -> int:
    args = _parser().parse_args()
    _configure_offline_runtime()
    sys.path.insert(0, os.getcwd())
    _validate_configs(args.config_file_list, args.name)
    if args.validate:
        return 0

    import torch

    if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
        torch.autograd.set_detect_anomaly(True)
    _seed_everything()

    from toolkit.accelerator import get_accelerator
    from toolkit.job import get_job
    from toolkit.print import print_acc, setup_log_to_file

    if args.log:
        setup_log_to_file(args.log)
    accelerator = get_accelerator()
    completed = failed = 0
    if accelerator.is_main_process:
        print_acc(f"Running {len(args.config_file_list)} DRaFT job(s) in offline mode")

    for config_file in args.config_file_list:
        job = None
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            completed += 1
        except KeyboardInterrupt:
            if job is not None and getattr(job, "process", None):
                job.process[0].on_error(KeyboardInterrupt())
            raise
        except Exception as exc:
            print_acc(f"Error running job: {exc}")
            failed += 1
            if job is not None and getattr(job, "process", None):
                try:
                    job.process[0].on_error(exc)
                except Exception as hook_exc:
                    print_acc(f"Error running on_error: {hook_exc}")
            if not args.recover:
                _print_end_message(print_acc, completed, failed)
                raise

    if accelerator.is_main_process:
        _print_end_message(print_acc, completed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
