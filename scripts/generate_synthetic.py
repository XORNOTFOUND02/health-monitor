#!/usr/bin/env python3
"""
CLI script to generate synthetic wearable sensor data.

Usage
-----
    python scripts/generate_synthetic.py --num-sessions 100 --output-dir data/synthetic/raw
    python scripts/generate_synthetic.py --conditions tachycardia fever --duration 60
    python scripts/generate_synthetic.py --num-sessions 1000 --seed 123 --include-labels

The script produces JSON session files compatible with
:class:`src.data.loader.SensorDataLoader`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so that ``data.synthetic`` resolves.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.synthetic.generator import SyntheticDataGenerator, CONDITIONS

logger = logging.getLogger("generate_synthetic")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic wearable sensor data for model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Supported conditions:\n"
            + "\n".join(f"  - {c}" for c in CONDITIONS)
            + "\n\nExamples:\n"
            "  python scripts/generate_synthetic.py --num-sessions 50\n"
            "  python scripts/generate_synthetic.py --conditions all --output-dir data/synthetic/raw\n"
            "  python scripts/generate_synthetic.py --num-sessions 1000 --seed 42 --duration 120\n"
        ),
    )

    parser.add_argument(
        "-n",
        "--num-sessions",
        type=int,
        default=10,
        help="Number of sessions to generate (default: 10).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data/synthetic/raw",
        help="Output directory for JSON files (default: data/synthetic/raw).",
    )
    parser.add_argument(
        "-c",
        "--conditions",
        nargs="+",
        default=None,
        choices=CONDITIONS + ["all"],
        help=(
            'Condition(s) to generate. Use "all" for every condition plus '
            "normal sessions. Default: balanced mix of all conditions."
        ),
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=300,
        help="Session duration in seconds (default: 300).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--include-labels",
        action="store_true",
        default=True,
        help="Include ground-truth labels in the output (default: True).",
    )
    parser.add_argument(
        "--no-labels",
        dest="include_labels",
        action="store_false",
        help="Omit ground-truth labels from the output.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the synthetic data generation CLI."""
    args = _parse_args(argv)

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve conditions
    conditions: list[str] | None = None
    if args.conditions is not None:
        if "all" in args.conditions:
            conditions = None  # generator default: balanced mix
        else:
            conditions = args.conditions

    # Resolve output path (relative to project root)
    output_path = Path(args.output_dir)
    if not output_path.is_absolute():
        output_path = _PROJECT_ROOT / output_path

    logger.info("Configuration:")
    logger.info("  num_sessions  = %d", args.num_sessions)
    logger.info("  output_dir    = %s", output_path)
    logger.info("  conditions    = %s", conditions if conditions else "balanced (all)")
    logger.info("  duration      = %d s", args.duration)
    logger.info("  seed          = %d", args.seed)
    logger.info("  include_labels = %s", args.include_labels)

    # Generate
    generator = SyntheticDataGenerator(seed=args.seed)

    t_start = time.time()
    sessions = generator.generate_dataset(
        num_sessions=args.num_sessions,
        conditions=conditions,
        output_dir=str(output_path),
        include_labels=args.include_labels,
    )
    elapsed = time.time() - t_start

    logger.info("Done. Generated %d sessions in %.2f s (%.1f sessions/s).",
                len(sessions), elapsed,
                len(sessions) / max(elapsed, 1e-6))

    # Summary stats
    condition_counts: dict[str, int] = {}
    for sess in sessions:
        cond = sess.get("metadata", {}).get("condition", "unknown")
        condition_counts[cond] = condition_counts.get(cond, 0) + 1

    logger.info("Session breakdown:")
    for cond, count in sorted(condition_counts.items()):
        logger.info("  %-20s %d", cond, count)


if __name__ == "__main__":
    main()
