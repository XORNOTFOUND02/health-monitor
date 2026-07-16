#!/usr/bin/env python3
"""
Prepare deployment package for Hugging Face Spaces.

Creates a deployment directory with all required files for HF Spaces,
or validates that the existing hf_space/ setup is correct.

Usage:
    python scripts/prepare_deploy.py              # check readiness
    python scripts/prepare_deploy.py --package     # create deploy package
"""
import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("prepare_deploy")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_DIR = PROJECT_ROOT / "hf_space"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
DEPLOY_DIR = PROJECT_ROOT / "deploy"


REQUIRED_FILES: List[Tuple[str, str, str]] = [
    ("app.py", "hf_space/app.py", "Gradio application"),
    ("requirements.txt", "hf_space/requirements.txt", "Python dependencies"),
    ("README.md", "hf_space/README.md", "Space documentation"),
    ("cardiac.joblib", "models/cardiac.joblib", "Cardiac model"),
    ("respiratory.joblib", "models/respiratory.joblib", "Respiratory model"),
    ("activity.joblib", "models/activity.joblib", "Activity model"),
    ("feature_names.json", "models/feature_names.json", "Feature names"),
    ("cardiac.meta.json", "models/cardiac.meta.json", "Cardiac metadata"),
    ("respiratory.meta.json", "models/respiratory.meta.json", "Respiratory metadata"),
    ("activity.meta.json", "models/activity.meta.json", "Activity metadata"),
]

REQUIRED_DIRS: List[Tuple[str, Path, str]] = [
    ("src package", SRC_DIR, "Source code package"),
    ("data synthetic", DATA_DIR / "synthetic", "Synthetic data generator"),
]


def check_readiness() -> List[str]:
    """Check all required files exist and return list of issues."""
    issues: List[str] = []

    # Check files
    for name, rel_path, description in REQUIRED_FILES:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            issues.append(f"MISSING {description}: {rel_path}")
        elif full_path.stat().st_size == 0:
            issues.append(f"EMPTY {description}: {rel_path}")

    # Check directories
    for name, full_path, description in REQUIRED_DIRS:
        if not full_path.exists():
            issues.append(f"MISSING {description}: {full_path.relative_to(PROJECT_ROOT)}")
        elif not full_path.is_dir():
            issues.append(f"NOT A DIR {description}: {full_path.relative_to(PROJECT_ROOT)}")

    # Check feature names count
    fn_path = MODELS_DIR / "feature_names.json"
    if fn_path.exists():
        try:
            with fn_path.open() as f:
                fn = json.load(f)
            if len(fn) < 100:
                issues.append(f"LOW FEATURE COUNT ({len(fn)} < 100): feature_names.json")
        except Exception as e:
            issues.append(f"INVALID feature_names.json: {e}")

    # Check model file sizes (info only — HF Spaces has 1 GB+ storage)
    for model_name in ["cardiac.joblib", "respiratory.joblib", "activity.joblib"]:
        model_path = MODELS_DIR / model_name
        if model_path.exists():
            size_kb = model_path.stat().st_size / 1024
            if size_kb > 10000:  # 10 MB+ is excessive
                issues.append(f"LARGE MODEL ({size_kb:.0f} KB): {model_name}"
                              f" — consider reducing num_boost_round or feature count")

    return issues


def create_package() -> None:
    """Create a deployment directory with all required files."""
    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)

    DEPLOY_DIR.mkdir(parents=True)

    # Copy hf_space files
    for fname in ["app.py", "requirements.txt", "README.md", ".gitignore"]:
        src = HF_DIR / fname
        if src.exists():
            shutil.copy2(src, DEPLOY_DIR / fname)
            logger.info(f"  Copied {fname}")

    # Copy src directory (minus __pycache__)
    dst_src = DEPLOY_DIR / "src"
    shutil.copytree(SRC_DIR, dst_src, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    logger.info(f"  Copied src/ directory")

    # Copy models
    dst_models = DEPLOY_DIR / "models"
    dst_models.mkdir(parents=True)
    for pattern in ["*.joblib", "*.meta.json", "feature_names.json"]:
        for f in MODELS_DIR.glob(pattern):
            shutil.copy2(f, dst_models / f.name)
            logger.info(f"  Copied models/{f.name}")

    # Copy synthetic data generator
    dst_data = DEPLOY_DIR / "data" / "synthetic"
    dst_data.mkdir(parents=True)
    gen_src = DATA_DIR / "synthetic" / "generator.py"
    if gen_src.exists():
        shutil.copy2(gen_src, dst_data / "generator.py")
        dst_data_init = DEPLOY_DIR / "data" / "synthetic" / "__init__.py"
        if not dst_data_init.exists():
            dst_data_init.write_text("# Synthetic data package\n")
        logger.info(f"  Copied data/synthetic/generator.py")

    # Create requirements.txt if not present (merge hf_space + project reqs)
    req_dst = DEPLOY_DIR / "requirements.txt"
    if not req_dst.exists():
        req_lines = [
            "gradio>=4.0.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "lightgbm>=4.0.0",
            "joblib>=1.3.0",
        ]
        req_dst.write_text("\n".join(req_lines) + "\n")

    logger.info(f"\nDeployment package created at: {DEPLOY_DIR}")
    logger.info(f"Total size: {sum(f.stat().st_size for f in DEPLOY_DIR.rglob('*') if f.is_file()) / 1024:.0f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare deployment package for Hugging Face Spaces"
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help="Create deployment package (default: check readiness only)",
    )
    args = parser.parse_args()

    if args.package:
        print("=" * 50)
        print("  Creating HF Space Deployment Package")
        print("=" * 50)
        create_package()
        return

    print("=" * 50)
    print("  HF Space Deployment Readiness Check")
    print("=" * 50)

    issues = check_readiness()

    if not issues:
        print("\n  All checks passed! Ready for deployment.")
        print()
        print("  Next steps:")
        print("    1. python scripts/prepare_deploy.py --package")
        print("    2. Upload contents of deploy/ to your HF Space")
        print("    3. Or manually copy hf_space/app.py + src/ + models/ to your Space")
        print()
        print("  See Makefile targets:")
        print("    make deploy-check   (this check)")
        print("    make deploy-prepare (create package)")
        print("    make deploy-run     (run Gradio app locally)")
    else:
        print(f"\n  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"    - {issue}")
        print()
        print("  Fix the issues above before deploying.")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
