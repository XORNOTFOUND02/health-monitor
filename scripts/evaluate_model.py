#!/usr/bin/env python3
"""
Comprehensive model evaluation CLI.

Generates publication-quality evaluation charts with accurate statistics
for all 7 health conditions across 3 model groups (cardiac, respiratory, activity).

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --output-dir evaluation_results --sessions 30 --thresholds 0.3,0.5,0.7
    python scripts/evaluate_model.py --skip-charts --quick
    python scripts/evaluate_model.py --open-report   # Open the report after generation
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.chdir(str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_model")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Comprehensive model evaluation with publication-quality charts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_model.py
  python scripts/evaluate_model.py --output-dir evaluation_results --sessions 50
  python scripts/evaluate_model.py --quick --skip-charts
  python scripts/evaluate_model.py --open-report
        """,
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results",
        help="Output directory for charts and reports (default: evaluation_results)",
    )
    parser.add_argument(
        "--models-dir", type=str, default="models",
        help="Directory with trained model files (default: models)",
    )
    parser.add_argument(
        "--sessions", type=int, default=30,
        help="Number of synthetic test sessions to generate (default: 30)",
    )
    parser.add_argument(
        "--thresholds", type=str, default="0.3,0.5,0.7",
        help="Comma-separated decision thresholds to test (default: 0.3,0.5,0.7)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 5 sessions, fewer charts",
    )
    parser.add_argument(
        "--skip-charts", action="store_true",
        help="Skip chart generation (metrics only)",
    )
    parser.add_argument(
        "--open-report", action="store_true",
        help="Attempt to open the report in browser after generation",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    num_sessions = 5 if args.quick else args.sessions
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    print()
    print("=" * 70)
    print("  HEALTH MONITOR — COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)
    print(f"  Output dir    : {output_dir}")
    print(f"  Models dir    : {models_dir}")
    print(f"  Test sessions : {num_sessions}")
    print(f"  Thresholds    : {thresholds}")
    print(f"  Generate charts: {'no' if args.skip_charts else 'yes'}")
    print("=" * 70)
    print()

    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Generate test data
    # ------------------------------------------------------------------
    print("[Step 1/4] Generating test data ...")
    from data.synthetic.generator import SyntheticDataGenerator
    gen = SyntheticDataGenerator(seed=args.seed)
    test_data = []  # list of (window_data, ground_truth_dict)
    for i in range(num_sessions):
        # Alternate between conditions for balanced test set
        conditions = [
            "normal", "tachycardia", "low_spo2", "fever",
            "fall_detected", "fatigue", "sleep_problem", "irregular_rhythm"
        ]
        cond = conditions[i % len(conditions)]
        try:
            if cond == "normal":
                session = gen.generate_normal_session(duration_sec=30)
            else:
                session = gen.generate_condition_session(cond, duration_sec=30, severity="moderate")
            window = session["windows"][0]["sensor_data"]
            gt = session.get("ground_truth", {})
            gt[cond] = True  # Ensure ground truth reflects the simulated condition
            test_data.append((window, gt))
        except Exception as e:
            logger.warning(f"Failed to generate {cond} session: {e}")

    print(f"  -> {len(test_data)} test windows generated")

    # ------------------------------------------------------------------
    # Step 2: Run inference
    # ------------------------------------------------------------------
    print("[Step 2/4] Running inference on test data ...")
    from src.inference import Predictor
    predictor = Predictor(models_dir=str(models_dir))

    # Map model output conditions
    cardiac_conditions = ["tachycardia", "irregular_rhythm"]
    respiratory_conditions = ["low_spo2"]
    activity_conditions = ["fall_detected", "sleep_problem", "fatigue"]
    all_conditions = cardiac_conditions + respiratory_conditions + activity_conditions + ["fever"]
    ml_conditions = [c for c in all_conditions if c != "fever"]

    # Collect predictions per condition
    y_true_dict = {c: [] for c in all_conditions}
    y_score_dict = {c: [] for c in all_conditions}

    for window_data, ground_truth in test_data:
        try:
            normalized = predictor._normalize_input(window_data)
            features = predictor._extract_features(normalized)
            X = predictor._align_features(features)
            raw_preds = predictor.predict(window_data)

            for cond in all_conditions:
                y_true_dict[cond].append(float(ground_truth.get(cond, False)))
                prob = raw_preds.get(cond, {}).get("probability", 0.0)
                y_score_dict[cond].append(prob)
        except Exception as e:
            logger.warning(f"Inference failed for a window: {e}")
            continue

    # Convert to numpy arrays
    for cond in all_conditions:
        y_true_dict[cond] = np.array(y_true_dict[cond], dtype=np.float64)
        y_score_dict[cond] = np.array(y_score_dict[cond], dtype=np.float64)

    print(f"  -> Predictions collected for {len(all_conditions)} conditions")
    for cond in all_conditions:
        n_pos = int(y_true_dict[cond].sum())
        print(f"     {cond:20s}: {len(y_true_dict[cond])} samples, {n_pos} positive")

    # ------------------------------------------------------------------
    # Step 3: Compute metrics for each condition and threshold
    # ------------------------------------------------------------------
    print("[Step 3/4] Computing evaluation metrics ...")
    from src.evaluation import MetricsCalculator, EvaluationReport

    calc = MetricsCalculator()

    # Per-condition, per-threshold metrics
    all_results = {}
    for cond in all_conditions:
        y_true = y_true_dict[cond]
        y_score = y_score_dict[cond]
        if len(y_true) == 0:
            continue

        cond_results = {"threshold_analysis": {}}
        for thresh in thresholds:
            metrics = calc.compute_all(y_true, y_score, threshold=thresh)
            cond_results["threshold_analysis"][f"thresh_{thresh}"] = metrics

        # Also store full data for charting
        cond_results["y_true"] = y_true.tolist()
        cond_results["y_score"] = y_score.tolist()
        cond_results["n_samples"] = len(y_true)
        cond_results["n_positive"] = int(y_true.sum())

        all_results[cond] = cond_results

    # Print summary
    print("\n  Per-condition summary (threshold=0.5):")
    print(f"  {'Condition':20s} {'Acc':>6s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'AUC':>6s} {'Brier':>6s}")
    print(f"  {'-'*58}")
    for cond in all_conditions:
        if cond not in all_results:
            continue
        m = all_results[cond]["threshold_analysis"].get("thresh_0.5", {})
        acc = m.get("accuracy", 0)
        prec = m.get("precision", 0)
        rec = m.get("recall", 0)
        f1 = m.get("f1", 0)
        auc = m.get("auc_roc", 0)
        brier = m.get("brier_score", 0)
        print(f"  {cond:20s} {acc:6.3f} {prec:6.3f} {rec:6.3f} {f1:6.3f} {auc:6.3f} {brier:6.3f}")

    # ------------------------------------------------------------------
    # Step 4: Generate charts
    # ------------------------------------------------------------------
    if not args.skip_charts:
        print("\n[Step 4/4] Generating evaluation charts ...")
        output_dir.mkdir(parents=True, exist_ok=True)
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        from src.evaluation import EvaluationVisualizer
        vis = EvaluationVisualizer()

        chart_count = 0

        for cond in all_conditions:
            if cond not in all_results:
                continue
            y_true = np.array(all_results[cond]["y_true"])
            y_score = np.array(all_results[cond]["y_score"])
            y_pred = (y_score >= 0.5).astype(np.float64)

            if len(np.unique(y_true)) < 2:
                logger.info(f"  Skipping charts for {cond}: only one class in test data")
                continue

            try:
                # Determine model group for naming
                if cond in cardiac_conditions:
                    group = "cardiac"
                elif cond in respiratory_conditions:
                    group = "respiratory"
                elif cond in activity_conditions:
                    group = "activity"
                else:
                    group = "rule_engine"

                base_name = f"{group}_{cond}"

                # 1. Confusion Matrix
                vis.plot_confusion_matrix(
                    y_true, y_pred,
                    str(charts_dir / f"{base_name}_confusion_matrix.png"),
                    title=f"{cond.replace('_', ' ').title()} - Confusion Matrix",
                )
                chart_count += 1

                # 2. ROC Curve
                vis.plot_roc_curve(
                    y_true, y_score,
                    str(charts_dir / f"{base_name}_roc_curve.png"),
                    title=f"{cond.replace('_', ' ').title()} - ROC Curve",
                )
                chart_count += 1

                # 3. Precision-Recall Curve
                vis.plot_precision_recall_curve(
                    y_true, y_score,
                    str(charts_dir / f"{base_name}_pr_curve.png"),
                    title=f"{cond.replace('_', ' ').title()} - Precision-Recall Curve",
                )
                chart_count += 1

                # 4. Calibration Curve
                vis.plot_calibration_curve(
                    y_true, y_score,
                    str(charts_dir / f"{base_name}_calibration.png"),
                    title=f"{cond.replace('_', ' ').title()} - Calibration Curve",
                )
                chart_count += 1

                # 5. Threshold Analysis
                vis.plot_threshold_analysis(
                    y_true, y_score,
                    str(charts_dir / f"{base_name}_threshold_analysis.png"),
                    title=f"{cond.replace('_', ' ').title()} - Threshold Analysis",
                )
                chart_count += 1

            except Exception as e:
                logger.warning(f"Chart generation failed for {cond}: {e}")
                import traceback
                traceback.print_exc()

        # 6. Radar Chart (cross-condition comparison)
        try:
            radar_data = {}
            for cond in all_conditions:
                if cond not in all_results:
                    continue
                m = all_results[cond]["threshold_analysis"].get("thresh_0.5", {})
                radar_data[cond] = {
                    "Accuracy": m.get("accuracy", 0),
                    "Precision": m.get("precision", 0),
                    "Recall": m.get("recall", 0),
                    "F1 Score": m.get("f1", 0),
                    "AUC": m.get("auc_roc", 0),
                }
            vis.plot_radar_chart(
                radar_data,
                str(charts_dir / "radar_comparison.png"),
                title="Model Performance Comparison - All Conditions",
            )
            chart_count += 1
        except Exception as e:
            logger.warning(f"Radar chart failed: {e}")

        # 7. Feature Importance (from loaded models)
        try:
            for group_name, group_models, model_file in [
                ("cardiac", cardiac_conditions, "cardiac.joblib"),
                ("activity", activity_conditions, "activity.joblib"),
            ]:
                import joblib
                model_path = models_dir / model_file
                if model_path.exists():
                    models_dict = joblib.load(model_path)
                    for cond_name, booster in models_dict.items():
                        if hasattr(booster, 'feature_importance'):
                            importances = booster.feature_importance(importance_type='gain')
                            fn_path = models_dir / "feature_names.json"
                            if fn_path.exists():
                                with open(fn_path) as f:
                                    fnames = json.load(f)
                                # Truncate to match
                                fnames = fnames[:len(importances)]
                                vis.plot_feature_importance(
                                    fnames, importances,
                                    str(charts_dir / f"{group_name}_{cond_name}_feature_importance.png"),
                                    title=f"{cond_name.replace('_', ' ').title()} - Feature Importance",
                                )
                                chart_count += 1
        except Exception as e:
            logger.warning(f"Feature importance charts failed: {e}")

        # 8. MCB-DSC Plot
        try:
            mcb_dsc_data = {}
            for cond in all_conditions:
                if cond not in all_results:
                    continue
                m = all_results[cond]["threshold_analysis"].get("thresh_0.5", {})
                mcb = m.get("brier_score", 0) - 0.0  # simplified MCB
                dsc = m.get("auc_roc", 0)
                mcb_dsc_data[cond] = {"MCB": mcb, "DSC": dsc, "Brier": m.get("brier_score", 0)}
            vis.plot_mcb_dsc(
                mcb_dsc_data,
                str(charts_dir / "mcb_dsc_plot.png"),
                title="MCB-DSC Plot - Discrimination vs Calibration",
            )
            chart_count += 1
        except Exception as e:
            logger.warning(f"MCB-DSC plot failed: {e}")

        # 9. Create Dashboard
        try:
            # Build model_groups_data dict for dashboard
            dashboard_data = {}
            for cond in all_conditions:
                if cond not in all_results:
                    continue
                dashboard_data[cond] = {
                    "y_true": np.array(all_results[cond].get("y_true", [])),
                    "y_score": np.array(all_results[cond].get("y_score", [])),
                }
            vis.create_evaluation_dashboard(
                str(charts_dir),
                dashboard_data,
                save_path=str(output_dir / "evaluation_dashboard.png"),
            )
            chart_count += 1
        except Exception as e:
            logger.warning(f"Dashboard creation failed: {e}")

        print(f"  -> {chart_count} charts generated in {charts_dir}")
    else:
        print("[Step 4/4] Chart generation skipped (--skip-charts)")

    # ------------------------------------------------------------------
    # Save evaluation report
    # ------------------------------------------------------------------
    print("\nSaving evaluation report ...")
    report_path = output_dir / "evaluation_report.json"

    # Clean up: remove raw arrays for JSON serialization
    serializable_results = {}
    for cond, data in all_results.items():
        serializable_results[cond] = {
            "threshold_analysis": data["threshold_analysis"],
            "n_samples": data["n_samples"],
            "n_positive": data["n_positive"],
        }

    report = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_test_sessions": num_sessions,
            "thresholds_tested": thresholds,
            "models_dir": str(models_dir),
            "model_versions": predictor.model_versions,
        },
        "results": serializable_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to {report_path}")

    # Text summary
    txt_path = output_dir / "evaluation_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  HEALTH MONITOR — MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {report['metadata']['generated_at']}\n")
        f.write(f"Test sessions: {num_sessions}\n")
        f.write(f"Thresholds: {thresholds}\n\n")

        f.write("Per-Condition Performance (threshold=0.5):\n")
        f.write(f"{'Condition':20s} {'Acc':>6s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'AUC':>6s} {'Brier':>6s}\n")
        f.write("-" * 58 + "\n")
        for cond in all_conditions:
            if cond not in all_results:
                continue
            m = all_results[cond]["threshold_analysis"].get("thresh_0.5", {})
            f.write(f"{cond:20s} {m.get('accuracy',0):6.3f} {m.get('precision',0):6.3f} ")
            f.write(f"{m.get('recall',0):6.3f} {m.get('f1',0):6.3f} {m.get('auc_roc',0):6.3f} {m.get('brier_score',0):6.3f}\n")

        f.write("\n\nThreshold Comparison (macro avg F1):\n")
        for thresh in thresholds:
            f1s = []
            for cond in all_conditions:
                if cond not in all_results:
                    continue
                m = all_results[cond]["threshold_analysis"].get(f"thresh_{thresh}", {})
                f1s.append(m.get("f1", 0))
            macro_f1 = np.mean(f1s) if f1s else 0
            f.write(f"  Threshold {thresh:.1f}: macro F1 = {macro_f1:.4f}\n")

        f.write("\n\nModel Versions:\n")
        for name, version in predictor.model_versions.items():
            f.write(f"  {name}: {version}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"  Summary saved to {txt_path}")

    total_time = time.time() - t_start
    print()
    print("=" * 70)
    print(f"  EVALUATION COMPLETE ({total_time:.1f}s)")
    print(f"  Results: {report_path}")
    print(f"  Charts:  {output_dir / 'charts'}")
    print(f"  Dashboard: {output_dir / 'evaluation_dashboard.png'}")
    print("=" * 70)

    # Open report if requested
    if args.open_report:
        try:
            import webbrowser
            webbrowser.open(str(report_path))
        except Exception:
            pass


if __name__ == "__main__":
    main()
