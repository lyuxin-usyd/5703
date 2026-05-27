from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METHOD_ORDER = [
    "ergo",
    "est",
    "event_pretraining",
    "evrepsl",
    "get",
    "matrixlstm",
    "event_frame",
    "event_count",
    "binary_event_image",
    "timestamp_image",
    "time_surface",
    "voxel_grid",
]
METHOD_LABELS = {
    "ergo": "ERGO",
    "est": "EST",
    "event_pretraining": "Event Pre-training",
    "evrepsl": "EvRepSL",
    "get": "GET",
    "matrixlstm": "MatrixLSTM",
    "event_frame": "Event Frame",
    "event_count": "Event Count",
    "binary_event_image": "Binary Event Image",
    "timestamp_image": "Timestamp Image",
    "time_surface": "Time Surface",
    "voxel_grid": "Voxel Grid",
}


def _load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    method = str(data["adapter_name"])
    file_value = path.as_posix()
    try:
        file_value = path.relative_to(path.parents[1]).as_posix()
    except ValueError:
        pass
    return {
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "aee": float(data["aee"]),
        "outlier_percent": float(data["outlier_percent"]),
        "best_val_aee": float(data.get("best_val_aee") or 0.0),
        "best_epoch": int(data.get("best_epoch") or 0),
        "epochs_completed": int(data.get("epochs_completed") or 0),
        "early_stopped": bool(data.get("early_stopped")),
        "train_windows": int(data.get("train_windows") or 0),
        "eval_windows": int(data.get("eval_windows") or 0),
        "metric_scope": str(data.get("metric_scope") or ""),
        "per_sequence_metrics": data.get("per_sequence_metrics") or {},
        "file": file_value,
    }


def _ordered_rows(result_dir: Path) -> list[dict[str, Any]]:
    rows_by_method = {}
    for path in sorted(result_dir.glob("*.json")):
        row = _load_result(path)
        rows_by_method[row["method"]] = row
    if not rows_by_method:
        raise SystemExit(f"No result JSON files found in {result_dir}")
    known = [rows_by_method[method] for method in METHOD_ORDER if method in rows_by_method]
    extras = [
        rows_by_method[method]
        for method in sorted(rows_by_method)
        if method not in METHOD_ORDER
    ]
    return known + extras


def _write_summary_csv(rows: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "method",
        "aee",
        "outlier_percent",
        "best_val_aee",
        "best_epoch",
        "epochs_completed",
        "early_stopped",
        "train_windows",
        "eval_windows",
        "metric_scope",
        "file",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _per_sequence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        per_sequence = row.get("per_sequence_metrics") or {}
        for sequence, metrics in sorted(per_sequence.items()):
            output_rows.append(
                {
                    "method": row["method"],
                    "label": row["label"],
                    "sequence": sequence,
                    "windows": int(metrics.get("windows") or 0),
                    "aee": float(metrics.get("aee") or 0.0),
                    "outlier_percent": float(metrics.get("outlier_percent") or 0.0),
                    "valid_count": int(metrics.get("valid_count") or 0),
                    "outlier_count": int(metrics.get("outlier_count") or 0),
                }
            )
    return output_rows


def _write_per_sequence_csv(rows: list[dict[str, Any]], output: Path) -> None:
    seq_rows = _per_sequence_rows(rows)
    if not seq_rows:
        return
    fields = [
        "method",
        "sequence",
        "windows",
        "aee",
        "outlier_percent",
        "valid_count",
        "outlier_count",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in seq_rows:
            writer.writerow({field: row[field] for field in fields})


def _write_summary_md(rows: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# Photometric + Smoothness V9 MatrixLSTM-Style Full-Split Summary",
        "",
        "Dataset: MVSEC `outdoor_day1 + outdoor_day2` for training and `indoor_flying1/2/3` for evaluation.",
        "",
        "V9 defaults: EV-FlowNet-style multi-scale decoder, MatrixLSTM-style image-pair windows, raw 0-255 grayscale images, train image strides 1..5 sampled lazily per epoch, 256 crop, random flip/rotation augmentation, self-supervised image-pair photometric warping loss, and flow smoothness loss. Ground-truth flow is used only for validation and evaluation metrics.",
        "",
        "Lower is better for AEE and Outlier %.",
        "",
        "| Method | AEE | Outlier % | Best val AEE | Best epoch | Epochs | Train windows | Eval windows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {aee:.4f} | {outlier_percent:.2f} | {best_val_aee:.4f} | {best_epoch} | {epochs_completed} | {train_windows} | {eval_windows} |".format(
                **row
            )
        )
    seq_rows = _per_sequence_rows(rows)
    if seq_rows:
        lines.extend(
            [
                "",
                "## Per-Sequence Breakdown",
                "",
                "| Method | Sequence | AEE | Outlier % | Windows | Valid pixels | Outlier pixels |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in seq_rows:
            lines.append(
                "| {label} | {sequence} | {aee:.4f} | {outlier_percent:.2f} | {windows} | {valid_count} | {outlier_count} |".format(
                    **row
                )
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_curve(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "epoch": float(row["epoch"]),
                    "val_aee": float(row["val_aee"]),
                    "best_val_aee": float(row["best_val_aee"]),
                }
            )
    return rows


def _plot_curves(curve_dir: Path, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        (output_dir / "PLOT_SKIPPED.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    for key, ylabel, filename in [
        ("val_aee", "Validation AEE", "validation_aee_curves.png"),
        ("best_val_aee", "Best validation AEE", "best_validation_aee_curves.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        plotted = False
        for method in METHOD_ORDER:
            candidates = sorted(curve_dir.glob(f"v9_{method}_*.csv"))
            if not candidates:
                continue
            curve = _load_curve(candidates[-1])
            ax.plot(
                [row["epoch"] for row in curve],
                [row[key] for row in curve],
                linewidth=1.8,
                label=METHOD_LABELS.get(method, method),
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Photometric + smoothness full split {ylabel}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def _resolve_artifact_path(run_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = run_root / value
    return candidate


def _collect_inference_rows(run_root: Path) -> list[dict[str, str]]:
    artifact_root = run_root / "artifacts"
    rows: list[dict[str, str]] = []
    if not artifact_root.exists():
        return rows
    for manifest in sorted(artifact_root.glob("*/sample_manifest.csv")):
        with manifest.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rows.append({key: value for key, value in row.items()})
    return rows


def _write_all_manifest(rows: list[dict[str, str]], output: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _choose_common_sample(rows: list[dict[str, str]]) -> str | None:
    grouped: dict[str, set[str]] = {}
    for row in rows:
        artifact_id = row.get("artifact_id")
        method = row.get("method")
        if not artifact_id or not method:
            continue
        grouped.setdefault(artifact_id, set()).add(method)
    if not grouped:
        return None
    return sorted(grouped.items(), key=lambda item: (len(item[1]), item[0]), reverse=True)[0][0]


def _plot_inference_grid(
    rows: list[dict[str, str]],
    *,
    run_root: Path,
    image_key: str,
    output: Path,
    title: str,
    include_gt: bool,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        (run_root / "INFERENCE_GRID_SKIPPED.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    artifact_id = _choose_common_sample(rows)
    if artifact_id is None:
        return
    selected = [row for row in rows if row.get("artifact_id") == artifact_id]
    by_method = {row.get("method", ""): row for row in selected}
    ordered_methods = [method for method in METHOD_ORDER if method in by_method]
    ordered_methods.extend(method for method in sorted(by_method) if method not in METHOD_ORDER)
    images: list[tuple[str, Path]] = []
    if include_gt and selected:
        gt_path = selected[0].get("gt_flow_png")
        if gt_path:
            images.append(("GT flow", _resolve_artifact_path(run_root, gt_path)))
    for method in ordered_methods:
        path_value = by_method[method].get(image_key)
        if not path_value:
            continue
        images.append((METHOD_LABELS.get(method, method), _resolve_artifact_path(run_root, path_value)))
    images = [(label, path) for label, path in images if path.exists()]
    if not images:
        return

    cols = min(4, len(images))
    rows_count = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(3.2 * cols, 3.2 * rows_count))
    axes_list = np.asarray(axes, dtype=object).reshape(-1)
    for ax in axes_list:
        ax.axis("off")
    for ax, (label, path) in zip(axes_list, images):
        ax.imshow(plt.imread(path))
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{title}\n{artifact_id}", fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build summary tables and curve figures for a photometric full-split run.")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root
    if run_root.name == "results":
        raise SystemExit("Refusing to overwrite bundled IF1 results. Pass a full run root such as results/photometric_full_YYYYMMDD_HHMM.")
    result_dir = run_root / "results"
    curve_dir = run_root / "curves"
    rows = _ordered_rows(result_dir)
    _write_summary_csv(rows, run_root / "summary.csv")
    _write_per_sequence_csv(rows, run_root / "per_sequence_summary.csv")
    _write_summary_md(rows, run_root / "summary.md")
    _plot_curves(curve_dir, run_root)
    inference_rows = _collect_inference_rows(run_root)
    if inference_rows:
        _write_all_manifest(inference_rows, run_root / "sample_manifest.csv")
        _plot_inference_grid(
            inference_rows,
            run_root=run_root,
            image_key="pred_flow_png",
            output=run_root / "figures" / "mvsec_flow_inference_grid.png",
            title="MVSEC fixed-sample predicted optical flow",
            include_gt=True,
        )
        _plot_inference_grid(
            inference_rows,
            run_root=run_root,
            image_key="error_map_png",
            output=run_root / "figures" / "mvsec_flow_error_grid.png",
            title="MVSEC fixed-sample endpoint-error maps",
            include_gt=False,
        )
    print(f"wrote {run_root / 'summary.csv'}")
    print(f"wrote {run_root / 'per_sequence_summary.csv'}")
    print(f"wrote {run_root / 'summary.md'}")


if __name__ == "__main__":
    main()
