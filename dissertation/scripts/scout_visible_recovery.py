"""Scout visually coherent, reference-supported additions made by Soft-clDice.

Unlike a TLD-only scout, this ranks connected volumes that are (1) inside the annotated
airway, (2) predicted by Soft-clDice, and (3) absent from the no-consistency prediction.
It favours elongated components containing airway-core voxels, which makes a genuine tube
segment rank above a one-voxel surface displacement.  The output is only a reproducible
qualitative-case selection aid; it is not a new performance endpoint.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import figure_arms as arms  # noqa: E402
import render_case_gallery as gallery  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=arms.COHORTS, default="val")
    parser.add_argument("--treatment", default="soft_f0")
    parser.add_argument("--comparator", default="control")
    parser.add_argument("--top-cases", type=int, default=10)
    parser.add_argument("--minimum-component-voxels", type=int, default=12)
    parser.add_argument("--core-depth-mm", type=float, default=1.5)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "dissertation" / "Figures" / "provenance" /
        "visible_recovery_candidates.csv",
    )
    return parser.parse_args()


def _prediction_path(directory: Path, case_name: str) -> Path:
    for name in (f"{case_name}.nii.gz", f"{case_name}_0000.nii.gz"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No prediction for {case_name} in {directory}")


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(path)
    return np.asanyarray(image.dataobj) > 0, np.asarray(image.header.get_zooms()[:3], float)


def _component_geometry(coordinates: np.ndarray, spacing: np.ndarray) -> tuple[float, float]:
    physical = coordinates.astype(float) * spacing
    if len(physical) < 3:
        return 0.0, 1.0
    centred = physical - physical.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centred.T @ centred / len(centred))
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    length_proxy = float(4.0 * np.sqrt(eigenvalues[-1]))
    elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[-2]))
    return length_proxy, elongation


def main() -> None:
    args = _parse_args()
    control_dir = arms.prediction_dir(args.comparator, args.cohort)
    treatment_dir = arms.prediction_dir(args.treatment, args.cohort)
    if control_dir is None or treatment_dir is None:
        raise SystemExit("Both requested prediction arms must exist.")

    ranking = gallery.rank_cases(
        args.treatment, args.comparator, args.cohort, "td_raw"
    )[::-1][:args.top_cases]
    rows: list[dict[str, object]] = []
    structure = ndi.generate_binary_structure(3, 3)

    for rank_row in ranking:
        case_id = rank_row["case_id"]
        case_name = f"ATM_{case_id}" if args.cohort != "ood" else case_id
        reference_path = ROOT / "data" / "ATM22" / "labelsTr" / f"{case_name}_0000.nii.gz"
        reference, spacing = _load(reference_path)
        control, _ = _load(_prediction_path(control_dir, case_name))
        treatment, _ = _load(_prediction_path(treatment_dir, case_name))

        foreground = np.argwhere(reference)
        padding = np.ceil(5.0 / spacing).astype(int)
        start = np.maximum(foreground.min(axis=0) - padding, 0)
        stop = np.minimum(foreground.max(axis=0) + padding + 1, reference.shape)
        crop = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
        local_reference = reference[crop]
        local_control = control[crop]
        local_treatment = treatment[crop]
        del reference, control, treatment

        added_true = local_reference & local_treatment & ~local_control
        labels, count = ndi.label(added_true, structure=structure)
        objects = ndi.find_objects(labels)
        radius = ndi.distance_transform_edt(local_reference, sampling=spacing)
        component_sizes = np.bincount(labels.ravel())

        for component_id, component_crop in enumerate(objects, start=1):
            size = int(component_sizes[component_id])
            if component_crop is None or size < args.minimum_component_voxels:
                continue
            local_labels = labels[component_crop]
            component = local_labels == component_id
            coordinates = np.argwhere(component)
            offset = np.asarray([part.start for part in component_crop], dtype=int)
            coordinates += offset
            core = radius[tuple(coordinates.T)] >= args.core_depth_mm
            core_voxels = int(core.sum())
            length_proxy, elongation = _component_geometry(coordinates, spacing)
            mean_radius = float(radius[tuple(coordinates.T)].mean())
            centre = coordinates.mean(axis=0) + start
            # A visible airway segment should have volume, a core, and longitudinal
            # extent. A pure wall disagreement has little or no core support.
            score = size * (1.0 + core_voxels / size) * np.sqrt(max(length_proxy, 0.1))
            rows.append(
                {
                    "case_id": case_id,
                    "patient_tld_gain": rank_row["difference"],
                    "component_id": component_id,
                    "score": score,
                    "added_reference_voxels": size,
                    "airway_core_voxels": core_voxels,
                    "core_fraction": core_voxels / size,
                    "length_proxy_mm": length_proxy,
                    "elongation": elongation,
                    "mean_reference_radius_mm": mean_radius,
                    "centre_i": float(centre[0]),
                    "centre_j": float(centre[1]),
                    "centre_k": float(centre[2]),
                }
            )
        print(f"Scouted {case_name}: {count} connected added-reference components", flush=True)

    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nTop visually coherent reference-supported additions:")
    print(
        f"{'case':>5} {'TLD+':>7} {'vox':>7} {'core':>7} {'core%':>7} "
        f"{'len mm':>8} {'elong':>7} {'r mm':>6}"
    )
    for row in rows[:25]:
        print(
            f"{row['case_id']:>5} {row['patient_tld_gain']:>+7.4f} "
            f"{row['added_reference_voxels']:>7} {row['airway_core_voxels']:>7} "
            f"{100 * row['core_fraction']:>6.1f}% {row['length_proxy_mm']:>8.1f} "
            f"{row['elongation']:>7.2f} {row['mean_reference_radius_mm']:>6.2f}"
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
