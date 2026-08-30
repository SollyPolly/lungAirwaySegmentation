"""Collect compute cost for the dissertation's sustainability/ethics reporting.

Two sources, in increasing order of authority:

  --logs  (default)  Retained nnU-Net ``training_log*.txt`` files. Recoverable offline, but
                     only covers runs whose results directory was downloaded, and only
                     training -- not preprocessing, prediction, scoring or failed launches.
                     Treat the output as a FLOOR.

  --pbs DIR          PBS epilogue blocks in job stdout files (``*.o<jobid>``). This is the
                     authoritative record: it reports the walltime the scheduler actually
                     billed, including every job type. Run this on the HPC against the
                     directory the ``#PBS -o`` files were written to.

Neither source measures power. The energy figure is an explicit model:
``energy = walltime x (gpu_draw + host_draw) x PUE``. State the assumptions wherever the
number is quoted -- an unqualified kWh figure is not defensible.
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+):")
EPOCH_T = re.compile(r"Epoch time:\s*([\d.]+)\s*s")

# PBS Pro epilogue lines vary between sites; accept the common spellings.
PBS_WALL = re.compile(r"(?:resources_used\.walltime|Wall-?clock time)\s*[=:]\s*([\d:]+)", re.I)
PBS_CPUT = re.compile(r"(?:resources_used\.cput|CPU time)\s*[=:]\s*([\d:]+)", re.I)


def _hms_to_hours(text: str) -> float:
    parts = [float(p) for p in text.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts[-3:]
    return h + m / 60.0 + s / 3600.0


def scan_training_logs(root: Path) -> list[dict]:
    rows, seen = [], set()
    for path in sorted(root.rglob("training_log*.txt")):
        stamps, epochs = [], []
        for line in path.read_text(errors="replace").splitlines():
            m = TS.match(line)
            if m:
                stamps.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f"))
            e = EPOCH_T.search(line)
            if e:
                epochs.append(float(e.group(1)))
        if not stamps:
            continue
        # A results dir copied to two places is the same job billed once.
        signature = (stamps[0].isoformat(), len(epochs))
        if signature in seen:
            continue
        seen.add(signature)

        parts = path.relative_to(root).parts
        rows.append(
            dict(
                name=next((p for p in parts if p.startswith("Dataset")), parts[0]),
                fold=next((p for p in parts if p.startswith("fold_")), "-"),
                epochs=len(epochs),
                span_h=(stamps[-1] - stamps[0]).total_seconds() / 3600.0,
                loop_h=sum(epochs) / 3600.0,
            )
        )
    return rows


def scan_pbs_outputs(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*.o*")):
        if not re.search(r"\.o\d+$", path.name):
            continue
        text = path.read_text(errors="replace")
        wall = PBS_WALL.search(text)
        if not wall:
            continue
        cput = PBS_CPUT.search(text)
        rows.append(
            dict(
                name=path.name,
                fold="-",
                epochs=0,
                span_h=_hms_to_hours(wall.group(1)),
                loop_h=_hms_to_hours(cput.group(1)) if cput else 0.0,
            )
        )
    return rows


def report(rows: list[dict], *, source: str, gpu_w: float, host_w: float,
           pue: float, grid: float) -> None:
    rows = sorted(rows, key=lambda r: -r["span_h"])
    print(f"{'run':<40} {'fold':<7} {'epochs':>7} {'hours':>8}")
    print("-" * 66)
    for r in rows:
        print(f"{r['name'][:40]:<40} {r['fold']:<7} {r['epochs']:>7} {r['span_h']:>8.2f}")

    total = sum(r["span_h"] for r in rows)
    loop = sum(r["loop_h"] for r in rows)
    print("-" * 66)
    print(f"source                : {source}")
    print(f"jobs / logs           : {len(rows)}")
    print(f"total epochs          : {sum(r['epochs'] for r in rows):,}")
    print(f"TOTAL GPU-hours       : {total:,.1f} h  ({total / 24:,.1f} GPU-days)")
    if loop:
        print(f"  inner loop only     : {loop:,.1f} h")

    kwh = total * (gpu_w + host_w) / 1000.0 * pue
    print(
        f"\nmodelled energy       : {kwh:,.0f} kWh"
        f"  [{gpu_w:.0f} W GPU + {host_w:.0f} W host, PUE {pue}]"
    )
    print(f"modelled emissions    : {kwh * grid:,.0f} kg CO2e  [{grid} kg/kWh]")
    print("\nThese two figures are a MODEL, not a measurement. Quote them with the")
    print("assumptions attached, and say which source the hours came from.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, metavar="DIR",
                    help="scan nnU-Net training logs under DIR (a floor, not a total)")
    ap.add_argument("--pbs", type=Path, metavar="DIR",
                    help="scan PBS job stdout files under DIR (authoritative)")
    ap.add_argument("--gpu-watts", type=float, default=250.0)
    ap.add_argument("--host-watts", type=float, default=100.0)
    ap.add_argument("--pue", type=float, default=1.2,
                    help="datacentre power usage effectiveness; ask RCS for the real value")
    ap.add_argument("--grid-intensity", type=float, default=0.20,
                    help="kg CO2e per kWh for the grid at the time of running")
    args = ap.parse_args()

    if args.pbs:
        rows, source = scan_pbs_outputs(args.pbs), f"PBS epilogue ({args.pbs})"
    else:
        root = args.logs or Path.cwd()
        rows, source = scan_training_logs(root), f"training logs ({root}) -- FLOOR"

    if not rows:
        raise SystemExit("no usable records found; check the directory")

    report(rows, source=source, gpu_w=args.gpu_watts, host_w=args.host_watts,
           pue=args.pue, grid=args.grid_intensity)


if __name__ == "__main__":
    main()
