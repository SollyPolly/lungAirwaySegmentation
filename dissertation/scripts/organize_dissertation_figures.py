"""Organise dissertation figure assets without deleting source material.

Generated scientific PDFs and PNG previews are produced directly in the hierarchy
used by LaTeX.  This helper moves only the fixed branding assets and reference PDF,
copies the blue title-page logo to its canonical name, and archives legacy flat
generated figures after their replacements have been regenerated.
"""

from __future__ import annotations

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DISSERTATION = ROOT / "dissertation"
FIGURES = DISSERTATION / "Figures"


def _move_if_present(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() == destination.read_bytes():
            source.unlink()
            return
        raise FileExistsError(f"Refusing to overwrite different file: {destination}")
    source.replace(destination)


def _copy_if_changed(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and source.read_bytes() == destination.read_bytes():
        return
    shutil.copy2(source, destination)


def main() -> None:
    for directory in (
        FIGURES / "branding" / "source",
        FIGURES / "pdf" / "background",
        FIGURES / "pdf" / "methods" / "cldice",
        FIGURES / "pdf" / "methods" / "mean_teacher",
        FIGURES / "pdf" / "analysis" / "teacher_targets",
        FIGURES / "pdf" / "appendix",
        FIGURES / "png" / "background",
        FIGURES / "png" / "methods" / "cldice",
        FIGURES / "png" / "methods" / "mean_teacher",
        FIGURES / "png" / "analysis" / "teacher_targets",
        FIGURES / "png" / "appendix",
        FIGURES / "provenance",
        DISSERTATION / "reference",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    blue_source = FIGURES / "ImperialLogo_blue2024.png"
    if not blue_source.exists():
        blue_source = FIGURES / "ImperialLogo.png"
    _copy_if_changed(blue_source, FIGURES / "branding" / "imperial_logo_blue_2024.png")
    _move_if_present(FIGURES / "ImperialLogo_pride.png", FIGURES / "branding" / "imperial_logo_pride.png")
    _move_if_present(
        FIGURES / "ImperialLogo_pride_source.webp",
        FIGURES / "branding" / "source" / "imperial_logo_pride.webp",
    )

    for redundant in (FIGURES / "ImperialLogo.png", FIGURES / "ImperialLogo_blue2024.png"):
        if redundant.exists():
            redundant.unlink()

    _move_if_present(
        DISSERTATION / "MSc_Project_booklet_2025-26.pdf",
        DISSERTATION / "reference" / "MSc_Project_booklet_2025-26.pdf",
    )

    # These flat assets are superseded by regenerated, organised copies. Move them
    # into a legacy directory rather than deleting them so no prior illustration is
    # lost while the dissertation is still being drafted.
    legacy = FIGURES / "legacy_flat"
    for source in sorted(FIGURES.glob("cldice_*")) + sorted(FIGURES.glob("mean_teacher_*")):
        if source.is_file():
            _move_if_present(source, legacy / source.name)

    print(f"Organised dissertation assets under {FIGURES}")


if __name__ == "__main__":
    main()
