"""Render docs/current_model_overview.md to a simple PDF.

The renderer is intentionally lightweight so it can run inside the project
environment without extra PDF dependencies. It supports the subset of Markdown
used in the explainer document: headings, paragraphs, and flat bullet lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lung_airway_segmentation.settings import PROJECT_ROOT


INPUT_PATH = PROJECT_ROOT / "docs" / "current_model_overview.md"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "current_model_overview.pdf"


@dataclass
class Block:
    kind: str
    text: str
    level: int = 0


def parse_markdown(path: Path) -> list[Block]:
    """Parse the small markdown subset used by the explainer."""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            paragraph = " ".join(line.strip() for line in paragraph_lines if line.strip())
            if paragraph:
                blocks.append(Block(kind="paragraph", text=paragraph))
            paragraph_lines.clear()

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append(Block(kind="heading", text=stripped[level:].strip(), level=level))
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            blocks.append(Block(kind="bullet", text=stripped[2:].strip()))
            continue

        paragraph_lines.append(stripped)

    flush_paragraph()
    return blocks


def wrap_lines(text: str, width: int) -> list[str]:
    """Wrap one text block without breaking long words aggressively."""
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]


def new_page(pdf: PdfPages):
    """Create one new A4 page and return the figure, axis, and cursor position."""
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax, 0.95


def render_blocks(blocks: list[Block], output_path: Path) -> None:
    """Render parsed blocks to PDF with simple pagination."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        fig, ax, y = new_page(pdf)

        def ensure_space(required_height: float):
            nonlocal fig, ax, y
            if y - required_height < 0.05:
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax, y = new_page(pdf)

        def draw_lines(lines: list[str], *, x: float, fontsize: int, line_height: float):
            nonlocal y
            for line in lines:
                ax.text(
                    x,
                    y,
                    line,
                    fontsize=fontsize,
                    va="top",
                    ha="left",
                    family="DejaVu Sans",
                )
                y -= line_height

        for index, block in enumerate(blocks):
            if block.kind == "heading":
                if block.level == 1:
                    fontsize = 22
                    line_height = 0.042
                    width = 46
                    gap_after = 0.018
                elif block.level == 2:
                    fontsize = 15
                    line_height = 0.030
                    width = 62
                    gap_after = 0.010
                else:
                    fontsize = 13
                    line_height = 0.026
                    width = 72
                    gap_after = 0.008

                lines = wrap_lines(block.text, width)
                required = len(lines) * line_height + gap_after + 0.008
                if index > 0:
                    required += 0.008
                ensure_space(required)
                if index > 0:
                    y -= 0.008
                draw_lines(lines, x=0.07, fontsize=fontsize, line_height=line_height)
                y -= gap_after

            elif block.kind == "paragraph":
                lines = wrap_lines(block.text, 92)
                line_height = 0.021
                required = len(lines) * line_height + 0.012
                ensure_space(required)
                draw_lines(lines, x=0.07, fontsize=11, line_height=line_height)
                y -= 0.012

            elif block.kind == "bullet":
                wrapped = wrap_lines(block.text, 82)
                lines = [f"• {wrapped[0]}"] + [f"  {line}" for line in wrapped[1:]]
                line_height = 0.021
                required = len(lines) * line_height + 0.006
                ensure_space(required)
                draw_lines(lines, x=0.09, fontsize=11, line_height=line_height)
                y -= 0.006

        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    blocks = parse_markdown(INPUT_PATH)
    render_blocks(blocks, OUTPUT_PATH)
    print(f"Saved PDF: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
