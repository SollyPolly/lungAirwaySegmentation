$ErrorActionPreference = 'Stop'

# This script lives in dissertation/scripts/, so its parent IS the dissertation root.
$Dissertation = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Source = Join-Path $Dissertation 'Figures\src\methods\mean_teacher'
$Build = Join-Path $Dissertation 'build\figures\mean_teacher'
$PdfOut = Join-Path $Dissertation 'Figures\pdf\methods\mean_teacher'
$PngOut = Join-Path $Dissertation 'Figures\png\methods\mean_teacher'

New-Item -ItemType Directory -Force -Path $Build, $PdfOut, $PngOut | Out-Null

# Compile from the figure source directory itself. The standalone sources use a
# same-directory \input, so this is the one working directory that satisfies both
# this script and an editor build (latex-workshop) of the same file.
Push-Location $Source
try {
    foreach ($Name in @('mean_teacher_flow', 'nnunet_backbone')) {
        & pdflatex `
            -interaction=nonstopmode `
            -halt-on-error `
            "-output-directory=$Build" `
            "$Name.tex"
        if ($LASTEXITCODE -ne 0) {
            throw "TikZ compilation failed for $Name."
        }

        $BuiltPdf = Join-Path $Build "$Name.pdf"
        $PublicationPdf = Join-Path $PdfOut "$Name.pdf"
        Copy-Item -LiteralPath $BuiltPdf -Destination $PublicationPdf -Force

        & pdftocairo -singlefile -png -r 300 $PublicationPdf (Join-Path $PngOut $Name)
        if ($LASTEXITCODE -ne 0) {
            throw "PNG preview rendering failed for $Name."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "Mean Teacher TikZ figures built under $PdfOut" -ForegroundColor Green

# ---------------------------------------------------------------------------------
# clDice operator panel. Same pattern, different source directory: the figure is
# operator geometry and dataflow only, so it has no dependency on probe outputs and
# rebuilds from source alone.
# ---------------------------------------------------------------------------------
$ClSource = Join-Path $Dissertation 'Figures\src\methods\cldice'
$ClBuild  = Join-Path $Dissertation 'build\figures\cldice'
$ClPdfOut = Join-Path $Dissertation 'Figures\pdf\methods\cldice'
$ClPngOut = Join-Path $Dissertation 'Figures\png\methods\cldice'

New-Item -ItemType Directory -Force -Path $ClBuild, $ClPdfOut, $ClPngOut | Out-Null

Push-Location $ClSource
try {
    foreach ($Name in @('soft_skeleton_operator')) {
        & pdflatex `
            -interaction=nonstopmode `
            -halt-on-error `
            "-output-directory=$ClBuild" `
            "$Name.tex"
        if ($LASTEXITCODE -ne 0) {
            throw "TikZ compilation failed for $Name."
        }

        $BuiltPdf = Join-Path $ClBuild "$Name.pdf"
        $PublicationPdf = Join-Path $ClPdfOut "$Name.pdf"
        Copy-Item -LiteralPath $BuiltPdf -Destination $PublicationPdf -Force

        & pdftocairo -singlefile -png -r 300 $PublicationPdf (Join-Path $ClPngOut $Name)
        if ($LASTEXITCODE -ne 0) {
            throw "PNG preview rendering failed for $Name."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "clDice TikZ figures built under $ClPdfOut" -ForegroundColor Green
