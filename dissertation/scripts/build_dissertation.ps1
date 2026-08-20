$ErrorActionPreference = 'Stop'

# This script lives in dissertation/scripts/, so its parent IS the dissertation root.
$Dissertation = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Build = Join-Path $Dissertation 'build'

New-Item -ItemType Directory -Force -Path $Build | Out-Null
# Push rather than Set so the caller's working directory survives the build and a
# following `.\dissertation\scripts\...` invocation still resolves.
Push-Location $Dissertation

function Invoke-PdfLaTeX {
    & pdflatex `
        -synctex=1 `
        -interaction=nonstopmode `
        -file-line-error `
        -halt-on-error `
        "-output-directory=$Build" `
        'main.tex'
    if ($LASTEXITCODE -ne 0) {
        throw "pdflatex failed with exit code $LASTEXITCODE."
    }
}

try {
    Invoke-PdfLaTeX

    # BIBER, not bibtex: main.tex declares backend=biber, under which biblatex emits
    # a .bcf control file and writes NO \citation/\bibdata/\bibstyle into main.aux.
    # Running bibtex against that aux finds nothing, exits 1, and truncates main.bbl
    # to zero bytes -- which silently empties the bibliography rather than failing
    # loudly at the pdflatex stage. Run from $Dissertation (the Push-Location above)
    # so \addbibresource{references.bib} resolves next to main.tex; --output-directory
    # keeps the .bcf read and the .bbl write inside build/.
    & biber "--output-directory=$Build" 'main'
    if ($LASTEXITCODE -ne 0) {
        throw "biber failed with exit code $LASTEXITCODE."
    }

    Invoke-PdfLaTeX
    Invoke-PdfLaTeX
}
finally {
    Pop-Location
}

$Pdf = Join-Path $Build 'main.pdf'
if (-not (Test-Path -LiteralPath $Pdf -PathType Leaf)) {
    throw "Expected dissertation PDF was not created: $Pdf"
}

Write-Host "Completed dissertation build: $Pdf" -ForegroundColor Green
