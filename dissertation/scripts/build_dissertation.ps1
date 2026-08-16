$ErrorActionPreference = 'Stop'

$Project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Dissertation = Join-Path $Project 'dissertation'
$Build = Join-Path $Dissertation 'build'

New-Item -ItemType Directory -Force -Path $Build | Out-Null
# Push rather than Set so the caller's working directory survives the build and a
# following `.\scripts\...` invocation still resolves.
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

$PreviousBibInputs = $env:BIBINPUTS
try {
    Invoke-PdfLaTeX

    # biblatex creates main-blx.bib beside the auxiliary files, so the build
    # directory must stay searchable. The SOURCE directory must come FIRST: a
    # stale references.bib left in build/ would otherwise shadow the real one and
    # silently freeze the bibliography (bibtex reports only a "didn't find a
    # database entry" warning, and the citation renders as an undefined "[?]").
    $env:BIBINPUTS = "$Dissertation;$Build"
    & bibtex (Join-Path $Build 'main')
    if ($LASTEXITCODE -ne 0) {
        throw "bibtex failed with exit code $LASTEXITCODE."
    }

    Invoke-PdfLaTeX
    Invoke-PdfLaTeX
}
finally {
    $env:BIBINPUTS = $PreviousBibInputs
    Pop-Location
}

$Pdf = Join-Path $Build 'main.pdf'
if (-not (Test-Path -LiteralPath $Pdf -PathType Leaf)) {
    throw "Expected dissertation PDF was not created: $Pdf"
}

Write-Host "Completed dissertation build: $Pdf" -ForegroundColor Green
