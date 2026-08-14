$ErrorActionPreference = 'Stop'

$Project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Project

$Python = Join-Path $Project '.venv\Scripts\python.exe'
$BestOut = Join-Path $Project 'data\nnunet\predict_out\Dataset126_val_mt240_5f_lungcrop_best_teacher'
$PartDir = Join-Path $BestOut 'score_parts_lcc6'
$FinalJson = Join-Path $BestOut 'nnunet126_val_mt240_5f_best_teacher_topology_lcc6.json'
$SplitConfig = Join-Path $Project 'configs\nnunet\atm22_split_l20_u240.yaml'
$Cases = @(
    '016', '027', '028', '033', '034', '043', '044', '046', '056', '068',
    '078', '081', '087', '116', '125', '126', '147', '150', '151', '152'
)

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable is missing: $Python"
}
if (-not (Test-Path -LiteralPath $BestOut -PathType Container)) {
    throw "Best-teacher prediction directory is missing: $BestOut"
}

$MissingPredictions = @(
    $Cases | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $BestOut "ATM_$_.nii.gz") -PathType Leaf)
    }
)
if ($MissingPredictions.Count) {
    throw "Missing best-teacher prediction(s): $($MissingPredictions -join ', ')"
}

New-Item -ItemType Directory -Force -Path $PartDir | Out-Null

foreach ($Case in $Cases) {
    $Part = Join-Path $PartDir "ATM_$Case.json"
    if (Test-Path -LiteralPath $Part -PathType Leaf) {
        Write-Host "Skipping completed case $Case" -ForegroundColor DarkGray
        continue
    }

    Write-Host "Scoring case $Case" -ForegroundColor Cyan
    & $Python -u -m scripts.evaluate_nnunet_predictions `
        --pred-dir $BestOut `
        --split-config $SplitConfig `
        --cases $Case `
        --report-split val `
        --branch `
        --lcc-connectivity 6 `
        --skip-radius-bins `
        --out $Part

    if ($LASTEXITCODE -ne 0) {
        throw "Scoring case $Case failed with exit code $LASTEXITCODE. Rerun this script to resume."
    }
}

$Parts = @($Cases | ForEach-Object { Join-Path $PartDir "ATM_$_.json" })
$MissingParts = @($Parts | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($MissingParts.Count) {
    throw "Missing score part(s): $($MissingParts -join ', ')"
}

Write-Host 'Merging all 20 per-case reports' -ForegroundColor Cyan
& $Python -u -m scripts.merge_nnunet_evaluation_parts `
    --parts @Parts `
    --out $FinalJson

if ($LASTEXITCODE -ne 0) {
    throw "Merging best-teacher score parts failed with exit code $LASTEXITCODE."
}

Write-Host "Completed best-teacher score: $FinalJson" -ForegroundColor Green
