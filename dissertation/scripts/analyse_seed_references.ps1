$ErrorActionPreference = 'Stop'

# Recompute the calibre and branch-depth analyses with the 16-label supervised seed
# included as an absolute reference. Existing analysis directories are not overwritten.
$Repository = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = Join-Path $Repository '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python environment not found: $Python"
}

$PredictionRoot = Join-Path $Repository 'data\nnunet\predict_out'
$RequiredPredictionDirectories = @(
    'Dataset123_val_seed_lungcrop',
    'Dataset123_test_seed_f0',
    'Dataset126_val_mt240_control_final_teacher',
    'Dataset126_val_mt240_softcldice_final_teacher',
    'Dataset126_val_mt240_softcldice_5fold_final_teacher',
    'Dataset126_test_mt240_control_final_teacher',
    'Dataset126_test_mt240_softcldice_final_teacher',
    'Dataset126_test_mt240_softcldice_5fold_final_teacher',
    'Dataset111_val_l110_nods_nomirror_f0',
    'Dataset111_test_l110_nods_nomirror_f0'
)

foreach ($DirectoryName in $RequiredPredictionDirectories) {
    $Directory = Join-Path $PredictionRoot $DirectoryName
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        throw "Required prediction directory not found: $Directory"
    }
}

$TestCases = @(
    'ATM_002', 'ATM_012', 'ATM_014', 'ATM_021', 'ATM_024',
    'ATM_036', 'ATM_037', 'ATM_040', 'ATM_049', 'ATM_077',
    'ATM_085', 'ATM_089', 'ATM_091', 'ATM_098', 'ATM_100',
    'ATM_120', 'ATM_124', 'ATM_160', 'ATM_163', 'ATM_170'
)

$CalibreOutput = Join-Path $Repository `
    'data\skeleton_scale_probe\results_recall_by_calibre_soft5f_seed'
$DepthValidationOutput = Join-Path $Repository `
    'data\skeleton_scale_probe\results_recall_by_generation_soft5f_seed'
$DepthTestOutput = Join-Path $Repository `
    'data\skeleton_scale_probe\results_recall_by_generation_soft5f_seed_test'

$ExistingDepthValidation = Join-Path $Repository `
    'data\skeleton_scale_probe\results_recall_by_generation_soft5f\generation_depth_analysis.json'
$ExistingDepthTest = Join-Path $Repository `
    'data\skeleton_scale_probe\results_recall_by_generation_soft5f_test\generation_depth_analysis.json'

Push-Location $Repository
try {
    Write-Host '1/3  Measuring validation recovery by calibre...' -ForegroundColor Cyan
    & $Python 'dissertation\scripts\measure_recall_by_calibre.py' `
        --arm 'seed=data\nnunet\predict_out\Dataset123_val_seed_lungcrop' `
        --arm 'control=data\nnunet\predict_out\Dataset126_val_mt240_control_final_teacher' `
        --arm 'mt_soft=data\nnunet\predict_out\Dataset126_val_mt240_softcldice_final_teacher' `
        --arm 'mt_soft_5f=data\nnunet\predict_out\Dataset126_val_mt240_softcldice_5fold_final_teacher' `
        --arm 'ceiling110=data\nnunet\predict_out\Dataset111_val_l110_nods_nomirror_f0' `
        --baseline-arm control `
        --with-precision `
        --output-dir $CalibreOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Calibre analysis failed with exit code $LASTEXITCODE."
    }

    $CalibreJson = Join-Path $CalibreOutput 'recall_by_calibre.json'

    Write-Host '2/3  Measuring validation recovery by branch depth...' -ForegroundColor Cyan
    & $Python 'dissertation\scripts\measure_recall_by_generation.py' `
        --arm 'seed=data\nnunet\predict_out\Dataset123_val_seed_lungcrop' `
        --arm 'control=data\nnunet\predict_out\Dataset126_val_mt240_control_final_teacher' `
        --arm 'mt_soft=data\nnunet\predict_out\Dataset126_val_mt240_softcldice_final_teacher' `
        --arm 'mt_soft_5f=data\nnunet\predict_out\Dataset126_val_mt240_softcldice_5fold_final_teacher' `
        --arm 'ceiling110=data\nnunet\predict_out\Dataset111_val_l110_nods_nomirror_f0' `
        --baseline-arm control `
        --depth-groups-from $ExistingDepthValidation `
        --verify-against $CalibreJson `
        --output-dir $DepthValidationOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Validation depth analysis failed with exit code $LASTEXITCODE."
    }

    Write-Host '3/3  Measuring held-out-test recovery by branch depth...' -ForegroundColor Cyan
    & $Python 'dissertation\scripts\measure_recall_by_generation.py' `
        --arm 'seed=data\nnunet\predict_out\Dataset123_test_seed_f0' `
        --arm 'control=data\nnunet\predict_out\Dataset126_test_mt240_control_final_teacher' `
        --arm 'mt_soft=data\nnunet\predict_out\Dataset126_test_mt240_softcldice_final_teacher' `
        --arm 'mt_soft_5f=data\nnunet\predict_out\Dataset126_test_mt240_softcldice_5fold_final_teacher' `
        --arm 'ceiling110=data\nnunet\predict_out\Dataset111_test_l110_nods_nomirror_f0' `
        --baseline-arm control `
        --cases $TestCases `
        --depth-groups-from $ExistingDepthTest `
        --skip-thickness `
        --output-dir $DepthTestOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Held-out-test depth analysis failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

$ExpectedOutputs = @(
    (Join-Path $CalibreOutput 'recall_by_calibre.json'),
    (Join-Path $DepthValidationOutput 'generation_depth_analysis.json'),
    (Join-Path $DepthTestOutput 'generation_depth_analysis.json')
)

foreach ($Output in $ExpectedOutputs) {
    if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) {
        throw "Expected analysis output was not created: $Output"
    }
    Write-Host "Created: $Output" -ForegroundColor Green
}
