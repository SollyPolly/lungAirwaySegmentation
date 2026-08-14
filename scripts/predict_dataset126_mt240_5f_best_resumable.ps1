$ErrorActionPreference = 'Stop'

$Project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Project

$env:nnUNet_raw = Join-Path $Project 'data\nnunet\nnUNet_raw'
$env:nnUNet_preprocessed = Join-Path $Project 'data\nnunet\nnUNet_preprocessed'
$env:nnUNet_results = Join-Path $Project 'data\nnunet\nnUNet_results'

$Predict = Join-Path $Project '.venv\Scripts\nnUNetv2_predict.exe'
$Input = Join-Path $Project 'data\nnunet\predict_in\val_lungroi_m8_s120'
$BestOut = Join-Path $Project 'data\nnunet\predict_out\Dataset126_val_mt240_5f_lungcrop_best_teacher'
$Trainer = 'nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring'
$TrainerDirectory = "${Trainer}__nnUNetPlans__3d_fullres"
$ModelRoot = Join-Path $env:nnUNet_results "Dataset126_ATM22MT240LungCrop\$TrainerDirectory"

if (-not (Test-Path -LiteralPath $Predict -PathType Leaf)) {
    throw "nnU-Net predictor is missing: $Predict"
}

$Inputs = @(Get-ChildItem -LiteralPath $Input -Filter '*_0000.nii.gz' -File)
if ($Inputs.Count -ne 20) {
    throw "Expected 20 external-validation CT inputs; found $($Inputs.Count)."
}

foreach ($Fold in 0..4) {
    $Checkpoint = Join-Path $ModelRoot "fold_$Fold\checkpoint_best_teacher.pth"
    $Sidecar = "$Checkpoint.mt"
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Missing fold-$Fold best-teacher checkpoint: $Checkpoint"
    }
    if (-not (Test-Path -LiteralPath $Sidecar -PathType Leaf)) {
        throw "Missing fold-$Fold Mean-Teacher sidecar: $Sidecar"
    }
}

# --continue_prediction skips completed cases on rerun. One preprocessing and one
# export worker reduce host-memory contention while the CPU scorer runs separately.
& $Predict `
    -i $Input `
    -o $BestOut `
    -d 126 `
    -c 3d_fullres `
    -f 0 1 2 3 4 `
    -p nnUNetPlans `
    -tr $Trainer `
    -chk checkpoint_best_teacher.pth `
    --disable_tta `
    --continue_prediction `
    -npp 1 `
    -nps 1

if ($LASTEXITCODE -ne 0) {
    throw "Best-teacher prediction failed with exit code $LASTEXITCODE. Rerun this script to resume."
}

Write-Host "Completed best-teacher predictions: $BestOut" -ForegroundColor Green
