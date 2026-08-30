# Run the soft-skeleton scale probe at 3x instead of 2x.
#
# Memory grows as scale**3: a patch tensor is 27x the native voxel count (248 MB in
# float32) against 8x at 2x. The forward-only measurements fit on an 8 GB card; the
# GRADIENT probe does not -- the checkpointed 2x backward already peaked at 6.53 GiB,
# so 3x needs roughly 22 GiB. It is therefore skipped by default here. Set
# -WithGradientProbe if you are on a >=40 GB GPU and want the decisive number.
#
# Output goes to results_3x/ so the reported 2x results are not overwritten.
#
# Usage, from the repository root:
#     .\dissertation\scripts\run_skeleton_scale_probe_3x.ps1
#     .\dissertation\scripts\run_skeleton_scale_probe_3x.ps1 -PatchesPerCase 3
#     .\dissertation\scripts\run_skeleton_scale_probe_3x.ps1 -WithGradientProbe

[CmdletBinding()]
param(
    [int]$Scale = 3,
    [int]$PatchesPerCase = 6,
    [string[]]$Cases = @(),
    [switch]$WithGradientProbe,
    [switch]$NoSyntheticCheck,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Script = Join-Path $RepoRoot "dissertation\scripts\measure_soft_skeleton_scale.py"

if (-not (Test-Path $Python)) { throw "Interpreter not found: $Python" }
if (-not (Test-Path $Script)) { throw "Probe script not found: $Script" }

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "data\skeleton_scale_probe\results_$($Scale)x"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$probeArgs = @(
    "-u", $Script,
    "--upsample-scale", $Scale,
    "--patches-per-case", $PatchesPerCase,
    "--output-dir", $OutputDir
)
if ($Cases.Count -gt 0) { $probeArgs += @("--cases") + $Cases }
if (-not $WithGradientProbe) { $probeArgs += "--no-gradient-probe" }
if ($NoSyntheticCheck) { $probeArgs += "--no-synthetic-check" }

$Log = Join-Path $OutputDir "console_run_$($Scale)x.txt"

Write-Output "Scale        : 1x vs $($Scale)x  ($($Scale * 10) skeleton iterations)"
Write-Output "Patches/case : $PatchesPerCase"
Write-Output "Gradient probe: $(if ($WithGradientProbe) { 'ON (needs >=40 GB GPU)' } else { 'OFF (skipped: needs ~22 GiB at 3x)' })"
Write-Output "Output       : $OutputDir"
Write-Output ""

Push-Location $RepoRoot
try {
    # Do NOT add 2>&1 here. Windows PowerShell 5.1 wraps each stderr line of a native
    # command in a NativeCommandError, which $ErrorActionPreference='Stop' then treats
    # as fatal -- so the harmless nnU-Net "nnUNet_raw is not defined" notices would kill
    # the run before it started. stdout is teed to the log; stderr goes to the console.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @probeArgs | Tee-Object -FilePath $Log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
}
finally {
    Pop-Location
}

Write-Output ""
if ($code -ne 0) {
    Write-Output "Probe exited with code $code. Results for any COMPLETED cases are still in:"
    Write-Output "  $OutputDir\skeleton_scale_probe.json"
    Write-Output "The probe writes after every case, so a late failure keeps earlier work."
    exit $code
}
Write-Output "Done. Wrote:"
Write-Output "  $OutputDir\skeleton_scale_probe.json"
Write-Output "  $OutputDir\skeleton_scale_probe_per_patch.csv"
Write-Output "  $Log"
Write-Output ""
Write-Output "Note: table columns are still labelled 'skel 2x' / 'dLoss 2x' -- they hold the"
Write-Output "$($Scale)x values. The JSON records the true pair under its 'scales' key."
