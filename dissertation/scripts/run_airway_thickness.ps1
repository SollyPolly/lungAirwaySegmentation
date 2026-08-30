# Measure what proportion of the GT airway tree is 1, 2, 3... voxels thick, avoiding the
# wall-shell artefact of a naive distance transform.
#
# Reports two measures side by side:
#   A. Operational thickness in index units, using clDice's OWN cross-erosion and cube
#      dilation. Class 0 of this histogram IS the clDice-degenerate set, so the degeneracy
#      census and the thickness histogram are one measurement, not two.
#   B. Anatomical thickness in mm, by Euclidean local thickness. Slower; -NoMillimetres
#      skips it.
#
# Each is reported by VOLUME share and by CENTRELINE-LENGTH share, which tell opposite
# stories -- the large airways dominate volume, the thin tree dominates length.
#
# Usage, from the repository root:
#     .\dissertation\scripts\run_airway_thickness.ps1
#     .\dissertation\scripts\run_airway_thickness.ps1 -NoMillimetres
#     .\dissertation\scripts\run_airway_thickness.ps1 -Cases ATM_046,ATM_125
#     .\dissertation\scripts\run_airway_thickness.ps1 -Cases (Get-Content val20.txt)

[CmdletBinding()]
param(
    [string[]]$Cases = @("ATM_034", "ATM_044", "ATM_046", "ATM_125"),
    [switch]$NoMillimetres,
    [switch]$RawGroundTruth,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Script = Join-Path $RepoRoot "dissertation\scripts\measure_airway_thickness.py"

if (-not (Test-Path $Python)) { throw "Interpreter not found: $Python" }
if (-not (Test-Path $Script)) { throw "Script not found: $Script" }

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "data\skeleton_scale_probe\results_thickness"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$scriptArgs = @("-u", $Script, "--output-dir", $OutputDir, "--cases") + $Cases
if ($NoMillimetres) { $scriptArgs += "--no-mm-thickness" }
if ($RawGroundTruth) { $scriptArgs += "--no-largest-component" }

$Log = Join-Path $OutputDir "console_thickness.txt"

Write-Output "Cases        : $($Cases -join ', ')"
Write-Output "Ground truth : $(if ($RawGroundTruth) { 'raw' } else { 'largest connected component' })"
Write-Output "mm sweep     : $(if ($NoMillimetres) { 'OFF' } else { 'ON (the slow part)' })"
Write-Output "Output       : $OutputDir"
Write-Output ""

Push-Location $RepoRoot
try {
    # Do NOT add 2>&1 here. Windows PowerShell 5.1 wraps each stderr line of a native
    # command in a NativeCommandError, which $ErrorActionPreference='Stop' then treats as
    # fatal -- the harmless nnU-Net "nnUNet_raw is not defined" notices would kill the run.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @scriptArgs | Tee-Object -FilePath $Log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
}
finally {
    Pop-Location
}

Write-Output ""
if ($code -ne 0) {
    Write-Output "Exited with code $code. See $Log"
    exit $code
}
Write-Output "Done. Wrote:"
Write-Output "  $OutputDir\airway_thickness.json"
Write-Output "  $OutputDir\airway_thickness_per_case.csv"
Write-Output "  $Log"
