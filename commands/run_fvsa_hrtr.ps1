[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Python,

    [Parameter(Mandatory = $true)]
    [string]$Step3TrainRoot,

    [Parameter(Mandatory = $true)]
    [string]$Step3TestRoot,

    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [Parameter(Mandatory = $true)]
    [string]$RectificationRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [switch]$SkipEvaluation
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Src = Join-Path $Repo "src"
$Config = Join-Path $Repo "configs"
$FvsaOutput = Join-Path $OutputRoot "step4_association"
$HrtrOutput = Join-Path $OutputRoot "step5_reconstruction"

$fvsaArgs = @(
    (Join-Path $Src "step4_stereo_association_eval.py"),
    "--step3-train-root", $Step3TrainRoot,
    "--step3-test-root", $Step3TestRoot,
    "--data-root", $DataRoot,
    "--rect-root", $RectificationRoot,
    "--save-root", $FvsaOutput,
    "--methods", "Proposed",
    "--fixed-params-json", (Join-Path $Config "fvsa_fixed_parameters.json")
)
if ($SkipEvaluation) {
    $fvsaArgs += "--skip-eval"
}

Write-Host "[1/2] Running FVSA with frozen paper parameters"
& $Python @fvsaArgs
if ($LASTEXITCODE -ne 0) {
    throw "FVSA failed with exit code $LASTEXITCODE"
}

Write-Host "[2/2] Running HRTR with frozen paper parameters"
& $Python (Join-Path $Src "step5_reliability_guided_3d_reconstruction.py") `
    --step4-root $FvsaOutput `
    --save-root $HrtrOutput `
    --fixed-step5-metadata (Join-Path $Config "hrtr_fixed_parameters.json")
if ($LASTEXITCODE -ne 0) {
    throw "HRTR failed with exit code $LASTEXITCODE"
}

Write-Host "FVSA-HRTR outputs: $OutputRoot"
