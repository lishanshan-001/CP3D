[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Python,

    [Parameter(Mandatory = $true)]
    [string]$PipelineRoot,

    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Src = Join-Path $Repo "src"
$ComparisonRoot = Join-Path $OutputRoot "point_3d_mot_comparison"
$AuditRoot = Join-Path $OutputRoot "threshold_sensitivity_audit"

Write-Host "[1/2] Running the canonical point-based 3D evaluation"
& $Python (Join-Path $Src "step5_point_3d_mot_comparison.py") `
    --pipeline-root $PipelineRoot `
    --data-root $DataRoot `
    --save-root $ComparisonRoot
if ($LASTEXITCODE -ne 0) {
    throw "3D evaluation failed with exit code $LASTEXITCODE"
}

Write-Host "[2/2] Auditing the 25 mm protocol and tolerance sensitivity"
& $Python (Join-Path $Src "step5_audit_3d_matching_threshold.py") `
    --pipeline-root $PipelineRoot `
    --data-root $DataRoot `
    --comparison-root $ComparisonRoot `
    --output-root $AuditRoot
if ($LASTEXITCODE -ne 0) {
    throw "3D protocol audit failed with exit code $LASTEXITCODE"
}

Write-Host "Evaluation outputs: $OutputRoot"
