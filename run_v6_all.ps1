$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "python"  # Uses the active Python environment

Set-Location $ProjectRoot

$Scripts = @(
    "run_v6_optimize.py",
    "run_v6_order_plan.py",
    "run_v6_snapshot.py",
    "run_v6_acceptance.py",
    "run_v6_snapshot.py",
    "run_v6_report.py"
)

foreach ($Script in $Scripts) {
    Write-Host "[V6] Running $Script" -ForegroundColor Cyan
    & $Python -u (Join-Path $ProjectRoot $Script)
    if ($LASTEXITCODE -ne 0) {
        throw "V6 script failed: $Script"
    }
}

Write-Host "[V6] Complete. See outputs\v6\V6_FINAL_REPORT.md" -ForegroundColor Green