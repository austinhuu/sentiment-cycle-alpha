$ErrorActionPreference = "Stop"
$Python = "python"  # Uses the active Python environment
$Scripts = @(
    "run_v5_walk_forward.py",
    "run_v5_health_monitor.py",
    "run_v5_bootstrap.py",
    "run_v5_monte_carlo.py"
)
foreach ($Script in $Scripts) {
    Write-Host "[V5] Running $Script"
    & $Python ".\$Script"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& powershell -ExecutionPolicy Bypass -File ".\run_v5_sensitivity.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& powershell -ExecutionPolicy Bypass -File ".\run_v5_pool_stability.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python ".\run_v5_acceptance.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python ".\run_v5_report.py"
exit $LASTEXITCODE