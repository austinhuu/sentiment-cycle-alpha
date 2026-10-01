$ErrorActionPreference = "Stop"

$Python = "python"  # Uses the active Python environment
$ProjectRoot = $PSScriptRoot
$ManifestPath = Join-Path $ProjectRoot "outputs\v5\sensitivity\scenario_cache\manifest.json"

Push-Location $ProjectRoot
try {
    & $Python (Join-Path $ProjectRoot "run_v5_prepare_sensitivity.py")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    if (-not (Test-Path -LiteralPath $ManifestPath)) {
        throw "Sensitivity manifest was not created: $ManifestPath"
    }

    # Windows PowerShell 5 does not reliably assume UTF-8 for Get-Content.
    # Read explicitly as UTF-8 so Chinese project paths cannot corrupt JSON.
    $Utf8 = New-Object System.Text.UTF8Encoding($false)
    $ManifestText = [System.IO.File]::ReadAllText($ManifestPath, $Utf8)
    $Manifest = $ManifestText | ConvertFrom-Json

    foreach ($Scenario in $Manifest.scenarios) {
        $ScenarioRelative = ([string]$Scenario.scenario_path).Replace('/', '\')
        $ResultRelative = ([string]$Scenario.result_path).Replace('/', '\')
        $ScenarioPath = Join-Path $ProjectRoot $ScenarioRelative
        $ResultPath = Join-Path $ProjectRoot $ResultRelative

        Write-Host ("[V5] Sensitivity {0}/{1}: {2}" -f $Scenario.index, $Manifest.count, $Scenario.name)
        & $Python (Join-Path $ProjectRoot "run_v5_scenario_worker.py") `
            --scenario $ScenarioPath `
            --output $ResultPath
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }

    & $Python (Join-Path $ProjectRoot "run_v5_finalize_sensitivity.py")
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}