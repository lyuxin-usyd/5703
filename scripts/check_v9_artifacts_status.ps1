param(
    [string]$LatestRunRootFile = "D:\event-benchmark\data\v9_artifacts_latest_run_root.txt",
    [int]$Tail = 25
)

$ErrorActionPreference = "SilentlyContinue"

$statusFile = "D:\event-benchmark\data\v9_artifacts_latest_status.txt"
$doneMarker = "D:\event-benchmark\data\v9_artifacts_autorun.done"
$failedMarker = "D:\event-benchmark\data\v9_artifacts_autorun.failed"

if (-not (Test-Path -LiteralPath $LatestRunRootFile)) {
    Write-Host "No latest run root file found: $LatestRunRootFile"
    exit 1
}

$runRootRaw = (Get-Content -LiteralPath $LatestRunRootFile | Select-Object -First 1).Trim()
$runRoot = $runRootRaw -replace "/", "\"
$runName = Split-Path -Leaf $runRoot
$masterLog = "D:\event-benchmark\data\$runName.log"
$errLog = "D:\event-benchmark\data\$runName.err.log"

Write-Host "Run root: $runRoot"
if (Test-Path -LiteralPath $statusFile) {
    Write-Host ""
    Write-Host "[status]"
    Get-Content -LiteralPath $statusFile | Select-Object -Last 8
}

Write-Host ""
if (Test-Path -LiteralPath $failedMarker) {
    Write-Host "STATE: FAILED"
    Get-Content -LiteralPath $failedMarker
} elseif (Test-Path -LiteralPath $doneMarker) {
    Write-Host "STATE: DONE"
    Get-Content -LiteralPath $doneMarker
} else {
    Write-Host "STATE: RUNNING"
}

$expected = @(
    "ergo",
    "est",
    "get",
    "matrixlstm",
    "evrepsl",
    "event_pretraining",
    "voxel_grid",
    "event_frame",
    "binary_event_image",
    "timestamp_image",
    "time_surface"
)

$artifactRoot = Join-Path $runRoot "artifacts"
$completed = @()
if (Test-Path -LiteralPath $artifactRoot) {
    foreach ($method in $expected) {
        $metrics = Join-Path $artifactRoot "$method\metrics.json"
        $checkpoint = Join-Path $artifactRoot "$method\best_checkpoint.pt"
        $manifest = Join-Path $artifactRoot "$method\sample_manifest.csv"
        if ((Test-Path -LiteralPath $metrics) -and (Test-Path -LiteralPath $checkpoint) -and (Test-Path -LiteralPath $manifest)) {
            $completed += $method
        }
    }
}

Write-Host ""
Write-Host ("Completed artifact sets: {0}/{1}" -f $completed.Count, $expected.Count)
if ($completed.Count -gt 0) {
    Write-Host ("Completed: " + ($completed -join ", "))
}

if (Test-Path -LiteralPath $masterLog) {
    $runLines = Get-Content -LiteralPath $masterLog | Select-String "===== running "
    $finishLines = Get-Content -LiteralPath $masterLog | Select-String "===== finished "
    $current = ""
    if ($runLines.Count -gt 0) {
        $lastRun = $runLines[-1].Line
        if ($lastRun -match "===== running ([^ ]+)") {
            $current = $Matches[1]
        }
    }
    if ($current) {
        Write-Host ("Current method: " + $current)
    }
    if ($finishLines.Count -gt 0) {
        Write-Host ("Last finished: " + $finishLines[-1].Line)
    }

    Write-Host ""
    Write-Host "[latest training lines]"
    Get-Content -LiteralPath $masterLog -Tail 300 |
        Select-String "\[train\] epoch|\[early-stop\]|\[eval\]|===== running|===== finished" |
        Select-Object -Last $Tail
}

if (Test-Path -LiteralPath $errLog) {
    $errTail = Get-Content -LiteralPath $errLog -Tail 10
    if ($errTail) {
        Write-Host ""
        Write-Host "[stderr tail]"
        $errTail
    }
}
