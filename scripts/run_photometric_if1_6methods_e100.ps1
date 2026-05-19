param(
    [string]$DataDir = "D:\event-benchmark\data\mvsec_official_probe\indoor_flying1_full",
    [string]$ImageH5 = "",
    [string]$RunRoot = "",
    [string[]]$Adapters = @("ergo", "est", "event_pretraining", "evrepsl", "get", "matrixlstm"),
    [int]$Epochs = 100,
    [int]$BatchSize = 8,
    [int]$EarlyStopValWindows = 100,
    [int]$EarlyStopPatience = 10,
    [double]$PhotometricWeight = 0.1,
    [double]$SmoothnessWeight = 0.05
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

if (-not $ImageH5) {
    $ImageH5 = Join-Path $DataDir "indoor_flying1_left_images_full.h5"
}
if (-not $RunRoot) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunRoot = Join-Path $DataDir "photometric_if1_6methods_e100_$stamp"
}

$events = Join-Path $DataDir "indoor_flying1_left_events_full.h5"
$flow = Join-Path $DataDir "indoor_flying1_gt_flow_full.npz"

if (-not (Test-Path -LiteralPath $events)) {
    throw "Missing events file: $events"
}
if (-not (Test-Path -LiteralPath $flow)) {
    throw "Missing flow file: $flow"
}
if (-not (Test-Path -LiteralPath $ImageH5)) {
    throw "Missing image file: $ImageH5"
}

$resultDir = Join-Path $RunRoot "results"
$curveDir = Join-Path $RunRoot "curves"
$logDir = Join-Path $RunRoot "logs"
New-Item -ItemType Directory -Force -Path $resultDir, $curveDir, $logDir | Out-Null

$summaryCsv = Join-Path $RunRoot "summary.csv"
"adapter,status,aee,outlier_percent,best_val_aee,best_epoch,epochs_completed,early_stopped,elapsed_seconds,result_json,curve_csv,log_file" |
    Set-Content -LiteralPath $summaryCsv -Encoding UTF8

$pair = "$events::$flow"
foreach ($adapter in $Adapters) {
    $start = Get-Date
    $name = "if1_${adapter}_e${Epochs}_bs${BatchSize}_photo${PhotometricWeight}_smooth${SmoothnessWeight}"
    $resultPath = Join-Path $resultDir "$name.json"
    $curvePath = Join-Path $curveDir "$name.csv"
    $logPath = Join-Path $logDir "$name.log"

    "===== running $adapter at $(Get-Date -Format o) =====" | Tee-Object -FilePath $logPath
    $args = @(
        "scripts/run_matrixlstm_paperlike_probe.py",
        "--adapter", $adapter,
        "--train-pair", $pair,
        "--eval-pair", $pair,
        "--train-image-h5", $ImageH5,
        "--eval-image-h5", $ImageH5,
        "--image-stride", "1",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--eval-batch-size", "1",
        "--device", "cuda",
        "--disable-cudnn",
        "--progress-every", "100",
        "--early-stop-patience", "$EarlyStopPatience",
        "--early-stop-min-delta", "0.001",
        "--early-stop-val-windows", "$EarlyStopValWindows",
        "--early-stop-val-strategy", "block-random",
        "--photometric-weight", "$PhotometricWeight",
        "--smoothness-weight", "$SmoothnessWeight",
        "--curve-log", $curvePath,
        "--output", $resultPath
    )

    $status = "ok"
    & python @args 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        $status = "failed"
    }

    $elapsed = [int]((Get-Date) - $start).TotalSeconds
    if ($status -eq "ok" -and (Test-Path -LiteralPath $resultPath)) {
        $json = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
        $line = @(
            $adapter,
            $status,
            $json.aee,
            $json.outlier_percent,
            $json.best_val_aee,
            $json.best_epoch,
            $json.epochs_completed,
            $json.early_stopped,
            $elapsed,
            $resultPath,
            $curvePath,
            $logPath
        ) -join ","
    } else {
        $line = @($adapter, $status, "", "", "", "", "", "", $elapsed, $resultPath, $curvePath, $logPath) -join ","
    }
    Add-Content -LiteralPath $summaryCsv -Value $line -Encoding UTF8
    "===== finished $adapter status=$status elapsed_seconds=$elapsed =====" | Tee-Object -FilePath $logPath -Append
}

"done_at=$(Get-Date -Format o)" | Set-Content -LiteralPath (Join-Path $RunRoot "DONE.txt") -Encoding UTF8
Write-Host "Run root: $RunRoot"
Write-Host "Summary: $summaryCsv"
