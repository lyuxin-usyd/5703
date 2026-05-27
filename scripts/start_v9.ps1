param(
    [string]$V9Root = "",
    [string]$DataRoot = "D:/event-benchmark/data/mvsec_full_processed",
    [string]$Method = "matrixlstm"
)

$ErrorActionPreference = "Stop"

if (-not $V9Root) {
    $V9Root = Split-Path -Parent $PSScriptRoot
}

$bash = "C:\Program Files\Git\bin\bash.exe"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = "D:/event-benchmark/data/v9_${Method}_$stamp"
$log = "D:\event-benchmark\data\v9_${Method}_$stamp.log"
$errLog = "D:\event-benchmark\data\v9_${Method}_$stamp.err.log"
$status = "D:\event-benchmark\data\v9_latest_status.txt"

function To-BashPath([string]$Path) {
    $converted = $Path -replace "\\", "/"
    return $converted -replace "^D:", "/d"
}

$v9RootBash = To-BashPath $V9Root

@(
    "$(Get-Date -Format o) starting V9 self-supervised run",
    "method=$Method",
    "run_root=$runRoot",
    "log=$log",
    "err_log=$errLog"
) | Set-Content -LiteralPath $status -Encoding UTF8

$cmd = @"
cd '$v9RootBash' && DATA_ROOT='$DataRoot' RUN_ROOT='$runRoot' OMP_NUM_THREADS=8 DEVICE=cuda EPOCHS=100 BATCH_SIZE=8 EVAL_BATCH_SIZE=1 VAL_WINDOWS=100 PATIENCE=10 PROGRESS_EVERY=50 MODEL_VARIANT=evflownet_multiscale TRAINING_OBJECTIVE=self_supervised SUPERVISED_WEIGHT=0.0 PHOTOMETRIC_WEIGHT=1.0 SMOOTHNESS_WEIGHT=0.5 PHOTOMETRIC_LOSS=evflownet PHOTOMETRIC_VALID_MASK=0 SMOOTHNESS_MODE=evflownet_8conn LR=3e-4 LR_SCHEDULE=evflownet LR_DECAY=0.9 WEIGHT_DECAY=1e-4 MODEL_BATCH_NORM=1 PAPER_CROP_SIZE=256 PAPER_TRAIN_RANDOM_CROP=1 PAPER_RANDOM_FLIP=1 PAPER_RANDOM_ROTATION_DEGREES=30 IMAGE_SCALE=raw255 TRAIN_IMAGE_STRIDE_MIN=1 TRAIN_IMAGE_STRIDE_MAX=5 RANDOM_TRAIN_STRIDE_PER_EPOCH=1 LAZY_RANDOM_TRAIN_STRIDE=1 METRIC_SCOPE=evflownet_official bash scripts/run_photometric_full_6methods_e100.sh '$Method'
"@

& $bash -lc $cmd >> $log 2>> $errLog
$exitCode = $LASTEXITCODE
"$(Get-Date -Format o) exited code=$exitCode" | Add-Content -LiteralPath $status -Encoding UTF8
exit $exitCode
