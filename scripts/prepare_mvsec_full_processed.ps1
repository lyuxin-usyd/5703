param(
  [string]$RawRoot = "D:\event-benchmark\data\mvsec_official_raw",
  [string]$OutputRoot = "D:\event-benchmark\data\mvsec_full_processed",
  [string]$Python = "py",
  [int]$MaxEvents = 6000000,
  [int]$MaxImages = 0,
  [int]$MaxFlowFrames = 0,
  [switch]$Force,
  [switch]$RemoveRawAfterSuccess
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$RawRoot = [System.IO.Path]::GetFullPath($RawRoot)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

$calibrations = @(
  @{
    Name = "outdoor_day_calib.zip"
    Url = "https://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day_calib.zip"
    Path = Join-Path $RawRoot "outdoor_day\outdoor_day_calib.zip"
    Bytes = 1290038
  },
  @{
    Name = "indoor_flying_calib.zip"
    Url = "https://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying_calib.zip"
    Path = Join-Path $RawRoot "indoor_flying\indoor_flying_calib.zip"
    Bytes = 1294736
  }
)

$sequences = @(
  @{
    Name = "outdoor_day1"
    Family = "outdoor_day"
    DataBag = Join-Path $RawRoot "outdoor_day\outdoor_day1_data.bag"
    GtBag = Join-Path $RawRoot "outdoor_day\outdoor_day1_gt.bag"
    Calib = Join-Path $RawRoot "outdoor_day\outdoor_day_calib.zip"
    DataBytes = 9253341329
    GtBytes = 10174697112
  },
  @{
    Name = "outdoor_day2"
    Family = "outdoor_day"
    DataBag = Join-Path $RawRoot "outdoor_day\outdoor_day2_data.bag"
    GtBag = Join-Path $RawRoot "outdoor_day\outdoor_day2_gt.bag"
    Calib = Join-Path $RawRoot "outdoor_day\outdoor_day_calib.zip"
    DataBytes = 28497983504
    GtBytes = 24170765728
  },
  @{
    Name = "indoor_flying1"
    Family = "indoor_flying"
    DataBag = Join-Path $RawRoot "indoor_flying\indoor_flying1_data.bag"
    GtBag = Join-Path $RawRoot "indoor_flying\indoor_flying1_gt.bag"
    Calib = Join-Path $RawRoot "indoor_flying\indoor_flying_calib.zip"
    DataBytes = 1278200655
    GtBytes = 2769160612
  },
  @{
    Name = "indoor_flying2"
    Family = "indoor_flying"
    DataBag = Join-Path $RawRoot "indoor_flying\indoor_flying2_data.bag"
    GtBag = Join-Path $RawRoot "indoor_flying\indoor_flying2_gt.bag"
    Calib = Join-Path $RawRoot "indoor_flying\indoor_flying_calib.zip"
    DataBytes = 1726175859
    GtBytes = 3351836116
  },
  @{
    Name = "indoor_flying3"
    Family = "indoor_flying"
    DataBag = Join-Path $RawRoot "indoor_flying\indoor_flying3_data.bag"
    GtBag = Join-Path $RawRoot "indoor_flying\indoor_flying3_gt.bag"
    Calib = Join-Path $RawRoot "indoor_flying\indoor_flying_calib.zip"
    DataBytes = 1854178527
    GtBytes = 3714570680
  }
)

function Invoke-Python {
  param([Parameter(Mandatory = $true)][string[]]$Arguments)
  if ($Python -eq "py") {
    & py -3.10 @Arguments
  } else {
    & $Python @Arguments
  }
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
  }
}

function Test-CompleteFile {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][int64]$Bytes
  )
  return (Test-Path $Path) -and ((Get-Item $Path).Length -ge $Bytes)
}

function Ensure-Calibration {
  param([Parameter(Mandatory = $true)][hashtable]$Item)
  if (Test-CompleteFile -Path $Item.Path -Bytes ([int64]$Item.Bytes)) {
    Write-Host "[calib ok] $($Item.Path)"
    return
  }
  New-Item -ItemType Directory -Force -Path (Split-Path $Item.Path) | Out-Null
  Write-Host "[calib download] $($Item.Url) -> $($Item.Path)"
  & curl.exe -L --retry 10 --retry-delay 5 --retry-all-errors --connect-timeout 30 `
    -o $Item.Path $Item.Url
  if ($LASTEXITCODE -ne 0) {
    throw "curl failed with exit code $LASTEXITCODE for $($Item.Url)"
  }
  if (-not (Test-CompleteFile -Path $Item.Path -Bytes ([int64]$Item.Bytes))) {
    $actual = if (Test-Path $Item.Path) { (Get-Item $Item.Path).Length } else { 0 }
    throw "Calibration size mismatch: $($Item.Path) expected at least $($Item.Bytes), got $actual"
  }
}

function Invoke-IfNeeded {
  param(
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][string[]]$Arguments
  )
  if ((Test-Path $Output) -and -not $Force) {
    Write-Host "[skip] $Output"
    return
  }
  if ((Test-Path $Output) -and $Force) {
    Remove-Item -LiteralPath $Output -Force
  }
  Invoke-Python -Arguments $Arguments
}

function Convert-Sequence {
  param([Parameter(Mandatory = $true)][hashtable]$Seq)

  $readyData = Test-CompleteFile -Path $Seq.DataBag -Bytes ([int64]$Seq.DataBytes)
  $readyGt = Test-CompleteFile -Path $Seq.GtBag -Bytes ([int64]$Seq.GtBytes)

  $outDir = Join-Path $OutputRoot $Seq.Family
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null

  $eventSuffix = if ($MaxEvents -le 0) {
    "full"
  } elseif (($MaxEvents % 1000000) -eq 0) {
    "{0}m" -f [int]($MaxEvents / 1000000)
  } else {
    [string]$MaxEvents
  }
  $eventsOut = Join-Path $outDir "$($Seq.Name)_left_events_$eventSuffix.h5"
  $imagesOut = Join-Path $outDir "$($Seq.Name)_left_images_full.h5"
  $flowOut = Join-Path $outDir "$($Seq.Name)_gt_flow_full.npz"

  if ($readyData) {
    $eventArgs = @(
      (Join-Path $ProjectRoot "scripts\convert_mvsec_bag_events.py"),
      $Seq.DataBag,
      "--output",
      $eventsOut,
      "--topic",
      "/davis/left/events"
    )
    if ($MaxEvents -gt 0) {
      $eventArgs += @("--max-events", [string]$MaxEvents)
    }
    Write-Host "[events] $($Seq.Name)"
    Invoke-IfNeeded -Output $eventsOut -Arguments $eventArgs

    $imageArgs = @(
      (Join-Path $ProjectRoot "scripts\convert_mvsec_bag_images.py"),
      $Seq.DataBag,
      "--output",
      $imagesOut,
      "--topic",
      "/davis/left/image_raw"
    )
    if ($MaxImages -gt 0) {
      $imageArgs += @("--max-images", [string]$MaxImages)
    }
    Write-Host "[images] $($Seq.Name)"
    Invoke-IfNeeded -Output $imagesOut -Arguments $imageArgs
  } else {
    Write-Host "[wait:data] $($Seq.Name): $($Seq.DataBag)"
  }

  if ($readyGt) {
    if (-not (Test-Path $Seq.Calib)) {
      throw "Missing calibration: $($Seq.Calib)"
    }
    $flowArgs = @(
      (Join-Path $ProjectRoot "scripts\generate_mvsec_flow_from_gt_bag.py"),
      "--gt-bag",
      $Seq.GtBag,
      "--calib",
      $Seq.Calib,
      "--output",
      $flowOut
    )
    if ($MaxFlowFrames -gt 0) {
      $flowArgs += @("--max-frames", [string]$MaxFlowFrames)
    }
    Write-Host "[flow] $($Seq.Name)"
    Invoke-IfNeeded -Output $flowOut -Arguments $flowArgs
  } else {
    Write-Host "[wait:gt] $($Seq.Name): $($Seq.GtBag)"
  }

  return ((Test-Path $eventsOut) -and (Test-Path $imagesOut) -and (Test-Path $flowOut))
}

Write-Host "ProjectRoot: $ProjectRoot"
Write-Host "RawRoot:     $RawRoot"
Write-Host "OutputRoot:  $OutputRoot"

foreach ($calib in $calibrations) {
  Ensure-Calibration -Item $calib
}

$converted = 0
foreach ($seq in $sequences) {
  $result = @(Convert-Sequence -Seq $seq)
  if ($result.Count -gt 0 -and $result[-1] -eq $true) {
    $converted += 1
  }
}

Write-Host "Converted sequences this pass: $converted / $($sequences.Count)"
Write-Host "Processed root: $OutputRoot"

if ($converted -eq $sequences.Count) {
  $done = Join-Path $OutputRoot "DONE.txt"
  Set-Content -LiteralPath $done -Encoding ascii -Value (Get-Date -Format s)
  Write-Host "[done] $done"

  if ($RemoveRawAfterSuccess) {
    Write-Host "[cleanup] removing raw bags under $RawRoot"
    foreach ($seq in $sequences) {
      Remove-Item -LiteralPath $seq.DataBag -Force
      Remove-Item -LiteralPath $seq.GtBag -Force
    }
    Write-Host "[cleanup] raw bag files removed; calibration zips kept"
  }
}
