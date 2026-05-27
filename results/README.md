# Optical Flow V9 Results

The current optical-flow results are split into two simple V9 folders:

```text
learning-base/
  summary.csv
  summary.md
  per_sequence_summary.csv
  validation_aee_curves.png
  best_validation_aee_curves.png
  curves/
  results/
  artifacts/

baseline/
  summary.csv
  summary.md
  per_sequence_summary.csv
  validation_aee_curves.png
  best_validation_aee_curves.png
  curves/
  results/
  artifacts/

figures/
sample_manifest.csv
```

`learning-base` is the main learned-representation table. `baseline` is the
traditional-representation table under the same V9 protocol.

The existing `summary.csv` files are the V9 paper-table results. The
`artifacts` folders add the later reproducibility package: model checkpoint,
fixed test-sample predicted flow, ground-truth flow, error map, image frames,
and per-sample manifest for each method.
