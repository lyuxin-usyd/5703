# Adapter Status

This file tracks how each of the seven methods should enter the unified MVSEC
benchmark.

## Legend

- `placeholder`: only benchmark-scaffold logic exists right now
- `next source`: upstream file(s) to inspect next
- `goal`: what the adapter should ultimately output into the unified flow head

## EST

- status: `first-pass`
- current adapter: `est`
- next source:
  - `D:\event-benchmark\EST-End-to-End-Learning-Representations-ICCV2019\utils`
- goal:
  - extract EST-style learnable/event-grid representation
  - emit `(C, H, W)` tensor for MVSEC windows
  - current scaffold uses NumPy trilinear quantization with EST-style channel
    layout so it can already enter the shared flow head

## ERGO

- status: `first-pass`
- current adapter: `ergo`
- next source:
  - `refs/event_representation_study/representations`
  - `refs/event_representation_study/ev-licious`
- goal:
  - fixed published ERGO representation
  - adapted to MVSEC optical flow through unified decoder
  - current scaffold reproduces the optimized representation recipe in NumPy
    with the published window/function/aggregation choices

## Event Pre-training

- status: `first-pass`
- current adapter: `event_pretraining`
- next source:
  - `refs/Event-Camera-Data-Pre-training/model`
  - `refs/Event-Camera-Data-Pre-training/trainer`
- goal:
  - preload backbone features or event-image frontend
  - connect to flow decoder for MVSEC
  - current scaffold now matches the upstream two-channel positive/negative
    count event frame used before the pretrained backbone

## GET

- status: `first-pass`
- current adapter: `get`
- next source:
  - `refs/GET-Group-Event-Transformer/event_based`
  - `refs/GET-Group-Event-Transformer/models`
- goal:
  - reuse tokenization or intermediate feature map
  - project tokens to dense flow features
  - current scaffold reproduces GET-style time-grouped patch token histograms
    and unwraps them back into a dense `(C, H, W)` map for MVSEC flow

## MatrixLSTM

- status: `first-pass`
- current adapter: `matrixlstm`
- next source:
  - `refs/matrixlstm/classification/layers/MatrixLSTM.py`
  - `refs/matrixlstm/classification/layers/extensions`
  - `refs/matrixlstm/opticalflow/src/EVFlowNet_MatrixLSTM.py`
- goal:
  - migrate representation logic into PyTorch benchmark path
  - keep official TensorFlow route as reference
  - current scaffold groups events per pixel and summarizes the local temporal
    sequence into four dense features, matching the default optical-flow output
    width without pulling in the legacy TensorFlow kernels

## EvRepSL

- status: `first-pass`
- current adapter: `evrepsl`
- next source:
  - `refs/EvRepSL/event_representations.py`
  - `refs/EvRepSL/models.py`
- goal:
  - load RepGen or its representation builder
  - emit EvRepSL tensor for MVSEC windows
  - current scaffold already supports EvRep in NumPy and can later upgrade to
    true EvRepSL when `RepGen.pth` is available

## OmniEvent

- status: `reported-only`
- current adapter: `omnievent`
- next source:
  - `refs/OmniEvent` once optical-flow-relevant code becomes usable
- goal:
  - keep reported results separate from the six-method runnable benchmark
  - only promote to runnable code if upstream optical-flow code becomes usable
