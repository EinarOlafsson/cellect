# Cellect workstation model evaluation

- Date: 2026-07-30
- Training device: NVIDIA GeForce RTX 3090
- Returned run: `full`
- Data split reported by the archive: 3,253 training, 570 validation, and 1,564 test images

The original `cellect_workstation_results.zip` is retained unchanged in `inbox/`. The import
script extracted it to the ignored `imported/` directory only after every file listed in
`SHA256SUMS.txt` passed SHA-256 verification.

## Nine-model comparison

| Model | Test Dice | Test IoU | Count MAE | Parameters | ONNX size | RTX 3090 inference |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B3 U-Net++ | **0.91933** | **0.85475** | **189.1** | 13.62 M | 49.7 MB | 8.70 ms |
| SegFormer-B2 | 0.91748 | 0.85177 | 242.5 | 24.72 M | 94.8 MB | 8.09 ms |
| SegFormer-B5 | 0.91745 | 0.85172 | 248.6 | 81.97 M | 314.0 MB | 22.35 ms |
| EfficientNet-B0 U-Net | 0.91662 | 0.85017 | 203.8 | 6.25 M | 22.3 MB | 4.00 ms |
| ResNet101 U-Net++ | 0.91619 | 0.84988 | 194.7 | 67.98 M | 259.2 MB | 22.79 ms |
| ResNet18 U-Net | 0.91486 | 0.84736 | 207.4 | 14.33 M | 54.7 MB | 3.20 ms |
| ResNet50 DeepLabV3+ | 0.91370 | 0.84564 | 250.4 | 26.68 M | 101.7 MB | 5.61 ms |
| MobileNetV3-Large DeepLabV3+ | 0.90790 | 0.83578 | 260.0 | 4.71 M | 18.0 MB | 2.36 ms |
| MobileNetV3-Small U-Net | 0.88005 | 0.79230 | 256.9 | 3.59 M | 13.8 MB | 2.03 ms |

Every model's `history.csv` contains epoch, training loss, validation loss, validation IoU,
validation Dice, and learning rate. Checkpoints, TorchScript and ONNX exports, preview overlays,
and the exact environment report are retained under `imported/<model>/`.

## Core ML conversion and app bundle

All nine returned models are bundled so speed, memory use, and accuracy can be compared on actual
iPhones. Each conversion starts with float16, automatically falls back to float32 if its numerical
parity gate fails, and is rejected entirely if neither representation is safe. SegFormer requires
the Core ML neural-network backend because the ML Program conversion did not preserve its output.

| Core ML model | Precision/backend | Test Dice | Package | Median Intel Mac prediction | PyTorch/Core ML mean abs. error |
|---|---|---:|---:|---:|---:|
| MobileNetV3-Small U-Net | float32 / ML Program | 0.88005 | 13.8 MiB | 0.142 s | 0.000006 |
| MobileNetV3-Large DeepLabV3+ | float32 / ML Program | 0.90790 | 18.0 MiB | 0.144 s | 0.000107 |
| EfficientNet-B0 U-Net | float16 / ML Program | 0.91662 | 11.2 MiB | 0.317 s | 0.034365 |
| ResNet18 U-Net | float16 / ML Program | 0.91486 | 27.4 MiB | 0.425 s | 0.004218 |
| EfficientNet-B3 U-Net++ | float16 / ML Program | **0.91933** | 25.0 MiB | 0.696 s | 0.017693 |
| ResNet50 DeepLabV3+ | float32 / ML Program | 0.91370 | 101.7 MiB | 0.654 s | 0.000002 |
| ResNet101 U-Net++ | float32 / ML Program | 0.91619 | 259.3 MiB | 3.839 s | 0.000004 |
| SegFormer-B2 | float32 / neural network | 0.91748 | 94.3 MiB | 1.716 s | 0.025819 |
| SegFormer-B5 | float32 / neural network | 0.91745 | 312.8 MiB | 3.653 s | 0.006497 |

These are warm Core ML measurements on this Intel Mac; actual iPhone timing must be reported
separately. The signed development app is approximately 872 MiB with all nine compiled models.

The app exposes all nine networks plus Classical CV in its counting-model picker. It recommends a
default from physical memory but does not hide heavier models. A **Test image** button allows a
microscope file to be counted without using the camera. Runtime controls expose confidence and
boundary thresholds, touching-cell separation, image inversion, morphology, object-size filters,
and Core ML compute hardware. Every saved result includes a JSON file recording the exact model
and settings. Core ML emits foreground and boundary logits; Cellect performs morphology,
boundary-derived marker extraction, marker expansion, size filtering, relabeling, and per-object
measurement on device.

## Interpretation and limitation

EfficientNet-B3 U-Net++ is the correct current choice for semantic foreground segmentation: it has
the highest held-out Dice and IoU, the lowest returned count MAE, and remains small after float16
Core ML conversion. EfficientNet-B0 is the best efficiency compromise, losing only 0.27 Dice
percentage points while using an 11.2 MiB package.

The count MAE values remain high for every candidate. Therefore, these results support a claim
about foreground segmentation quality, but **do not yet support a claim of accurate cell instance
counting**. Before a paper or release makes that claim, run a second held-out evaluation with
instance-level AP/mAP, precision, recall, count bias, and MAE broken down by LIVECell cell type.
Thresholds and instance separation should be tuned only on validation data, then frozen before the
test set is evaluated.

LIVECell images and annotations are CC BY-NC 4.0. Review the dataset and derived-weight licensing
before commercial distribution.

## Reproduce the Mac-side steps

```bash
python3 WorkstationResults/import_results.py
/usr/bin/python3 -m venv /tmp/cellect-coreml-venv
/tmp/cellect-coreml-venv/bin/pip install -r WorkstationResults/coreml_requirements.txt
/tmp/cellect-coreml-venv/bin/python WorkstationResults/convert_coreml.py
xcodegen generate
xcodebuild -project Cellect.xcodeproj -scheme Cellect \
  -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' \
  CODE_SIGNING_ALLOWED=NO build
```

Machine-readable conversion metrics are in `coreml_evaluation.json`.
