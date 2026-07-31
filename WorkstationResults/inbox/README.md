# Workstation result drop

Place `cellect_workstation_results.zip` from the RTX 3090 workstation in this directory.

Do not extract or rename it. The archive contains the metrics, checkpoints, ONNX exports,
previews, environment report, and SHA-256 checksums needed for evaluation and Core ML conversion.

After copying the archive, validate and unpack it with:

```bash
python3 WorkstationResults/import_results.py
```

Core ML conversion is performed separately on the Mac; it does not retrain the networks.
