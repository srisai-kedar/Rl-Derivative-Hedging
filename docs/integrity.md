# Manifest and checkpoint integrity

Training and evaluation write `run_manifest.json`. It records the canonical
configuration hash, dependency/system information, Git state, covered
checkpoint filenames, and SHA-256 hashes for covered files.

Evaluation verifies integrity before deserializing the SAC archive or
VecNormalize state. The default legacy behavior warns when a manifest is
missing. For strict loading, use:

```bash
python -m src.evaluation.evaluate \
  --checkpoint checkpoints/<run>/final_model.zip \
  --vecnorm checkpoints/<run>/final_vecnorm.pkl \
  --require-manifest
```

For a trust anchor stored outside the checkpoint directory, provide both file
hashes with repeatable `--expected-sha256 FILE=HASH` arguments. Use
`--allow-unverified` only when intentionally loading a legacy or unverifiable
checkpoint. `--allow-manifest-mismatch` downgrades environment mismatch errors
to warnings.

## Deserialization warning

SAC/PyTorch archives and VecNormalize `.pkl` files are serialization formats,
not safe data formats. Loading them can execute arbitrary Python code. Hash
verification protects against accidental corruption and, with an out-of-band
expected hash, replacement; it does not make an untrusted source safe. Never
load files from an untrusted source merely because they contain a nearby
manifest.
