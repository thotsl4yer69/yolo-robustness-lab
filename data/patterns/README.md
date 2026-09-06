# Visual Pattern Pack

This directory is reserved for generated visual QA outputs from `tools/generate_pattern_pack.py`.

The generated pack is intentionally **not benchmark data** and must not be used as ground-truth samples. It exists to visually verify that each deterministic corruption behaves as intended before running the detector benchmark.

Generate locally with:

```bash
python tools/generate_pattern_pack.py --output artifacts/patterns
```

Expected outputs: one clean reference plus one image for every transform in `standard_suite()`.
