# MAPO HTML

GitHub Pages site for MAPO experiment reports and mechanistic analyses.

## Rebuild the attention analysis

From a MAPO checkout and this repository:

```bash
python scripts/export_report_data.py \
  --mapo-root /path/to/mapo \
  --output data/attention-normalization-summary.json
python scripts/build_attention_analysis.py
```

Published analyses are:

- `attention/normalizing-audio-attention.html`
- `attention/cot-quality-comparison.html`

The MAPO checkout is the canonical source for the report generators and manual
audit. This repository mirrors the generated static pages, compact data
snapshots, selected raster figures, and standalone builders for GitHub Pages.
