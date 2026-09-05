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

The published analysis is `attention/normalizing-audio-attention.html`. The
compact data snapshot and selected raster figures are committed so the page can
be served as a static GitHub Pages site.
