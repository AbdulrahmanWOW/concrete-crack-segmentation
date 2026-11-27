# Frozen split manifests

After organizing the data, create the manifests once from the repository root:

```powershell
python -m src.data --images data/raw/images --masks data/raw/masks `
  --split-dir data/splits --seed 42 --test-frac 0.30
```

The files list image and mask paths separated by a tab. Commit both files so the
test set stays fixed; use only `dev.txt` for tuning and baseline studies. Never
regenerate or tune against `test.txt` after final evaluation.
