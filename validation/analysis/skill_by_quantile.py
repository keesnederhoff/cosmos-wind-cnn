"""Pool the per-station wind-speed quantile-bin stats (validate_met_models.py's
_validate_quantile_bins) into one tidy CSV per era, using the same
sample-size-weighted-within-category / WEIGHTS-across-category pooling as
combined_skill.py's combine_skill() -- no plot, per the 2026-08-28 request.

Reads validation_statistics.csv rows shaped "Wind Speed [m/s] (q75-90)" etc.
Run standalone: python analysis\\skill_by_quantile.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from analysis.combined_skill import WEIGHTS, ERA_DIRS, BASE, RANK_DIR, combine_skill

QUANTILE_BIN_LABELS = ['q00-25', 'q25-50', 'q50-75', 'q75-90', 'q90-100']
VAR_BASE = 'Wind Speed [m/s]'


def load_era(d):
    fp = BASE / d / 'validation_statistics.csv'
    if not fp.exists():
        print(f"  (missing: {fp})")
        return None
    raw = pd.read_csv(fp)
    return raw[(~raw['station'].astype(str).str.startswith('CWOP')) & raw['source'].isin(WEIGHTS)]


def main():
    out_rows = []
    for label, d in ERA_DIRS.items():
        raw = load_era(d)
        if raw is None:
            continue
        print(f"=== {label} ({d}) ===")
        for bin_label in QUANTILE_BIN_LABELS:
            var = f"{VAR_BASE} ({bin_label})"
            df = raw[raw['variable'] == var]
            df = df[(df['obs_std'] > 0.05) & (df['n'] >= 10)
                    & np.isfinite(df['rmse']) & np.isfinite(df['obs_std'])]
            if df.empty:
                print(f"  {bin_label}: no rows")
                continue
            for m, g in df.groupby('model'):
                cs = combine_skill(g)
                if cs is None:
                    continue
                print(f"  {bin_label:8s} {m:20s} skill={cs['skill']:.3f}  "
                      f"n_sta={sum(v['n_sta'] for v in cs['cats_detail'].values())}")
                out_rows.append({'era': label, 'quantile_bin': bin_label, 'model': m,
                                  'skill': cs['skill'], 'rmse': cs['rmse'], 'bias': cs['bias'],
                                  'corr': cs['corr'], 'std_ratio': cs['std_ratio'],
                                  'cats': cs['cats']})
    if not out_rows:
        print("No quantile-bin rows found in any era -- nothing written.")
        return
    out_df = pd.DataFrame(out_rows)
    out_dir = BASE / RANK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / 'skill_by_quantile.csv'
    out_df.to_csv(out_fp, index=False)
    print(f"\nWrote {len(out_df)} rows -> {out_fp}")


if __name__ == '__main__':
    main()
