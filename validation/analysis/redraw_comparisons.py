"""Redraw the per-group comparison figures (Taylor + multi-model bars) from the
existing era CSVs, so every model gets its fixed MODEL_COLORS color. No model
data is re-read — this only re-renders plots from validation_statistics.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root on path
import config
import pandas as pd
import validate_met_models as V

BASE = config.OUTPUT_ROOT
ERA_DIRS = [
    'era1_1990-2010',
    'era2_2011-2021',
    'era3_2022-present',
]

for d in ERA_DIRS:
    csv = BASE / d / 'validation_statistics.csv'
    if not csv.exists():
        print(f"  {d}: no CSV, skip"); continue
    df = pd.read_csv(csv)
    # Match how main() feeds the plotters: per-station rows only (drop the
    # *_MEAN / ALL aggregate rows the exporter appends to the CSV).
    df = df[~df['station'].astype(str).str.contains('MEAN')]
    df = df[df['source'].isin(['IEM', 'NDBC', 'USGS', 'CWOP'])]
    records = df.to_dict('records')

    for g in V.GROUPS_FOR_SUMMARY_PLOTS:
        recs_g = [r for r in records if r.get('source') == g]
        if not recs_g:
            continue
        gdir = BASE / d / g
        gdir.mkdir(parents=True, exist_ok=True)
        try:
            V.plot_taylor_diagram(recs_g, gdir)
            V.plot_multi_model_comparison(recs_g, gdir)
        except Exception as exc:
            print(f"    {d}/{g}: summary plots failed: {exc}")
    cwop = [r for r in records if r.get('source') == 'CWOP']
    if cwop:
        (BASE / d / 'CWOP').mkdir(parents=True, exist_ok=True)
        try:
            V.plot_cwop_summary(cwop, BASE / d / 'CWOP')
        except Exception as exc:
            print(f"    {d}/CWOP: summary failed: {exc}")
    print(f"  redrew comparisons for {d}")

print("DONE.")
