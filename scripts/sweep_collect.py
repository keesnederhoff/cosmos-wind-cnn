#!/usr/bin/env python
"""Aggregate the sf_bay_rtma hyperparameter sweep: best val_loss per variant.

Run anytime (during or after the sweep) from the repo root:
    python scripts/sweep_collect.py

Scans sweep_logs/*.log, extracts each variant's params + minimum "Val Loss",
and prints a table sorted best-first. Also folds in the baseline run
(gpu_pipeline_rtma_mm2_*.log) for comparison. Lower val_loss = better.
"""
import glob
import os
import re
from pathlib import Path


def parse(path):
    txt = Path(path).read_text(errors="ignore")
    m = re.search(r"run=(\S+)\s+base_channels=(\S+)\s+seq_len=(\S+)\s+"
                  r"dropout=(\S+)\s+lr=(\S+)", txt)
    if m:
        name, bc, sl, do, lr = m.groups()
    else:
        e = re.search(r"base_channels=(\d+), sequence_length=(\d+), "
                      r"dropout_rate=(\S+), learning_rate=(\S+)", txt)
        name = Path(path).stem
        bc, sl, do, lr = e.groups() if e else ("?", "?", "?", "?")
    vls = [float(x) for x in re.findall(r"Val Loss:\s*([0-9.]+)", txt)]
    best = min(vls) if vls else None
    done = ("PIPELINE COMPLETE" in txt) or ("exit: 0" in txt)
    return dict(name=name, bc=bc, sl=sl, do=do, lr=lr,
                best=best, epochs=len(vls), done=done)


def main():
    results = []
    for p in sorted(glob.glob(os.path.join("sweep_logs", "*.log"))):
        results.append(parse(p))
    for p in glob.glob("gpu_pipeline_rtma_mm2_*.log"):
        r = parse(p)
        r["name"] = (r["name"] or "baseline") + " (baseline)"
        results.append(r)

    results = [r for r in results if r["best"] is not None]
    results.sort(key=lambda r: r["best"])

    print(f"{'variant':24} {'bc':>3} {'sl':>3} {'drop':>5} {'lr':>8} "
          f"{'best_val':>9} {'ep':>4} {'done':>6}")
    print("-" * 74)
    for r in results:
        print(f"{r['name']:24} {r['bc']:>3} {r['sl']:>3} {r['do']:>5} {r['lr']:>8} "
              f"{r['best']:>9.4f} {r['epochs']:>4} {str(r['done']):>6}")
    if results:
        b = results[0]
        print(f"\nBest so far: {b['name']}  val_loss={b['best']:.4f}  "
              f"(bc={b['bc']} sl={b['sl']} dropout={b['do']} lr={b['lr']})")
        print(f"{sum(1 for r in results if r['done'])}/{len(results)} runs finished.")


if __name__ == "__main__":
    main()
