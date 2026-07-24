import os, config
want = ["ERA5", "CONUS404", "RTMA-SFbay", "CNN-allvars", "CNN-windonly", "CNN-extreme"]
ok = True
for m in want:
    d = config.MODELS[m]
    for k in ("u_file", "v_file"):
        p = d.get(k)
        e = bool(p) and os.path.exists(str(p))
        ok = ok and e
        print("%-13s %-7s %-8s %s" % (m, k, "EXISTS" if e else "MISSING", p))
for g, f, h in config.PWS_SOURCES:
    e = os.path.exists(str(f))
    ok = ok and e
    print("obs %-9s %-8s %s" % (g, "EXISTS" if e else "MISSING", f))
print("INCLUDE_USGS_MOORINGS =", config.INCLUDE_USGS_MOORINGS)
print("SMOKE", "OK" if ok else "FAILED")
