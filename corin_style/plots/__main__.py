import importlib

for name in ("before_rl", "aster_rl", "early_seeds", "personas_rl", "spillover"):
    importlib.import_module(f"corin_style.plots.gen_fig_{name}").main()
