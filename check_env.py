import sys

packages = ['pandas', 'numpy', 'scipy', 'sklearn', 'lightgbm', 'xgboost', 'rapidfuzz', 'tqdm']
for pkg in packages:
    try:
        m = __import__(pkg)
        print(f"{pkg}: {getattr(m, '__version__', 'installed')}")
    except ImportError as e:
        print(f"{pkg}: NOT installed ({e})")
