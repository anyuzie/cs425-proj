import os

for name, value in {'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}.items():
    os.environ.setdefault(name, value)
