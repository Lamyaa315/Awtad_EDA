"""Write app/envelopes.json: the left/right EMG envelope of every set (same filters as the notebook), at 25 Hz.

Run from the project folder:  python tools/export_envelopes.py
"""
import csv, json, re
from pathlib import Path
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
FS = 2148.1481
STEP = 86                                   # 2148 Hz / 86 ≈ 25 Hz; the envelope is low-passed at 6 Hz


def load_trial(path):
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        rows = list(csv.reader(f, skipinitialspace=True))
    left, right, markers = [], [], []
    for row in rows[8:]:
        if len(row) > 1 and row[1].strip(): left.append(float(row[1]))
        if len(row) > 3 and row[3].strip(): right.append(float(row[3]))
        if len(row) > 8 and row[5].strip() == 'General': markers.append(float(row[8]))
    return np.array(left), np.array(right), markers


def preprocess(x):
    x = x - x.mean()
    b, a = signal.butter(4, [20, 450], btype='bandpass', fs=FS)
    x = signal.filtfilt(b, a, x)
    b, a = signal.iirnotch(60, 30, fs=FS)
    return signal.filtfilt(b, a, x)


def envelope(x):
    b, a = signal.butter(4, 6, btype='low', fs=FS)
    return np.clip(signal.filtfilt(b, a, np.abs(x)), 0, None)


out = {}
for path in sorted((ROOT / 'data' / 'raw').glob('*/*.csv')):
    trial = int(re.search(r'_(\d)\.csv$', path.name).group(1))
    left, right, markers = load_trial(path)
    n = min(len(left), len(right))
    env = {side: envelope(preprocess(x[:n]))[::STEP] * 1000 for side, x in [('L', left), ('R', right)]}  # mV -> µV
    out[f'{path.parent.name}_{trial}'] = {'dt': STEP / FS, 'markers': [round(m, 3) for m in markers],
                                         'L': np.round(env['L'], 1).tolist(), 'R': np.round(env['R'], 1).tolist()}

(ROOT / 'app' / 'envelopes.json').write_text(json.dumps(out, separators=(',', ':')))
print(len(out), 'sets written')
