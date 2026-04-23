#!/usr/bin/env python3
"""
export_model.py — pack the trained FLARE model into a portable .bin archive.

Binary layout:
  [magic     :  8 B  ] "FLAREMDL" (ASCII)
  [header_len:  4 B  ] uint32 big-endian — length of the JSON header below
  [header    :  N B  ] UTF-8 JSON — all training metadata
  [model_data: rest  ] raw joblib .pkl bytes (sklearn objects + scalers + column lists)

Usage:
  python3 export_model.py                  # auto-named flare_model_<ts>_f1_<score>.bin
  python3 export_model.py my_model.bin     # explicit output path
"""
import argparse
import json
import os
import struct
import sys
from datetime import datetime

MAGIC      = b'FLAREMDL'
VERSION    = 1
MODEL_PKL  = os.path.join('model', 'model.pkl')
MODEL_CFG  = os.path.join('model', 'config.json')


def main() -> None:
    ap = argparse.ArgumentParser(description='Export FLARE model to .bin archive')
    ap.add_argument('output', nargs='?', help='Output path (default: auto-named)')
    args = ap.parse_args()

    if not os.path.exists(MODEL_PKL):
        print(f'ERROR: {MODEL_PKL} not found — run classify.py first.', file=sys.stderr)
        sys.exit(1)

    # Load config.json (written by main.js after classify.py finishes)
    config: dict = {}
    if os.path.exists(MODEL_CFG):
        with open(MODEL_CFG) as f:
            config = json.load(f)

    # Pull rich metadata from inside the pkl bundle
    import joblib
    bundle = joblib.load(MODEL_PKL)

    header = {
        'magic':        'FLAREMDL',
        'version':      VERSION,
        'exported_at':  datetime.now().isoformat(),
        'trained_at':   bundle.get('trained_at') or config.get('trainedAt'),
        'sessions':     config.get('sessions'),
        'rounds':       config.get('rounds'),
        'window':       config.get('window'),
        'min_packets':  config.get('minPackets'),
        'min_size':     config.get('minSize', 0),
        'trees':        config.get('trees'),
        'best_variant': bundle.get('best_variant'),
        'cv_f1_mean':   round(float(bundle.get('cv_f1_mean', 0)), 4),
        'cv_f1_std':    round(float(bundle.get('cv_f1_std',  0)), 4),
        'n_windows':    bundle.get('n_windows'),
        'arch_names':   bundle.get('arch_names'),
        'n_flow_cols':  len(bundle.get('flow_cols', [])),
        'n_pkt_cols':   len(bundle.get('pkt_cols',  [])),
    }

    header_bytes = json.dumps(header, indent=2, ensure_ascii=False).encode('utf-8')
    header_len   = struct.pack('>I', len(header_bytes))

    with open(MODEL_PKL, 'rb') as f:
        pkl_bytes = f.read()

    if args.output:
        out_path = args.output
    else:
        ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
        f1_str = f"{header['cv_f1_mean']:.3f}".replace('.', 'p')
        out_path = f'flare_model_{ts}_f1{f1_str}.bin'

    with open(out_path, 'wb') as f:
        f.write(MAGIC)
        f.write(header_len)
        f.write(header_bytes)
        f.write(pkl_bytes)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f'Exported → {out_path}  ({size_mb:.1f} MB)')
    print(f'  Variant   : {header["best_variant"]}')
    print(f'  CV F1     : {header["cv_f1_mean"]:.3f} ± {header["cv_f1_std"]:.3f}')
    print(f'  Windows   : {header["n_windows"]}  ({header["n_flow_cols"]} flow + {header["n_pkt_cols"]} pkt features)')
    print(f'  Sessions  : {header["sessions"]}  Rounds: {header["rounds"]}  Window: {header["window"]}s')
    print(f'  Trained   : {header["trained_at"]}')
    print(f'  Exported  : {header["exported_at"]}')


if __name__ == '__main__':
    main()
