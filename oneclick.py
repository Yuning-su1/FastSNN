#!/usr/bin/env python
import sys, subprocess, os, importlib

def run(cmd):
    print('>>', ' '.join(cmd))
    return subprocess.call(cmd)

def main():
    # 1) Install package in editable mode (optional)
    if os.path.exists('setup.py'):
        run([sys.executable, '-m', 'pip', 'install', '-e', '.'])
    # 2) Forward smoke test
    run([sys.executable, '-m', 'fastsnn.cli.main', '--attn', 'hybrid_alt'])
    # 3) Mini train
    run([sys.executable, 'tests/test_train.py'])
    # 4) ANN->SNN conversion demo
    run([sys.executable, '-c', 'from fastsnn.convert.ann2snn import demo_convert_and_run; import torch; a,b=demo_convert_and_run(); print(a.shape,b.shape)'])

if __name__ == '__main__':
    sys.exit(main())
