import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class StartupDefaultsTests(unittest.TestCase):
    def test_defaults_and_explicit_overrides(self):
        defaults = {'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
        code = 'import app, os, json; print(json.dumps({k: os.environ[k] for k in ' + repr(list(defaults)) + '}))'
        root = Path(__file__).resolve().parents[1]
        for overrides in ({}, {'HF_HUB_OFFLINE': '0', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '3'}):
            env = dict(os.environ)
            for key in defaults:
                env.pop(key, None)
            env.update(overrides)
            result = subprocess.run([sys.executable, '-c', code], cwd=root, env=env,
                                    text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout), defaults | overrides)


if __name__ == '__main__':
    unittest.main()
