import gzip
import base64
import os
from pathlib import Path
from typing import Dict


# this is base64 encoded source code
file_data: Dict = {file_data}

for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)


run('python setup.py develop --install-dir /kaggle/working')
# run('python -m smishing.make_folds')
run('python -m smishing.main train model_1 --n-epochs 1 --loss BCE --model_path "../backup/resnet_50/20190421_resnet50_ksk_0.598.pt"')
run('python -m smishing.inference model_1/test.h5 submission.csv --threshold 0.1')
