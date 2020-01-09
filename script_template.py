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
run('python -m smishing.main train --embedding mix --vector_size 300 --n_epochs 6')
run('python -m smishing.main inference --embedding mix --vector_size 300 --n_epochs 6')
# inference도 epoch, vector size 동일하게 할 것.

# glove fasttext mix