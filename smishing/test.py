import shutil
from pathlib import Path


run_root = Path(args.run_root)
if run_root.exists():
    shutil.rmtree(run_root)
    run_root.mkdir(exist_ok=True, parents=True)
(run_root / 'params.json').write_text(
    json.dumps(vars(args), indent=4, sort_keys=True))