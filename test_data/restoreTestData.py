import shutil
from pathlib import Path

pwd = Path(__file__).parent

copy_src = pwd / "src"
copy_dest = pwd / "current"

if copy_dest.is_dir():
    shutil.rmtree(str(copy_dest))

shutil.copytree(copy_src, copy_dest)
