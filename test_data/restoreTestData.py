import shutil
from pathlib import Path

pwd = Path(__file__).parent

copy_src = pwd / "src"
copy_dest = pwd / "current"

if copy_dest.is_dir():
    copy_dest.rmdir()

shutil.copytree(copy_src, copy_dest)
