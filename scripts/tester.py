# %%

from audiobooks import Book
from audiobooks.utils.cli import confirm
from IPython.display import display
import audiobooks
import logging
import subprocess

# %% Set up testing
logging.basicConfig(level="INFO")
subprocess.run(["uv", "run", r"G:\Projects\audiobooks\test_data\restoreTestData.py"])

# %% Test
epub_path = r"G:\Projects\audiobooks\test_data\current\real_example.epub"

audiobooks.main(
    paths=[epub_path],
    voice="am_michael",
    speed=1.2,
    lexicon_g2g=[r"G:\Projects\audiobooks\lexicons\g2g\AoaB.yaml"],
    lexicon_g2p=[r"G:\Projects\audiobooks\lexicons\g2p\AoaB.yaml"],
    override_author="An Author",
    override_series="Example Epubs",
    output=None,
    init_only=False,
    clean=False,
)

logging.info("From epub finished")

# %% Resume Test

epub_path = r"G:\Projects\audiobooks\test_data\current\real_example"

audiobooks.main(
    paths=[epub_path],
    voice="am_michael",
    speed=1.2,
    lexicon_g2g=[r"G:\Projects\audiobooks\lexicons\g2g\AoaB.yaml"],
    lexicon_g2p=[r"G:\Projects\audiobooks\lexicons\g2p\AoaB.yaml"],
    override_author="An Author",
    override_series="Example Epubs",
    output=None,
    init_only=False,
    clean=False,
)

logging.info("From dir finished")
