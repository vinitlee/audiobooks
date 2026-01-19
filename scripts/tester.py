# %%

import audiobooks
import audiobooks.project
from audiobooks.utils.cli import confirm
from IPython.display import display
import audiobooks
import logging
import subprocess
from pathlib import Path

# %% Set up testing
logging.basicConfig(level="INFO")
subprocess.run(
    ["uv", "run", r"/home/vinitlee/Projects/audiobooks/test_data/restoreTestData.py"]
)

# %%
epub_path = Path(
    r"/home/vinitlee/Projects/audiobooks/test_data/current/real_example.epub"
)
config = audiobooks.project.ProjectConfig(
    tts_voice="am_michael",
    tts_speed=1.2,
    lex_g2g_path=Path(r"/home/vinitlee/Projects/audiobooks/lexicons/g2g/AoaB.lex"),
    lex_g2p_path=Path(
        r"/home/vinitlee/Projects/audiobooks/test_data/current/scratch.lex"
    ),
    override_author="OVERRIDEAUTHOR",
    override_series="OVERRIDESERIES",
    output_path=Path(r"/home/vinitlee/Projects/audiobooks/test_data/current/output"),
    flag_init_only=False,
)

my_project = audiobooks.project.AudiobookProject.open(epub_path, config)
my_project.run_all()

# # %% Test
# epub_path = r"/home/vinitlee/Projects/audiobooks/test_data/current/real_example.epub"

# audiobooks.cli.main(
#     paths=[epub_path],
#     voice="am_michael",
#     speed=1.2,
#     lexicon_g2g=r"/home/vinitlee/Projects/audiobooks/lexicons/lex/AoaB.lex",
#     lexicon_g2p=r"/home/vinitlee/Projects/audiobooks/lexicons/lex/exclamations.lex",
#     override_author="An Author",
#     override_series="Example Epubs",
#     output=None,
#     init_only=False,
#     clean=False,
# )

# logging.info("From epub finished")

# # %% Resume Test

# epub_path = r"/home/vinitlee/Projects/audiobooks/test_data/current/real_example"

# audiobooks.cli.main(
#     paths=[epub_path],
#     voice="am_michael",
#     speed=1.2,
#     lexicon_g2g=[r"/home/vinitlee/Projects/audiobooks/lexicons/g2g/AoaB.yaml"],
#     lexicon_g2p=[r"/home/vinitlee/Projects/audiobooks/lexicons/g2p/AoaB.yaml"],
#     override_author="An Author",
#     override_series="Example Epubs",
#     output=None,
#     init_only=False,
#     clean=False,
# )

# logging.info("From dir finished")
