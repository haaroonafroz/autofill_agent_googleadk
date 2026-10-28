import os
import tempfile
from pathlib import Path

# Use a throwaway database for every test run, before any autofill module is imported.
_tmp_dir = Path(tempfile.mkdtemp(prefix="autofill-tests-"))
os.environ["AUTOFILL_DB_PATH"] = str(_tmp_dir / "test.db")
