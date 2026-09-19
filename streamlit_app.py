"""Entry point shim, so the app runs under either file name.

The dashboard lives in app.py, which is the name the rest of my projects use.
This file exists because Streamlit Community Cloud stores the main file path
when an app is first deployed, and this app was deployed before the rename.
Pointing that setting at a file that no longer existed produced a bare
"Error running app" with no traceback, which is a confusing failure for
something as ordinary as a renamed file.

Keeping this two line shim means the deployment works whether it is configured
with app.py or streamlit_app.py. It can be deleted once the main file path in
the Streamlit Cloud app settings says app.py.
"""

import runpy
from pathlib import Path

# run_name="__main__" so app.py behaves exactly as it would if Streamlit had
# loaded it directly, rather than as an imported module.
runpy.run_path(str(Path(__file__).parent / "app.py"), run_name="__main__")
