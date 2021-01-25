"""
`FlowCal`: Calibration and analysis of flow cytometry data.

"""

# Versions should comply with PEP440.  For a discussion on single-sourcing
# the version across setup.py and the project code, see
# https://packaging.python.org/en/latest/single_source_version.html
__version__ = '1.3.0'

from . import io
from . import excel_ui
from . import gate
from . import transform
from . import mef
from . import plot
from . import stats
