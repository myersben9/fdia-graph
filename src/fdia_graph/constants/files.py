"""The names and layouts of the files around a timeline: the unfinished file, the index of local
datasets, and an operator's load export. A timeline file's own names are in `schema.py`."""

from __future__ import annotations

PART = ".part"  # the ending of a file still being written: it takes its own name only once it is whole
# In the cache folder: the locally generated datasets, name -> path and meta.
LOCAL_INDEX = "local_datasets.json"
# How each operator's downloadable CSV export names its (timestamp, load) columns.
EXPORT_COLUMNS = {"caiso": ("interval_start_local", "load"), "nyiso": ("Time Stamp", "Load")}
