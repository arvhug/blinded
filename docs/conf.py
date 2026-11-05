project = "blinded"
author = "Arvind Hughes"
copyright = "2026, Arvind Hughes"

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
]
myst_enable_extensions = ["dollarmath"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "_extra"]
# copied into the built site as they are: the run reports, at /reports/
html_extra_path = ["_extra"]

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

html_theme = "furo"
html_title = "blinded"
