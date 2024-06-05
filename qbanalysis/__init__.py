# Import version from meta info (from project.toml)
# Note this is currently only provided for use with Scooby, https://pypi.org/project/scooby/
# See https://stackoverflow.com/a/76756889
try:
    import importlib.metadata
    __version__ = importlib.metadata.version(__package__)

# For Python < 3.8 may need this version
# Or just skip...
except ModuleNotFoundError:
    try:
        from importlib_metadata import metadata
        __version__ = metadata.version(__package__)

    except AttributeError:
        __version__ = 'Check project.toml'
    

# Config
from qbanalysis import config  # noqa: F401

