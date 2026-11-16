"""Version of the installed xcaper distribution."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xcaper")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "1.0.0"
