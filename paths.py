"""Central data/output path policy for the package.

Both layers resolve artifact locations through here, so the loop closes:
the extraction layer (extract/) WRITES its activations to the same place the
figure layer (figures/) READS them. Override with environment variables:

    CIK_DATA   root of the cached-artifact tree   (default: <repo>/data)
    CIK_OUT    root for regenerated figures        (default: <repo>/figures/output)

Usage from any script (any depth): add the repo root to sys.path, then
    from paths import data_dir, out_dir
    OUT = os.path.join(data_dir("geometry"), f"multiscr_{tag}.json")   # extract writes
    d   = data_dir("geometry", "cache")                               # figures read
"""
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))


def data_dir(*sub):
    """Location under the cached-artifact tree (default <repo>/data; override $CIK_DATA)."""
    base = os.environ.get("CIK_DATA", os.path.join(_ROOT, "data"))
    return os.path.join(base, *sub) if sub else base


def out_dir(*sub):
    """Location under the figure-output tree (default <repo>/figures/output; override $CIK_OUT). Created on demand."""
    base = os.environ.get("CIK_OUT", os.path.join(_ROOT, "figures", "output"))
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, *sub) if sub else base
