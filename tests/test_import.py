import os
import subprocess
import sys
import textwrap

OPTIONAL = ("rasterio", "h5py", "pyogrio", "csep", "openquake", "geopandas")

CHECK = """
import importlib, pkgutil, sys
import matplotlib as mpl

calls = []
mpl.use = lambda *a, **k: calls.append(a)
before = dict(dict.items(mpl.rcParams))

{imports}

assert not calls, f"matplotlib.use called: {{calls}}"
assert "matplotlib.pyplot" not in sys.modules, "pyplot imported"
after = dict(dict.items(mpl.rcParams))
diff = {{k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}}
assert not diff, f"rcParams changed: {{diff}}"
assert after["backend"] == "svg", "backend changed"
loaded = {{m.partition(".")[0] for m in sys.modules}} & set({optional})
assert not loaded, f"optional packages imported: {{loaded}}"
"""

TOP = "import pituplots"
ALL = """
import pituplots
for m in pkgutil.walk_packages(pituplots.__path__, "pituplots."):
    importlib.import_module(m.name)
"""


def run(imports):
    code = CHECK.format(imports=textwrap.dedent(imports), optional=OPTIONAL)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=os.environ | {"MPLBACKEND": "svg"})


def test_import_is_clean():
    r = run(TOP)
    assert r.returncode == 0, r.stderr


def test_all_modules_import_clean():
    r = run(ALL)
    assert r.returncode == 0, r.stderr


def test_check_detects_violations():
    for bad in ("import pituplots, json; mpl.use('Agg')",
                "mpl.rcParams['font.size'] = 3",
                "mpl.rcParams['backend'] = 'agg'",
                "import matplotlib.pyplot",
                "sys.modules['h5py'] = sys"):
        assert run(bad).returncode != 0, bad
