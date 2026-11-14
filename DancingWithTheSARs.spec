# PyInstaller recipe for the viewer.
#   Folder build (default, most reliable):   pyinstaller --noconfirm --clean DancingWithTheSARs.spec
#   Single-file build (slower start-up):     set ONEFILE=1 first (see build_exe.ps1 -OneFile)
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("pyproj") + collect_data_files("pyqtgraph")
hidden = collect_submodules("h5py") + ["process_pair", "fetch_weather"]
excludes = ["tkinter", "matplotlib", "IPython", "pytest", "scipy", "pandas"]

a = Analysis(["viewer.py"], pathex=["."], datas=datas, hiddenimports=hidden, excludes=excludes)
pyz = PYZ(a.pure)

if os.environ.get("ONEFILE"):
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="DancingWithTheSARs",
              console=False, upx=False)
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DancingWithTheSARs",
              console=False, upx=False)
    coll = COLLECT(exe, a.binaries, a.datas, name="DancingWithTheSARs", upx=False)
