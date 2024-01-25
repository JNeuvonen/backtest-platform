import os
import shutil
import subprocess


pyserver_path = os.path.join(os.getcwd(), "pyserver", "src") + os.sep

binaries_path = os.path.join(os.getcwd(), "src-tauri", "binaries")
pyserver_build_path = os.path.join(os.getcwd(), "pyserver", "build")

if os.path.exists(binaries_path):
    shutil.rmtree(binaries_path)
if os.path.exists(pyserver_build_path):
    shutil.rmtree(pyserver_build_path)

os.makedirs(binaries_path, exist_ok=True)

subprocess.run(
    ["pyoxidizer", "build", "--var", "PYSERVER_PATH", pyserver_path],
    cwd=os.path.join(os.getcwd(), "pyserver"),
)

shutil.copytree(
    pyserver_build_path,
    os.path.join(os.getcwd(), "src-tauri", "binaries"),
    dirs_exist_ok=True,
)

print("Done packaging python environment")
