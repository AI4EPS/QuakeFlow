# %%
!pip install HinetPy
# !wget https://github.com/AI4EPS/software/releases/download/win32tools/win32tools.tar.gz
! [ -e win32tools.tar.gz ] || wget https://github.com/AI4EPS/software/releases/download/win32tools/win32tools.tar.gz
!tar -xvf win32tools.tar.gz
!cd win32tools && make


# %%
from HinetPy import Client, win32
import os

os.environ["PATH"] += os.pathsep + os.path.abspath("win32tools/catwin32.src") + os.pathsep + os.path.abspath("win32tools/win2sac.src")

# %%
waveform_path = "local/wavefroms/"

# %%
client = Client("", "")

data, ctable = client.get_continuous_waveform("0101", "201001010000", 20, outdir=f"{waveform_path}/cnt")


# %%
# data = "2010010100000101VM.cnt"
# ctable = "01_01_20100101.euc.ch"

win32.extract_sac(data, ctable, outdir="local/wavefroms")
win32.extract_sacpz(ctable)