import platform

from pathlib import Path, WindowsPath

match platform.system():
    case "Windows":
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        def resolve_symlink(path: str|WindowsPath):
            if isinstance(path, WindowsPath):
                return shell.CreateShortCut(path).Targetpath if path.suffix == ".lnk" else path
            return shell.CreateShortCut(path).Targetpath if path.endswith(".lnk") else path
    case _:
        def resolve_symlink(path: Path):
            return path.resolve(strict=True)

from .ontology import *
from .brain_slice import *
from .sliced_brain import *
from .brain_data import *
from .animal_brain import *
from .animal_group import *
from .experiment import *

del ontology
del brain_slice
del sliced_brain
del brain_data
del animal_brain
del animal_group
del experiment