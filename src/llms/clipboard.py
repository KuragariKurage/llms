from __future__ import annotations

import platform
import shutil
import subprocess


def copy_to_clipboard(text: str) -> bool:
    system = platform.system()
    if system == "Darwin":
        cmd = "pbcopy"
    elif system == "Linux":
        cmd = shutil.which("xclip") or shutil.which("xsel")
        if cmd and "xclip" in cmd:
            cmd = "xclip -selection clipboard"
        elif cmd and "xsel" in cmd:
            cmd = "xsel --clipboard --input"
        else:
            return False
    else:
        return False

    try:
        subprocess.run(
            cmd.split(),
            input=text.encode(),
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
