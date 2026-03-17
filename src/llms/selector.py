from __future__ import annotations

import shutil
import subprocess
import sys


class FzfNotFoundError(RuntimeError):
    pass


def ensure_fzf() -> str:
    fzf = shutil.which("fzf")
    if not fzf:
        raise FzfNotFoundError(
            "fzf is required but not found. Install it: https://github.com/junegunn/fzf#installation"
        )
    return fzf


def run_fzf(lines: str, preview_cmd: str | None = None) -> str | None:
    fzf = ensure_fzf()

    args = [
        fzf,
        "--reverse",
        "--prompt",
        "Model> ",
        "--header",
        "Enter: copy ID | Ctrl-C: exit",
        "--layout",
        "reverse",
        "--info",
        "inline",
    ]

    if preview_cmd:
        args.extend(
            [
                "--preview",
                preview_cmd,
                "--preview-window",
                "right:50%:wrap",
            ]
        )

    try:
        result = subprocess.run(
            args,
            input=lines.encode(),
            stdout=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as err:
        raise FzfNotFoundError("fzf not found") from err

    if result.returncode != 0:
        return None

    selected = result.stdout.decode().strip()
    if not selected:
        return None

    return selected


def run_fzf_multi(
    lines: str, *, limit: int = 2, preview_cmd: str | None = None
) -> list[str]:
    """Run fzf in multi-select mode and return selected items."""
    fzf = ensure_fzf()

    args = [
        fzf,
        "--reverse",
        "--multi",
        str(limit),
        "--prompt",
        "Compare> ",
        "--header",
        f"Tab: select (max {limit}) | Enter: compare | Ctrl-C: exit",
        "--layout",
        "reverse",
        "--info",
        "inline",
    ]

    if preview_cmd:
        args.extend(
            [
                "--preview",
                preview_cmd,
                "--preview-window",
                "right:50%:wrap",
            ]
        )

    try:
        result = subprocess.run(
            args,
            input=lines.encode(),
            stdout=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as err:
        raise FzfNotFoundError("fzf not found") from err

    if result.returncode != 0:
        return []

    selected = result.stdout.decode().strip()
    if not selected:
        return []

    return selected.splitlines()


def get_preview_command() -> str:
    python = sys.executable
    return f"{python} -m llms --preview {{}}"
