#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


MYSEWNET_APP_NAME = "mySewnet Embroidery"


def run_applescript(script: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def ensure_app_installed() -> None:
    app_path = Path(f"/Applications/{MYSEWNET_APP_NAME}.app")
    if not app_path.exists():
        raise RuntimeError(
            f"{MYSEWNET_APP_NAME}.app not found in /Applications. "
            "Install the Mac desktop version of mySewnet first."
        )


def ensure_input_exists(input_path: Path) -> None:
    if not input_path.exists():
        raise RuntimeError(f"Input file not found: {input_path}")


def escape_applescript_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def activate_app() -> None:
    script = f'''
    tell application "{MYSEWNET_APP_NAME}"
        activate
    end tell
    '''
    code, _, err = run_applescript(script)
    if code != 0:
        raise RuntimeError(f"Failed to activate mySewnet: {err}")
    time.sleep(2)


def open_file(input_path: Path) -> None:
    path_str = escape_applescript_string(str(input_path.resolve()))
    script = f'''
    tell application "{MYSEWNET_APP_NAME}"
        activate
        open POSIX file "{path_str}"
    end tell
    '''
    code, _, err = run_applescript(script)
    if code != 0:
        raise RuntimeError(f"Failed to open file in mySewnet: {err}")
    time.sleep(4)


def export_pes(output_path: Path) -> None:
    """
    This uses System Events UI scripting.
    Menu names may vary slightly by version/language.
    """
    out_parent = escape_applescript_string(str(output_path.parent.resolve()))
    out_name = escape_applescript_string(output_path.name)

    script = f'''
    tell application "{MYSEWNET_APP_NAME}"
        activate
    end tell

    tell application "System Events"
        tell process "{MYSEWNET_APP_NAME}"
            set frontmost to true
            delay 1

            -- Try File > Export first
            tell menu bar 1
                tell menu bar item "File"
                    click
                    delay 0.5
                    if exists menu item "Export..." of menu 1 then
                        click menu item "Export..." of menu 1
                    else if exists menu item "Save As..." of menu 1 then
                        click menu item "Save As..." of menu 1
                    else
                        error "Could not find Export... or Save As... in File menu."
                    end if
                end tell
            end tell

            delay 2

            -- Save dialog
            if exists sheet 1 of window 1 then
                tell sheet 1 of window 1
                    -- filename box
                    try
                        set value of text field 1 to "{out_name}"
                    end try
                    delay 0.5

                    -- expand dialog if needed
                    try
                        if exists button 2 then click button 2
                    end try
                    delay 0.5

                    -- go to folder using shortcut
                    keystroke "g" using {{command down, shift down}}
                    delay 1
                end tell
            else if exists window 1 then
                keystroke "g" using {{command down, shift down}}
                delay 1
            else
                error "Save/export dialog did not appear."
            end if
        end tell
    end tell
    '''
    code, _, err = run_applescript(script)
    if code != 0:
        raise RuntimeError(f"Failed during export dialog stage: {err}")

    # Go to folder dialog
    script2 = f'''
    tell application "System Events"
        tell process "{MYSEWNET_APP_NAME}"
            set frontmost to true
            delay 0.5

            keystroke "{out_parent}"
            delay 0.5
            key code 36
            delay 1

            -- Try to set file format to PES
            if exists sheet 1 of window 1 then
                tell sheet 1 of window 1
                    try
                        click pop up button 1
                        delay 0.5
                        click menu item "Brother/Baby Lock/Bernina PES (*.pes)" of menu 1 of pop up button 1
                    end try
                    delay 0.5
                    click button "Save"
                end tell
            else
                -- fallback: hit Enter and hope current format is PES if app remembers it
                key code 36
            end if
        end tell
    end tell
    '''
    code, _, err = run_applescript(script2)
    if code != 0:
        raise RuntimeError(f"Failed during export save stage: {err}")

    time.sleep(5)


def wait_for_output(output_path: Path, timeout_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if output_path.exists() and output_path.stat().st_size > 0:
            return
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for output PES: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        ensure_app_installed()
        ensure_input_exists(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        activate_app()
        open_file(input_path)
        export_pes(output_path)
        wait_for_output(output_path)

        print(str(output_path))
        return 0

    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())