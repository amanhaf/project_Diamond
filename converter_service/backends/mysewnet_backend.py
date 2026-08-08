import subprocess
import time
from pathlib import Path

from backends.base import BaseConverter
from models import ConversionResult


class MySewnetConverter(BaseConverter):
    supported_exts = (".vip", ".art")
    backend_name = "mysewnet"

    def convert_to_pes(self, src: Path, dst_dir: Path) -> ConversionResult:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{src.stem}.pes"

        if dst.exists() and dst.stat().st_size > 0:
            return ConversionResult(
                ok=True,
                backend=self.backend_name,
                output_path=str(dst),
                error=None,
            )

        project_root = Path(__file__).resolve().parents[2]
        wrapper = project_root / "tools" / "mysewnet_wrapper.py"

        if not wrapper.exists():
            return ConversionResult(
                ok=False,
                backend=self.backend_name,
                output_path=None,
                error=f"Wrapper script not found: {wrapper}",
            )

        cmd = [
            "python3",
            str(wrapper),
            "--input",
            str(src.resolve()),
            "--output",
            str(dst.resolve()),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )

            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip() or f"exit code {result.returncode}"
                return ConversionResult(
                    ok=False,
                    backend=self.backend_name,
                    output_path=None,
                    error=f"Wrapper failed: {detail}",
                )

            # small wait in case mySewnet finishes flushing file slightly late
            deadline = time.time() + 15
            while time.time() < deadline:
                if dst.exists() and dst.stat().st_size > 0:
                    return ConversionResult(
                        ok=True,
                        backend=self.backend_name,
                        output_path=str(dst),
                        error=None,
                    )
                time.sleep(0.5)

            return ConversionResult(
                ok=False,
                backend=self.backend_name,
                output_path=None,
                error="mySewnet reported success but no PES file was created.",
            )

        except subprocess.TimeoutExpired:
            return ConversionResult(
                ok=False,
                backend=self.backend_name,
                output_path=None,
                error="mySewnet wrapper timed out.",
            )
        except Exception as e:
            return ConversionResult(
                ok=False,
                backend=self.backend_name,
                output_path=None,
                error=str(e),
            )