from pathlib import Path

from backends.base import BaseConverter
from models import ConversionResult


class ArtlinkConverter(BaseConverter):
    supported_exts = ()
    backend_name = "artlink"

    def convert_to_pes(self, src: Path, dst_dir: Path) -> ConversionResult:
        return ConversionResult(
            ok=False,
            backend=self.backend_name,
            output_path=None,
            error="ARTlink backend is disabled on this Mac build.",
        )