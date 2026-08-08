from pathlib import Path

from backends.mysewnet_backend import MySewnetConverter
from models import ConversionResult


class ConversionRegistry:
    def __init__(self):
        self.converters = [
            MySewnetConverter(),
        ]

    def convert_to_pes(self, src: Path, dst_dir: Path) -> ConversionResult:
        for converter in self.converters:
            if converter.can_handle(src):
                return converter.convert_to_pes(src, dst_dir)

        return ConversionResult(
            ok=False,
            backend=None,
            output_path=None,
            error=f"No converter registered for extension: {src.suffix.lower()}",
        )