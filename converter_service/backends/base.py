from abc import ABC, abstractmethod
from pathlib import Path

from models import ConversionResult


class BaseConverter(ABC):
    supported_exts: tuple[str, ...] = ()
    backend_name: str = "base"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_exts

    @abstractmethod
    def convert_to_pes(self, src: Path, dst_dir: Path) -> ConversionResult:
        raise NotImplementedError