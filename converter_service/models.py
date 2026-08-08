from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversionResult:
    ok: bool
    backend: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None