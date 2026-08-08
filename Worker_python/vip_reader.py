"""
VIP (Pfaff / Husqvarna-Viking) reader for the Diamond pipeline.

pyembroidery ships HUS and VP3 readers but NOT VIP. VIP is HUS's sibling
format: its three data sections use the identical Greenleaf ArchiveLib
compression that pyembroidery's EmbCompress already decodes. The only real
differences from HUS are:
  1. a slightly different header, and
  2. VIP stores true 24-bit RGB colors (double-XOR encoded) instead of
     HUS-style palette indices.

This module reuses pyembroidery's own EmbCompress / EmbPattern / EmbThread /
ReadHelper, so it produces a normal EmbPattern that the rest of the Diamond
worker (render, pHash, metadata) consumes unchanged.

Spec: Jason Weiler, "HUS and VIP Embroidery File Formats" (2006),
cross-checked against pyembroidery's HusReader.

Usage:
    from vip_reader import read_vip
    pattern = read_vip("design.vip")
"""
from __future__ import annotations

from pyembroidery.EmbCompress import expand
from pyembroidery.EmbPattern import EmbPattern
from pyembroidery.EmbThread import EmbThread
from pyembroidery.ReadHelper import (
    read_int_16le,
    read_int_32le,
    read_string_8,
    signed8,
    signed16,
)

# Fixed 400-byte table used by VIP's color decoder (supports up to 100 colors,
# far more than any real design). Transcribed from the published VIP spec.
_VIP_DECODING_TABLE_HEX = """
2E 82 E4 6F 38 A9 DC C6 7B B6 28 AC FD AA 8A 4E
76 2E F0 E4 25 1B 8A 68 4E 92 B9 B4 95 F0 3E EF
F7 40 24 18 39 31 BB E1 53 A8 1F B1 3A 07 FB CB
E6 00 81 50 0E 40 E1 2C 73 50 0D 91 D6 0A 5D D6
8B B8 62 AE 47 00 53 5A B7 80 AA 28 F7 5D 70 5E
2C 0B 98 E3 A0 98 60 47 89 9B 82 FB 40 C9 B4 00
0E 68 6A 1E 09 85 C0 53 81 D1 98 89 AF E8 85 4F
E3 69 89 03 A1 2E 8F CF ED 91 9F 58 1E D6 84 3C
09 27 BD F4 C3 90 C0 51 1B 2B 63 BC B9 3D 40 4D
62 6F E0 8C F5 5D 08 FD 3D 50 36 D7 C9 C9 43 E4
2D CB 95 B6 F4 0D EA C2 FD 66 3F 5E BD 69 06 2A
03 19 47 2B DF 38 EA 4F 80 49 95 B2 D6 F9 9A 75
F4 D8 9B 1D B0 A4 69 DB A9 21 79 6F D8 DE 33 FE
9F 04 E5 9A 6B 9B 73 83 62 7C B9 66 76 F2 5B C9
5E FC 74 AA 6C F1 CD 93 CE E9 80 53 03 3B 97 4B
39 76 C2 C1 56 CB 70 FD 3B 3E 52 57 81 5D 56 8D
51 90 D4 76 D7 D5 16 02 6D F2 4D E1 0E 96 4F A1
3A A0 60 59 64 04 1A E4 67 B6 ED 3F 74 20 55 1F
FB 23 92 91 53 C8 65 AB 9D 51 D6 73 DE 01 B1 80
B7 C0 D6 80 1C 2E 3C 83 63 EE BC 33 25 E2 0E 7A
67 DE 3F 71 14 49 9C 92 93 0D 26 9A 0E DA ED 6F
A4 89 0C 1B F0 A1 DF E1 9E 3C 04 78 E4 AB 6D FF
9C AF CA C7 88 17 9C E5 B7 33 6D DC ED 8F 6C 18
1D 71 06 B1 C5 E2 CF 13 77 81 C5 B7 0A 14 0A 6B
40 26 A0 88 D1 62 6A B3 50 12 B9 9B B5 83 9B 37
"""
VIP_DECODING_TABLE = bytes.fromhex(_VIP_DECODING_TABLE_HEX.replace("\n", " "))

# VIP command bytes (identical to HUS).
_STITCH = 0x80
_JUMP = 0x81
_COLOR_CHANGE = 0x84
_TRIM = 0x88
_END = 0x90


def decode_colors(encoded: bytes, number_of_colors: int) -> bytearray:
    """Decode VIP's double-XOR'd color block into RGB-0 quadruplets."""
    n = number_of_colors * 4
    out = bytearray(n)
    prev = 0
    for i in range(n):
        tmp = encoded[i] ^ VIP_DECODING_TABLE[i]
        out[i] = tmp ^ prev
        prev = encoded[i]
    return out


def read(f, out: EmbPattern, settings=None):
    """pyembroidery-style reader: parse stream `f` into EmbPattern `out`."""
    _magic = read_int_32le(f)                 # 0x0190FC5D for VIP
    number_of_stitches = read_int_32le(f)
    number_of_colors = read_int_32le(f)

    # Hoop extents (read for completeness; not needed downstream).
    _pos_x = signed16(read_int_16le(f))
    _pos_y = signed16(read_int_16le(f))
    _neg_x = signed16(read_int_16le(f))
    _neg_y = signed16(read_int_16le(f))

    attribute_offset = read_int_32le(f)       # Section 1: stitch attributes
    x_offset = read_int_32le(f)               # Section 2: x coordinates
    y_offset = read_int_32le(f)               # Section 3: y coordinates

    _string_id = read_string_8(f, 8)
    _unknown_a = read_int_16le(f)

    # Color block: 4 unused bytes at 0x2A, then 4*colors encoded bytes at 0x2E.
    _unknown_b = read_int_32le(f)
    encoded_colors = bytearray(f.read(number_of_colors * 4))
    if number_of_colors > 0 and len(encoded_colors) >= number_of_colors * 4:
        decoded = decode_colors(encoded_colors, number_of_colors)
        for i in range(number_of_colors):
            thread = EmbThread()
            thread.set_color(decoded[i * 4], decoded[i * 4 + 1], decoded[i * 4 + 2])
            out.add_thread(thread)

    # Read the three compressed sections by their absolute offsets.
    f.seek(attribute_offset, 0)
    command_compressed = bytearray(f.read(x_offset - attribute_offset))
    f.seek(x_offset, 0)
    x_compressed = bytearray(f.read(y_offset - x_offset))
    f.seek(y_offset, 0)
    y_compressed = bytearray(f.read())

    command_decompressed = expand(command_compressed, number_of_stitches)
    x_decompressed = expand(x_compressed, number_of_stitches)
    y_decompressed = expand(y_compressed, number_of_stitches)

    stitch_count = min(
        len(command_decompressed), len(x_decompressed), len(y_decompressed)
    )

    for i in range(stitch_count):
        cmd = command_decompressed[i]
        x = signed8(x_decompressed[i])
        y = -signed8(y_decompressed[i])  # match HusReader's y convention
        if cmd == _STITCH:
            out.stitch(x, y)
        elif cmd == _JUMP:
            out.move(x, y)
        elif cmd == _COLOR_CHANGE:
            out.color_change(x, y)
        elif cmd == _TRIM:
            if x != 0 or y != 0:
                out.move(x, y)
            out.trim()
        elif cmd == _END:
            break
        else:  # unmapped/garbage -> stop, like HusReader
            break
    out.end()


def read_vip(path_or_file) -> EmbPattern:
    """Read a VIP file (path or open binary stream) into a new EmbPattern."""
    pattern = EmbPattern()
    if hasattr(path_or_file, "read"):
        read(path_or_file, pattern)
    else:
        with open(path_or_file, "rb") as f:
            read(f, pattern)
    return pattern
