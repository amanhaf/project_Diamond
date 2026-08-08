# Save as: /Users/amans/Desktop/Mark-VLM/diagnose_render_split.py
import pyembroidery
from pathlib import Path
from collections import Counter

ROOT = Path("/Users/amans/Desktop/Mark-VLM/Metadata_Capture")

# Map pyembroidery's integer command codes to names
CMD_NAMES = {
    pyembroidery.STITCH: "STITCH",
    pyembroidery.JUMP: "JUMP",
    pyembroidery.TRIM: "TRIM",
    pyembroidery.COLOR_CHANGE: "COLOR_CHANGE",
    pyembroidery.STOP: "STOP",
    pyembroidery.END: "END",
    pyembroidery.SEQUIN_MODE: "SEQUIN_MODE",
    pyembroidery.SEQUIN_EJECT: "SEQUIN_EJECT",
}

def analyze(path):
    p = pyembroidery.read(str(path))
    if not p:
        return None
    stitches = p.stitches
    cmd_counts = Counter()
    long_stitches = 0  # stitches > 10mm (1000 in pyembroidery's 0.1mm units)
    very_long = 0      # > 20mm
    for i, (x, y, cmd) in enumerate(stitches):
        cmd_counts[CMD_NAMES.get(cmd, f"UNKNOWN({cmd})")] += 1
        if cmd == pyembroidery.STITCH and i > 0:
            px, py, _ = stitches[i-1]
            dist = ((x-px)**2 + (y-py)**2) ** 0.5
            if dist > 1000: long_stitches += 1
            if dist > 2000: very_long += 1
    return {
        "total": len(stitches),
        "commands": dict(cmd_counts),
        "long_stitches_>10mm": long_stitches,
        "very_long_>20mm": very_long,
        "bounds": p.bounds(),
    }

cases = [
    ("CCNapk", ["Test_Set_4/CCNapk.dst", "Test_Set_4/CCNapk.exp",
                "Test_Set_4/CCNapk.hus", "Test_Set_4/CCNapk.jef",
                "Test_Set_4/CCNapk.pes", "Test_Set_4/CCNapk.vp3"]),
    ("LGWreath", ["Test_Set_1/E-ASD-LGWreath.dst", "Test_Set_1/E-ASD-LGWreath.exp",
                  "Test_Set_1/E-ASD-LGWreath.hus", "Test_Set_1/E-ASD-LGWreath.jef",
                  "Test_Set_1/E-ASD-LGWreath.pes", "Test_Set_1/E-ASD-LGWreath.vp3"]),
    ("MdWreath", ["Test_Set_1/E-ASD-MdWreath.dst", "Test_Set_1/E-ASD-MdWreath.exp",
                  "Test_Set_1/E-ASD-MdWreath.hus", "Test_Set_1/E-ASD-MdWreath.jef",
                  "Test_Set_1/E-ASD-MdWreath.pes", "Test_Set_1/E-ASD-MdWreath.vp3"]),
    ("ButFlyL_control", ["Test_Set_1/E-ASD-ButFlyL.dst", "Test_Set_1/E-ASD-ButFlyL.pes"]),
]

for design, paths in cases:
    print(f"\n{'='*70}\n{design}\n{'='*70}")
    for rel in paths:
        full = ROOT / rel
        if not full.exists():
            print(f"  {rel}: NOT FOUND")
            continue
        r = analyze(full)
        if r is None:
            print(f"  {rel}: READ FAILED")
            continue
        print(f"  {rel}")
        print(f"    total commands: {r['total']}")
        print(f"    by type: {r['commands']}")
        print(f"    long stitches (>10mm): {r['long_stitches_>10mm']}")
        print(f"    very long (>20mm):     {r['very_long_>20mm']}")
        print(f"    bounds: {r['bounds']}")