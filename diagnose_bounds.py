# Save as: /Users/amans/Desktop/Mark-VLM/diagnose_bounds.py
import pyembroidery
from pathlib import Path

ROOT = Path("/Users/amans/Desktop/Mark-VLM/Metadata_Capture")

def stitch_bounds(path):
    p = pyembroidery.read(str(path))
    if not p:
        return None
    xs, ys = [], []
    for x, y, cmd in p.stitches:
        if cmd == pyembroidery.STITCH:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))

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
    ("ButFlyL_control", ["Test_Set_1/E-ASD-ButFlyL.dst", "Test_Set_1/E-ASD-ButFlyL.pes",
                         "Test_Set_1/E-ASD-ButFlyL.vp3"]),
]

for design, paths in cases:
    print(f"\n{design}")
    for rel in paths:
        full = ROOT / rel
        p = pyembroidery.read(str(full))
        if not p:
            continue
        full_bounds = p.bounds()
        stitch_only = stitch_bounds(full)
        full_w = float(full_bounds[2]) - float(full_bounds[0])
        full_h = float(full_bounds[3]) - float(full_bounds[1])
        stitch_w = float(stitch_only[2]) - float(stitch_only[0])
        stitch_h = float(stitch_only[3]) - float(stitch_only[1])
        print(f"  {rel}")
        print(f"    pattern.bounds():     w={full_w:7.1f}, h={full_h:7.1f}")
        print(f"    stitch-only bounds:   w={stitch_w:7.1f}, h={stitch_h:7.1f}")
        print(f"    diff (jump padding):  w={full_w-stitch_w:+7.1f}, h={full_h-stitch_h:+7.1f}")