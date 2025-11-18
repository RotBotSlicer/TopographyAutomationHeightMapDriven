# _02cutstl.py — Clean, safe interior cutting with top-gap protection
import os
import subprocess
from stl import mesh
from _10config import get_slic3r_binary, CUT_CONFIG


# -----------------------------------------------------------
#  READ CUT HEIGHTS
# -----------------------------------------------------------
def _read_cut_heights(cuts_file):
    """
    Read cut heights from cuts.txt → sorted list of floats.
    Ignores anything <= ignore_min (usually 0 mm).
    """
    ignore_min = CUT_CONFIG.get("ignore_cuts_at_or_below_mm", 0.0)

    heights = []
    with open(cuts_file, "r") as f:
        for line in f:
            line = line.strip().replace(",", ".")
            if not line:
                continue
            try:
                val = float(line)
            except ValueError:
                continue
            if val > ignore_min:
                heights.append(val)

    return sorted(set(heights))


# -----------------------------------------------------------
#  MAIN CUT FUNCTION
# -----------------------------------------------------------
def cutSTL(in_stl, cuts_file, out_folder):
    """
    Cut STL into stacked parts using safe interior cuts:
      • NEVER cut at z_min or z_max
      • NEVER cut too close to top (min_top_gap_mm)
      • Use safety offset (cut slightly LOWER inside geometry)
      • Works for ANY model height or shape
    """

    print("[cutSTL] Loading STL:", in_stl)
    os.makedirs(out_folder, exist_ok=True)

    # --- Load STL for z-min/max detection ---
    m = mesh.Mesh.from_file(in_stl)
    verts = m.vectors.reshape(-1, 3)
    z_min = float(verts[:, 2].min())
    z_max = float(verts[:, 2].max())
    height = z_max - z_min

    print(f"  Model height = {height:.3f} mm (z_min={z_min:.3f}, z_max={z_max:.3f})")

    # --- Read cut heights ---
    raw = _read_cut_heights(cuts_file)
    print("  Raw cut heights:", raw)

    # ---------------------------------------------------------
    # RULES FOR INTERIOR CUT SELECTION
    # ---------------------------------------------------------
    # 1. Remove z_min
    # 2. Remove z_max
    # 3. Remove cuts too close to the top (to avoid thin caps)
    min_top_gap = CUT_CONFIG.get("min_top_gap_mm", 2.0)

    interior = [
        z for z in raw
        if (z > z_min + 1e-6) and (z < (z_max - min_top_gap))
    ]

    print("  Interior usable cuts:", interior)

    # ---------------------------------------------------------
    # If no cuts → copy model as single segment
    # ---------------------------------------------------------
    base_name = os.path.splitext(os.path.basename(in_stl))[0]

    if not interior:
        dst = os.path.join(out_folder, f"{base_name}_1.stl")
        print("  No interior cuts → output single STL:", dst)
        os.replace(in_stl, dst)
        return

    # ---------------------------------------------------------
    # Cutting configuration
    # ---------------------------------------------------------
    safety_offset   = CUT_CONFIG.get("safety_offset_mm", 0.7)
    safety_min_edge = CUT_CONFIG.get("safety_min_margin_mm", 0.2)
    slicer          = get_slic3r_binary()

    # Important: process from top → down
    interior_desc = sorted(interior, reverse=True)

    tmp_parts = []
    tmp_counter = 0

    # ---------------------------------------------------------
    # PERFORM CUTS
    # ---------------------------------------------------------
    for cut_z in interior_desc:
        # Effective cut is slightly below --> ensures slicing inside the model
        cut_eff = cut_z - safety_offset

        # Clamp to ensure we never cut below z_min
        if cut_eff < z_min + safety_min_edge:
            cut_eff = z_min + safety_min_edge

        print(f"  Cutting at nominal {cut_z:.3f} → effective {cut_eff:.3f}")

        cmd = [
            slicer,
            "--dont-arrange",
            "--cut", str(cut_eff),
            "-o", "out.stl",
            in_stl
        ]
        print("    Running:", " ".join(cmd))
        res = subprocess.run(cmd)

        if res.returncode != 0:
            raise RuntimeError(f"Slic3r cut failed at plane {cut_eff:.3f}")

        upper = in_stl + "_upper.stl"
        lower = in_stl + "_lower.stl"

        if not os.path.exists(upper):
            raise FileNotFoundError(f"    ERROR: missing file {upper}")
        if not os.path.exists(lower):
            raise FileNotFoundError(f"    ERROR: missing file {lower}")

        # Move upper chunk to temp list (this is the top fragment)
        tmp_name = os.path.join(out_folder, f"_tmp_seg_{tmp_counter}.stl")
        os.replace(upper, tmp_name)
        tmp_parts.append(tmp_name)
        tmp_counter += 1

        # Continue cutting by replacing current model with lower part
        os.replace(lower, in_stl)

    # After all cuts → in_stl = bottom-most chunk
    bottom_chunk = in_stl

    # Final ordering: bottom-first → top-last
    ordered = [bottom_chunk] + list(reversed(tmp_parts))

    print("  Final ordered parts:")
    for i, part in enumerate(ordered, 1):
        print(f"    Part {i}: {part}")

    # Rename final segments
    for i, src in enumerate(ordered, 1):
        dst = os.path.join(out_folder, f"{base_name}_{i}.stl")
        os.replace(src, dst)
        print(f"    → Saved: {dst}")

    print("[cutSTL] Finished.\n")
