# _01analysestl.py — Clean Version (Automatic Cuts + Overhang Detection + 3-column cuts.txt)
import numpy as np
from stl import mesh
from _10config import CUT_CONFIG


# --------------------------------------------------------------------
#  NORMAL UTILITIES
# --------------------------------------------------------------------

def _triangle_normals(vectors):
    """
    Compute normal directions for each triangle in an Nx3x3 array.
    """
    v1 = vectors[:, 1, :] - vectors[:, 0, :]
    v2 = vectors[:, 2, :] - vectors[:, 0, :]
    normals = np.cross(v1, v2)

    # Normalize safely
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1e-12
    normals /= lengths[:, None]

    return normals


# --------------------------------------------------------------------
#  AUTOMATIC CUT HEIGHTS (candidate planes)
# --------------------------------------------------------------------

def compute_cut_heights(stl_path, dz_min=6.0):
    """
    Automatically compute vertical cut heights (candidate planes).

    Parameters
    ----------
    stl_path : str
        Path to the input STL.
    dz_min : float
        Minimal height difference allowed between cuts.
        Prevents too many unnecessary small cuts.

    Returns
    -------
    list of floats
        Sorted Z-values (including z=0 and z_max) that define
        natural candidate layers. These are later filtered into
        interior cuts for actual segmentation.
    """
    body = mesh.Mesh.from_file(stl_path)
    verts = body.vectors.reshape(-1, 3)

    z_values = verts[:, 2]
    z_min, z_max = float(np.min(z_values)), float(np.max(z_values))

    # Detect natural changes in silhouette / shape by checking
    # triangle normal changes vs Z.
    vectors = body.vectors
    normals = _triangle_normals(vectors)

    # A triangle is "horizontal-ish" if its normal is almost vertical.
    # These typically form natural stable layers.
    vertical_alignment = np.abs(normals[:, 2])  # 1 = vertical normal = horizontal triangle
    is_flat = vertical_alignment > 0.95  # threshold — tune if needed

    flat_triangle_z = np.mean(vectors[is_flat, :, 2], axis=1) if np.any(is_flat) else []

    # Combine: natural flat regions + min/max + base at 0.0
    raw_cuts = [0.0]   # MUST include zero as a reference
    raw_cuts.extend(flat_triangle_z)
    raw_cuts.extend([z_min, z_max])

    raw_cuts = sorted(list(set([float(z) for z in raw_cuts])))

    # Reduce cuts: keep only those separated by >= dz_min
    final_cuts = [raw_cuts[0]]
    for z in raw_cuts[1:]:
        if z - final_cuts[-1] >= dz_min:
            final_cuts.append(z)

    # Ensure last cut is top
    if final_cuts[-1] != z_max:
        final_cuts.append(z_max)

    return final_cuts


# --------------------------------------------------------------------
#  INTERNAL: SEGMENT BOUNDS + TRANSFORM FLAGS
# --------------------------------------------------------------------

def _compute_segment_bounds_and_flags(stl_path, dz_min=6.0, angle_deg=45.0):
    """
    Compute vertical segments [z_low, z_high] and a transform flag per segment.

    - Uses compute_cut_heights() to get candidate planes.
    - Uses CUT_CONFIG to mimic the same "interior cut" logic as cutSTL:
        * ignore_cuts_at_or_below_mm
        * min_top_gap_mm
    - Each segment gets a flag:
        1 => heightmap-transform this segment
        0 => keep planar

    Additional rules:
        - If there are >=2 segments:
            * bottom segment flag = 0 (planar)
            * top segment flag   = 1 (transformed)
        - If there is only 1 segment:
            * flag is based solely on overhang detection
              (single-part behaviour same as old pipeline).
    """
    body = mesh.Mesh.from_file(stl_path)
    vectors = body.vectors
    verts = vectors.reshape(-1, 3)
    z_values = verts[:, 2]
    z_min, z_max = float(np.min(z_values)), float(np.max(z_values))

    if z_max <= z_min:
        # Degenerate case — treat as a single flat segment, planar by default.
        return [(z_min, z_max)], [0]

    # Candidate cuts from geometry (includes 0 and z_max)
    candidate_cuts = compute_cut_heights(stl_path, dz_min=dz_min)
    candidate_cuts = sorted(candidate_cuts)

    # Mimic _02cutstl.py logic for which cuts become interior planes
    ignore_min = CUT_CONFIG.get("ignore_cuts_at_or_below_mm", 0.0)
    min_top_gap = CUT_CONFIG.get("min_top_gap_mm", 2.0)

    # First emulate _read_cut_heights behaviour
    raw = []
    for z in candidate_cuts:
        if z <= ignore_min:
            continue
        raw.append(float(z))

    # Interior cuts (actual planes we will cut at)
    interior_cuts = [
        z for z in raw
        if (z > z_min + 1e-6) and (z < (z_max - min_top_gap))
    ]
    interior_cuts = sorted(list(set(interior_cuts)))

    # Build segment bounds: [z_min → first_cut], [cut_i → cut_{i+1}], ..., [last_cut → z_max]
    segment_bounds = []
    lower = z_min
    for zc in interior_cuts:
        segment_bounds.append((lower, zc))
        lower = zc
    segment_bounds.append((lower, z_max))  # top segment

    # Precompute triangle normals and z-extent
    normals = _triangle_normals(vectors)
    tri_z = vectors[:, :, 2]
    tri_z_min = tri_z.min(axis=1)
    tri_z_max = tri_z.max(axis=1)

    flags = []
    for (low, high) in segment_bounds:
        # Triangles that intersect [low, high]
        mask = ~((tri_z_max < low) | (tri_z_min > high))
        if not np.any(mask):
            # No triangles in this band → treat as planar
            flags.append(0)
            continue

        seg_normals = normals[mask]
        dot_up = seg_normals[:, 2]
        angle = np.degrees(np.arccos(np.clip(dot_up, -1.0, 1.0)))

        overhang_mask = angle > angle_deg
        flag = 1 if np.any(overhang_mask) else 0
        flags.append(flag)

    # Apply bottom/top overrides only when there are multiple segments
    if len(flags) >= 2:
        # Bottom segment → forced planar
        flags[0] = 0
        # Top segment → forced transform
        flags[-1] = 1

    return segment_bounds, flags


# --------------------------------------------------------------------
#  WRITE 3-COLUMN CUTS TO FILE
# --------------------------------------------------------------------

def analyseSTL(stl_path, cuts_txt_path, dz_min=6.0, angle_deg=45.0):
    """
    Analyse STL → determine segment bounds + transform flags → write 3-column cuts.txt

    Output format (per line):
        index  transform_flag  z_value
    for all segments except the last, and:
        index  transform_flag  TOP
    for the topmost segment.

    Example:
        1 0 20.0000
        2 1 30.0000
        3 0 37.6410
        4 1 45.3000
        5 1 TOP
    """
    print("  [analyseSTL] Loading:", stl_path)
    segment_bounds, flags = _compute_segment_bounds_and_flags(
        stl_path, dz_min=dz_min, angle_deg=angle_deg
    )

    with open(cuts_txt_path, "w") as f:
        for idx, ((low, high), flag) in enumerate(zip(segment_bounds, flags), start=1):
            if idx == len(segment_bounds):
                # Topmost segment → write TOP
                f.write(f"{idx} {int(flag)} TOP\n")
            else:
                # Non-top segments → write upper boundary as the cut Z-value
                f.write(f"{idx} {int(flag)} {high:.6f}\n")

    print("  [analyseSTL] Segments (low → high, flag):")
    for idx, ((low, high), flag) in enumerate(zip(segment_bounds, flags), start=1):
        col3 = "TOP" if idx == len(segment_bounds) else f"{high:.6f}"
        print(
            f"    {idx}: flag={int(flag)}  "
            f"range=[{low:.6f}, {high:.6f}]  → cuts.txt col3={col3}"
        )

    print("  [analyseSTL] Saved 3-column cuts file to:", cuts_txt_path)


# --------------------------------------------------------------------
#  LEGACY SEGMENT OVERHANG DETECTION (fallback use)
# --------------------------------------------------------------------

def segment_has_overhang(stl_path, angle_deg=45.0):
    """
    Determine if an STL contains ANY downward-facing triangles.

    Overhang rule:
        If the angle between triangle normal and +Z is > angle_deg,
        the part is considered “nonplanar” or “needs deformation”.

    This is kept for backwards compatibility / fallback logic.
    """
    body = mesh.Mesh.from_file(stl_path)
    normals = _triangle_normals(body.vectors)

    # Dot with +Z
    dot_up = normals[:, 2]

    # Angle between normal and +Z
    angle = np.arccos(np.clip(dot_up, -1.0, 1.0)) * 180.0 / np.pi

    overhang_mask = angle > angle_deg
    return bool(np.any(overhang_mask))
