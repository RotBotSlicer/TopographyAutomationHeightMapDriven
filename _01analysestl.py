# _01analysestl.py — Clean Version (Automatic Cuts + Overhang Detection)
import numpy as np
from stl import mesh


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
#  AUTOMATIC CUT HEIGHTS
# --------------------------------------------------------------------

def compute_cut_heights(stl_path, dz_min=6.0):
    """
    Automatically compute vertical cut heights.

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
        Sorted Z-values at which to cut the STL.
        Always includes z=0.
    """
    body = mesh.Mesh.from_file(stl_path)
    verts = body.vectors.reshape(-1, 3)

    z_values = verts[:, 2]
    z_min, z_max = float(np.min(z_values)), float(np.max(z_values))

    # We detect natural changes in silhouette / shape
    # by checking triangle normal changes vs Z.
    vectors = body.vectors
    normals = _triangle_normals(vectors)

    # A triangle is "horizontal-ish" if its normal is almost vertical.
    # These typically form natural stable layers.
    vertical_alignment = np.abs(normals[:, 2])  # 1 = vertical normal = horizontal triangle
    is_flat = vertical_alignment > 0.95  # threshold — tune if needed

    flat_triangle_z = np.mean(vectors[is_flat, :, 2], axis=1) if np.any(is_flat) else []

    # Combine: natural flat regions + min/max
    raw_cuts = [0.0]   # MUST include zero
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
#  WRITE CUTS TO FILE
# --------------------------------------------------------------------

def analyseSTL(stl_path, cuts_txt_path):
    """
    Analyse STL → determine cut heights → write to cuts.txt
    """
    print("  [analyseSTL] Loading:", stl_path)
    cuts = compute_cut_heights(stl_path, dz_min=6.0)

    with open(cuts_txt_path, "w") as f:
        for z in cuts:
            f.write(f"{z:.6f}\n")

    print("  [analyseSTL] Computed cut heights:", cuts)
    print("  [analyseSTL] Saved to:", cuts_txt_path)


# --------------------------------------------------------------------
#  SEGMENT OVERHANG DETECTION
# --------------------------------------------------------------------

def segment_has_overhang(stl_path, angle_deg=45.0):
    """
    Determine if a segmented STL contains ANY downward-facing triangles.

    Overhang rule:
        If the angle between triangle normal and +Z is > angle_deg,
        the part is considered “nonplanar” or “needs deformation”.

    Parameters
    ----------
    stl_path : str
        Path to a segment STL (from stl_parts/)
    angle_deg : float
        Threshold angle. 45° is a good universal default.

    Returns
    -------
    bool
        True  → this segment should be heightmap-transformed
        False → this segment can remain planar
    """
    body = mesh.Mesh.from_file(stl_path)
    normals = _triangle_normals(body.vectors)

    # Dot with +Z
    dot_up = normals[:, 2]

    # Angle between normal and +Z
    # angle = arccos(dot_up)
    angle = np.arccos(np.clip(dot_up, -1.0, 1.0)) * 180.0 / np.pi

    # Downward & overhang if angle > angle_deg
    overhang_mask = angle > angle_deg

    return bool(np.any(overhang_mask))
