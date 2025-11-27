# _07combine.py  (FINAL — collision-safe, no-seam-extrusion version)
import os

# ---------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------

def _find_payload_start_index(lines):
    """
    Find where real printing begins in a G-code file.
    We look for the first ';TYPE:' except ';TYPE:Custom'.
    If none, fall back safely.
    """

    type_all = []
    type_non_custom = []

    for idx, row in enumerate(lines):
        if row.startswith(";TYPE:"):
            type_all.append(idx)
            if not row.startswith(";TYPE:Custom"):
                type_non_custom.append(idx)

    if type_non_custom:
        return type_non_custom[0]

    if type_all:
        return type_all[0]

    return 0


def _extract_print_payload(lines, is_first_file, context_lines_before_type=2):
    """
    For the first file -> return whole G-code (keep headers).
    For later files   -> strip headers and return only the print moves.

    We begin a little BEFORE the first ';TYPE:' line to keep necessary Z/E setup.
    """

    if is_first_file:
        return lines[:]  # keep full file

    type_idx = _find_payload_start_index(lines)

    # keep some lines before first type block for correct Z/E state
    start_idx = max(type_idx - context_lines_before_type, 0)

    return lines[start_idx:]


def _patch_first_movement(payload):
    """
    IMPORTANT FIX:
    Ensure the FIRST XY movement of this appended segment is a *travel move*
    (G0 without extrusion), even if slicer emits:
        G1 X… Y… Z… E0.0021  <-- (bad)
    We convert it to:
        G0 X… Y… Z…          <-- (good)
    """

    patched = []
    first_move_fixed = False

    for line in payload:
        stripped = line.strip()

        # Detect XY movement line
        if (not first_move_fixed and
            (stripped.startswith(("G1", "G0"))) and
            ("X" in stripped or "Y" in stripped)):

            # Build clean G0 travel move WITHOUT extrusion
            parts = stripped.split()
            new_parts = ["G0"]  # travel only

            for p in parts[1:]:
                # keep X,Y,Z,F — drop E
                if p.startswith(("X", "Y", "Z", "F")):
                    new_parts.append(p)

            patched.append(" ".join(new_parts) + "\n")
            first_move_fixed = True
            continue

        # Remove any early pure extrusion (E-only) before first XY move
        if (not first_move_fixed and
            (" E" in stripped) and
            ("X" not in stripped and "Y" not in stripped)):
            # skip this E-only command
            continue

        patched.append(line)

    return patched


# ---------------------------------------------------------------------
# MAIN COMBINER
# ---------------------------------------------------------------------

def combineGCode(in_folder, out_file, context_lines_before_type=2):
    """
    Safely combine multiple part G-code files:
      ✔ Keep full header from the FIRST file
      ✔ Strip headers from all later files
      ✔ Patch first movement of each segment to avoid:
            - nozzle collisions
            - extrusion across seam plane
      ✔ Add helpful segment headers for debugging
    """

    in_folder_abs = os.path.abspath(in_folder)
    out_file_abs  = os.path.abspath(out_file)

    print("[combineGCode] Combining from:", in_folder_abs)
    print("[combineGCode] Output:", out_file_abs)

    # Find all .gcode files
    gcode_files = [
        f for f in os.listdir(in_folder_abs)
        if f.lower().endswith(".gcode") and not f.startswith(".")
    ]
    gcode_files.sort()

    print("[combineGCode] Order:", gcode_files)

    with open(out_file_abs, "w", newline="\n") as fout:
        for idx, fname in enumerate(gcode_files):
            full_path = os.path.join(in_folder_abs, fname)
            print(f"[combineGCode] Appending {full_path}")

            with open(full_path, "r") as fin:
                lines = fin.readlines()

            # Strip headers from non-first files
            payload = _extract_print_payload(
                lines,
                is_first_file=(idx == 0),
                context_lines_before_type=context_lines_before_type,
            )

            # FIX: Patch first XY movement for all but first file
            if idx > 0:
                payload = _patch_first_movement(payload)

            # Write combined output
            fout.write(f"; --- START OF SEGMENT {fname} ---\n")
            for line in payload:
                fout.write(line)
            fout.write(f"; --- END OF SEGMENT {fname} ---\n")

    print("[combineGCode] DONE:", out_file_abs)
