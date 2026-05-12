"""
SPANGY smoothing-effect sweep.

For each mesh, runs the SPANGY pipeline four times:
    no smoothing, nb_iter=5, nb_iter=10, nb_iter=20
and stores only the analyzed folding power (AFP) and the B4/B5/B6
band powers, plus subject metadata.

One CSV is written per SLURM array task to avoid concurrent-write races.
Aggregation and plotting are done in plot_smoothing_effects.py.
"""

import os
import re
import sys
import time

import numpy as np
import pandas as pd

import slam.io as sio
import slam.curvature as scurv
import slam.spangy as spgy
import slam.texture as stex
from slam.differential_geometry import laplacian_mesh_smoothing


# ----------------------------- configuration -----------------------------

MESH_PATH = "/envau/work/meca/users/dienye.h/B7_analysis/mesh/"
OUTPUT_PATH = "/envau/work/meca/users/dienye.h/B7_analysis/smoothing_effect/"
SMOOTHING_VALUES = [0, 5, 10, 20]   # 0 = no smoothing
SMOOTHING_DT = 0.1
N_EIGENPAIRS = 5000


# ----------------------------- helpers -----------------------------------

def extract_sub_sess_hemi(filename):
    """Return 'sub-XXXX_ses-XXXX_<hemi>' or None."""
    pattern = r"(sub-\d+)_(ses-\d+).*?(left|right)"
    match = re.search(pattern, filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
    return None


def run_spangy(mesh, n_eigenpairs=N_EIGENPAIRS):
    """Run the core SPANGY pipeline on a (possibly smoothed) mesh.

    Returns the grouped spectrum array.
    """
    eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, n_eigenpairs)

    PrincipalCurvatures, _, _ = scurv.curvatures_and_derivatives(mesh)
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])

    tex_mean_curv = stex.TextureND(mean_curv)
    tex_mean_curv.z_score_filtering(z_thresh=3)
    mean_curv = tex_mean_curv.darray.squeeze()

    grouped_spectrum, _, _, _ = spgy.spectrum(mean_curv, lap_b, eigVects, eigVal)
    return grouped_spectrum


def process_single_file_at_smoothing(mesh_file, smoothing_iter):
    """Run SPANGY once on the given file with `smoothing_iter` Laplacian steps.

    Returns a dict with afp, B4, B5, B6 and processing time.
    """
    t0 = time.time()

    mesh = sio.load_mesh(mesh_file)
    if smoothing_iter > 0:
        mesh = laplacian_mesh_smoothing(
            mesh, nb_iter=smoothing_iter, dt=SMOOTHING_DT
        )

    grouped_spectrum = run_spangy(mesh)

    afp = float(np.sum(grouped_spectrum[1:]))
    bands = {}
    for k in (4, 5, 6):
        bands[f"B{k}"] = (
            float(grouped_spectrum[k]) if k < len(grouped_spectrum) else 0.0
        )

    return {
        "smoothing_iter": smoothing_iter,
        "afp": afp,
        "B4": bands["B4"],
        "B5": bands["B5"],
        "B6": bands["B6"],
        "processing_time_s": time.time() - t0,
    }


# ----------------------------- driver ------------------------------------

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print(f"Scanning directory: {MESH_PATH}")
    all_files = sorted(
        f for f in os.listdir(MESH_PATH)
        if f.endswith("left.surf.gii") or f.endswith("right.surf.gii")
    )
    print(f"Found {len(all_files)} mesh files")

    # SLURM array partition (falls back to single task if not in SLURM)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    chunk = len(all_files) // n_tasks + (1 if len(all_files) % n_tasks else 0)
    start_idx = task_id * chunk
    end_idx = min((task_id + 1) * chunk, len(all_files))
    my_files = all_files[start_idx:end_idx]
    print(f"Task {task_id + 1}/{n_tasks}: processing files {start_idx}–{end_idx}")

    csv_file = os.path.join(OUTPUT_PATH, f"smoothing_results_task{task_id:03d}.csv")
    rows = []

    for filename in my_files:
        mesh_file = os.path.join(MESH_PATH, filename)
        subj = extract_sub_sess_hemi(filename)
        print(f"\n→ {subj}")

        for smooth in SMOOTHING_VALUES:
            try:
                res = process_single_file_at_smoothing(mesh_file, smooth)
                res["subject"] = subj
                res["filename"] = filename
                res["hemisphere"] = "left" if "left" in filename else "right"
                rows.append(res)
                print(
                    f"   iter={smooth:>2}: AFP={res['afp']:.2f}  "
                    f"B4={res['B4']:.2f}  B5={res['B5']:.2f}  "
                    f"B6={res['B6']:.2f}  ({res['processing_time_s']:.1f}s)"
                )
            except Exception as e:
                print(f"   iter={smooth}: FAILED — {e}")
                rows.append({
                    "subject": subj, "filename": filename,
                    "hemisphere": "left" if "left" in filename else "right",
                    "smoothing_iter": smooth,
                    "afp": np.nan, "B4": np.nan, "B5": np.nan, "B6": np.nan,
                    "processing_time_s": np.nan,
                })

        # Incremental flush: safe against crashes / wall-time kills.
        pd.DataFrame(rows).to_csv(csv_file, index=False)

    print(f"\nDone. Wrote {csv_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Critical error in main: {exc}")
        sys.exit(1)