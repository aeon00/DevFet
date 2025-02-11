import numpy as np
from scipy import stats as sps
import os
import slam.io as sio
import slam.texture as stex
import pandas as pd




def qc_measure(texture_file, z_thresh=3):
    """
    Accepts a texture file to check it's quality.
    :param z_thresh: z_score threshold
    :return: tuple of (most_extreme_value, its_z_score)
    """
    print(texture_file.darray.shape)
    filtered_darray = texture_file.darray.copy()
    max_abs_z = 0
    extreme_value = None
    
    for ind, d in enumerate(texture_file.darray):
        z = sps.zscore(d)
        # Find the index of maximum absolute z-score
        abs_z = np.abs(z)
        max_z_idx = np.argmax(abs_z)
        if abs_z[max_z_idx] > max_abs_z:
            max_abs_z = abs_z[max_z_idx]
            extreme_value = d[max_z_idx]
    
    texture_file.darray = filtered_darray
    print(texture_file.darray.shape)
    texture_file.metadata["z_score_filtered"] = True
    texture_file.metadata["z_score_threshold"] = z_thresh
    
    return extreme_value, max_abs_z

texture_path = '/envau/work/meca/users/dienye.h/Curvature/mean_curv_tex/'
data = []

for filename in os.listdir(texture_path):
    if filename.endswith('left.surf.gii') or filename.endswith(('right.surf.gii')):
        subject_id = filename.split('_')[3]
        session_id = filename.split('_')[4]
        texture = sio.load_texture(filename)
        extreme_value, max_abs_z = qc_measure(texture, z_thresh = 3)
        metrics = {
            "subject_id": subject_id,
            "session_id" : session_id,
            "extreme_value" : extreme_value,
            "max_abs_z" : max_abs_z
        }
        data.append(metrics)

df = pd.DataFrame(data)
