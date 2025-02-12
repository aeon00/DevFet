import numpy as np
from scipy import stats as sps
import os
import slam.io as sio
import slam.texture as stex
import pandas as pd

def qc_measure(texture_file, z_thresh=3):
    """
    Accepts a texture file to check its quality and analyze distribution characteristics.
    :param z_thresh: z_score threshold
    :return: dict containing distribution characteristics
    """
    print(texture_file.darray.shape)
    filtered_darray = texture_file.darray.copy()
    max_abs_z = 0
    extreme_value = None
    z_scores_all = []
    
    for ind, d in enumerate(texture_file.darray):
        z = sps.zscore(d)
        z_scores_all.append(z)
        abs_z = np.abs(z)
        max_z_idx = np.argmax(abs_z)
        if abs_z[max_z_idx] > max_abs_z:
            max_abs_z = abs_z[max_z_idx]
            extreme_value = d[max_z_idx]
    
    # Combine all z-scores
    z_scores_combined = np.concatenate(z_scores_all)
    
    # Calculate comprehensive distribution statistics
    stats = {
        # Basic statistics
        'mean': np.mean(z_scores_combined),
        'median': np.median(z_scores_combined),
        'variance': np.var(z_scores_combined),
        'std_dev': np.std(z_scores_combined),
        # Range statistics
        'min': np.min(z_scores_combined),
        'max': np.max(z_scores_combined),
        'range': np.ptp(z_scores_combined),
        # Quartile statistics
        'q1': np.percentile(z_scores_combined, 25),
        'q3': np.percentile(z_scores_combined, 75),
        'iqr': sps.iqr(z_scores_combined),
        # Shape statistics
        'skewness': sps.skew(z_scores_combined),
        'kurtosis': sps.kurtosis(z_scores_combined),
        # Original QC measures
        'extreme_value': extreme_value,
        'max_abs_z': max_abs_z,
        # Additional distribution characteristics
        'mode': sps.mode(z_scores_combined)[0],
        'coefficient_variation': sps.variation(z_scores_combined),
        # Normality tests
        'shapiro_test': sps.shapiro(z_scores_combined[:5000])[1],
        'normaltest': sps.normaltest(z_scores_combined)[1]
    }
    
    # Add quantiles
    quantile_levels = [1, 5, 10, 90, 95, 99]
    for q in quantile_levels:
        stats[f'percentile_{q}'] = np.percentile(z_scores_combined, q)
    
    texture_file.darray = filtered_darray
    print(texture_file.darray.shape)
    texture_file.metadata["z_score_filtered"] = True
    texture_file.metadata["z_score_threshold"] = z_thresh
    
    return stats

# Main processing loop
texture_path = '/envau/work/meca/users/dienye.h/Curvature/mean_curv_tex/'
data = []

for filename in os.listdir(texture_path):
    if filename.endswith('left.surf.gii'):
        subject_id = filename.split('_')[3]
        session_id = filename.split('_')[4]
        file = os.path.join(texture_path, filename)
        texture = sio.load_texture(file)
        
        # Get all statistics for this texture file
        stats = qc_measure(texture, z_thresh=3)
        
        # Create metrics dictionary with all statistics
        metrics = {
            "subject_id": subject_id,
            "session_id": session_id,
            **stats  # Unpack all statistics into the metrics dictionary
        }
        
        data.append(metrics)

# Create DataFrame with all metrics
df = pd.DataFrame(data)

# Save to CSV with all statistics
df.to_csv('/envau/work/meca/users/dienye.h/qc_measure_full.csv', index=False)