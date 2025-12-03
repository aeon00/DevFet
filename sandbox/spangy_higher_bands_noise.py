import numpy as np
import pandas as pd
import os
from glob import glob

# Directory containing frecomposed textures
frecomposed_dir = '/envau/work/meca/users/dienye.h/meso_envau_sync/dhcp_full_info/frecomposed/full/'
output_csv = '/envau/work/meca/users/dienye.h/meso_envau_sync/dhcp_full_info/info/power_above_b6_results.csv'

# Find all .npy files in directory
texture_files = sorted(glob(os.path.join(frecomposed_dir, '*.npy')))

print(f"Found {len(texture_files)} texture files")

# Store results
results = []

# Process each texture file
for texture_path in texture_files:
    # Extract subject/filename identifier
    filename = os.path.basename(texture_path)
    subject_id = os.path.splitext(filename)[0]  # Remove .npy extension
    
    print(f"\nProcessing: {subject_id}")
    
    try:
        # Load the frecomposed array (allow_pickle=True for object arrays)
        frecomposed = np.load(texture_path, allow_pickle=True)
        
        # Convert object array to numeric array if needed
        if frecomposed.dtype == 'object':
            print(f"  Converting from object array to numeric array")
            frecomposed = np.array(frecomposed.tolist(), dtype=float)
        
        # Check shape
        print(f"  Shape: {frecomposed.shape}")
        
        # Get number of bands
        nlevels = frecomposed.shape[1] + 1
        
        # Compute power for all bands
        all_band_powers = np.zeros(frecomposed.shape[1])
        for i in range(frecomposed.shape[1]):
            all_band_powers[i] = np.sum(frecomposed[:, i]**2)
        
        # Compute AFP (excluding B0)
        afp_all = np.sum(all_band_powers[1:])
        
        # Sum of power in all bands above B6
        power_above_b6 = np.sum(all_band_powers[7:])
        relative_power_above_b6 = power_above_b6 / afp_all if afp_all > 0 else 0
        
        # Store results
        result_dict = {
            'subject_id': subject_id,
            'total_power_above_b6': power_above_b6,
            'relative_power_above_b6': relative_power_above_b6,
            'total_afp': afp_all,
            'n_bands': frecomposed.shape[1]
        }
        
        # Optionally store individual band powers above B6
        for i in range(7, len(all_band_powers)):
            result_dict[f'B{i}_power'] = all_band_powers[i]
            result_dict[f'B{i}_relative'] = all_band_powers[i] / afp_all if afp_all > 0 else 0
        
        results.append(result_dict)
        
        print(f"  Total power B7+: {power_above_b6:.6f}")
        print(f"  Relative power B7+: {relative_power_above_b6:.6f}")
        
    except Exception as e:
        print(f"  ERROR processing {subject_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Create DataFrame and save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv(output_csv, index=False)

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Total subjects processed: {len(results)}")
print(f"Results saved to: {output_csv}")
print(f"{'='*60}")

# Display summary statistics
if len(results) > 0:
    print("\n** Summary Statistics **")
    print(df_results[['total_power_above_b6', 'relative_power_above_b6', 'total_afp']].describe())