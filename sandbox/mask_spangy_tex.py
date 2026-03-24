#!/usr/bin/env python3
"""
Mask one GIFTI file using another as a reference.
Vertices with value 100 in the reference file define which vertices to keep.
All other vertices in the target file are set to 0.
"""

import nibabel as nib
import numpy as np

def mask_gifti(reference_file, target_file, output_file):
    """
    Mask a GIFTI file using a reference file.
    
    Parameters:
    -----------
    reference_file : str
        Path to the reference GIFTI file (with values 0 or 100)
    target_file : str
        Path to the GIFTI file to be masked
    output_file : str
        Path to save the masked output
    """
    
    # Load the GIFTI files
    print(f"Loading reference file: {reference_file}")
    ref_gii = nib.load(reference_file)
    ref_data = ref_gii.darrays[0].data
    
    print(f"Loading target file: {target_file}")
    target_gii = nib.load(target_file)
    target_data = target_gii.darrays[0].data.copy()
    
    # Create mask from reference (vertices with value 100)
    mask = (ref_data == 100)
    n_selected = np.sum(mask)
    n_total = len(ref_data)
    
    print(f"\nReference statistics:")
    print(f"  Total vertices: {n_total}")
    print(f"  Selected vertices (value=100): {n_selected}")
    print(f"  Percentage selected: {100*n_selected/n_total:.2f}%")
    
    # Apply mask: set vertices to 0 where mask is False
    print(f"\nTarget statistics before masking:")
    print(f"  Min: {target_data.min():.6f}")
    print(f"  Max: {target_data.max():.6f}")
    print(f"  Mean: {target_data.mean():.6f}")
    print(f"  Non-zero vertices: {np.sum(target_data != 0)}")
    
    target_data[~mask] = 0
    
    print(f"\nTarget statistics after masking:")
    print(f"  Min: {target_data.min():.6f}")
    print(f"  Max: {target_data.max():.6f}")
    print(f"  Mean: {target_data.mean():.6f}")
    print(f"  Non-zero vertices: {np.sum(target_data != 0)}")
    
    # Create new GIFTI image with masked data
    masked_darray = nib.gifti.GiftiDataArray(
        data=target_data,
        intent=target_gii.darrays[0].intent,
        datatype=target_gii.darrays[0].datatype,
        meta=target_gii.darrays[0].meta
    )
    
    masked_gii = nib.gifti.GiftiImage(
        darrays=[masked_darray],
        meta=target_gii.meta
    )
    
    # Save the result
    print(f"\nSaving masked file to: {output_file}")
    nib.save(masked_gii, output_file)
    print("Done!")
    
    return target_data, mask

if __name__ == "__main__":
    
    reference_file = '/home/INT/dienye.h/python_files/rough/Segmented_mesh_incl_gyri.gii'
    target_file = '/home/INT/dienye.h/python_files/rough/spangy_dom_band_textures/smooth_5_spangy_dom_band_sub-0858_ses-0995_left.gii'
    output_file = '/home/INT/dienye.h/python_files/rough/segmented_smooth_5_spangy_dom_band_sub-0858_ses-0995_left_gyri_and_sulci.gii'
    
    mask_gifti(reference_file, target_file, output_file)