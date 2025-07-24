import os

def clean_duplicate_gii(directory):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if filename ends with double .gii
        if filename.endswith('.gii.gii'):
            # Create the new filename by removing one .gii
            new_filename = filename[:-4]
            
            # Get full file paths
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            
            try:
                # Rename the file
                os.rename(old_filepath, new_filepath)
                print(f"Renamed: {filename} -> {new_filename}")
            except OSError as e:
                print(f"Error renaming {filename}: {e}")

# Example usage
if __name__ == "__main__":
    # Replace with your directory path
    directory_path = '/envau/work/meca/users/dienye.h/meso_envau_sync/dhcp_full_info/mean_curv_tex/'
    clean_duplicate_gii(directory_path)