import os
import pandas as pd
from pathlib import Path

def match_files_to_csv(directory_path, csv_file_path, name_column, output_csv_path=None):
    """
    Check file names in a directory against names in a CSV column.
    Creates an 'impact' column: 'Excluded' if name matches a file, 'Included' if not.
    
    Args:
        directory_path (str): Path to directory containing files
        csv_file_path (str): Path to the CSV file
        name_column (str): Name of the column containing names to match
        output_csv_path (str, optional): Path for output CSV. If None, overwrites input file.
    """
    
    # Get list of file names (without extensions) from directory
    try:
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Get file names without extensions
        file_names = set()
        for file_path in directory.iterdir():
            if file_path.is_file():
                new_file_name = file_path.stem
                new_file_name = new_file_name.replace("smooth_5_", "").replace("reo-SVR-output-brain-mask-brain_bounti-white.", "").replace(".surf", "")
                file_names.add(new_file_name)  # stem gives filename without extension
        
        print(f"Found {len(file_names)} files in directory")
        
    except Exception as e:
        print(f"Error reading directory: {e}")
        return
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"CSV loaded with {len(df)} rows")
        
        if name_column not in df.columns:
            raise ValueError(f"Column '{name_column}' not found in CSV. Available columns: {list(df.columns)}")
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Create impact column based on matches
    def check_impact(name):
        if pd.isna(name):  # Handle NaN values
            return "Included"
        return "Excluded" if str(name) in file_names else "Included"
    
    df['impact'] = df[name_column].apply(check_impact)
    
    # Count matches
    excluded_count = (df['impact'] == 'Excluded').sum()
    included_count = (df['impact'] == 'Included').sum()
    
    print(f"Results:")
    print(f"  Excluded (matched): {excluded_count}")
    print(f"  Included (no match): {included_count}")
    
    # Save the updated CSV
    output_path = output_csv_path or csv_file_path
    try:
        df.to_csv(output_path, index=False)
        print(f"Updated CSV saved to: {output_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return
    
    return df

# Example usage
if __name__ == "__main__":
    # Configuration - modify these paths and column name as needed
    DIRECTORY_PATH = "/home/INT/dienye.h/python_files/qc_identified_meshes/dHCP/mesh"  # Change this to your directory
    CSV_FILE_PATH = "/home/INT/dienye.h/python_files/dhcp_dataset_info/combined_results.csv"  # Change this to your CSV file
    NAME_COLUMN = "participant_session"  # Change this to your column name
    OUTPUT_CSV_PATH = "/home/INT/dienye.h/python_files/dhcp_dataset_info/included_excluded_subjects.csv"  # Optional: specify output file
    
    # Run the matching
    result_df = match_files_to_csv(
        directory_path=DIRECTORY_PATH,
        csv_file_path=CSV_FILE_PATH,
        name_column=NAME_COLUMN,
        output_csv_path=OUTPUT_CSV_PATH  # Remove this line to overwrite original file
    )
    
    # Optional: Display first few rows of results
    if result_df is not None:
        print("\nFirst 5 rows of results:")
        print(result_df[['participant_session', 'impact']].head())  # Adjust column names as needed