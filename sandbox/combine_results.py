import pandas as pd
import glob
import os

try:
    # Combine all chunk results
    result_files = glob.glob('Bounti info/Results/results/chunk_*_results.csv')
    if not result_files:
        print("No result files found!")
        exit(1)
        
    print(f"Found {len(result_files)} chunk files to combine")
    combined_df = pd.concat([pd.read_csv(f) for f in result_files])
    
    # Save combined results
    output_file = 'sandbox/combined_results.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"Combined results saved to {output_file}")
    print(f"Total records processed: {len(combined_df)}")
    
except Exception as e:
    print(f"Error combining results: {str(e)}")