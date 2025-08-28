import pandas as pd

# Read the CSV files
scores_df = pd.read_csv('/home/INT/dienye.h/python_files/dhcp_dataset_info/qc_scores/updated_columns/dhcp_combined_qc_results.csv')  # File with names and scores
entries_df = pd.read_csv('/home/INT/dienye.h/python_files/dhcp_dataset_info/combined_results.csv')  # File with entry names

# Filter scores_df to get only entries with scores >= 3
high_scores_df = scores_df[scores_df['qc_score'] >= 3]

# Get the list of entry names that have scores >= 3
valid_entry_names = high_scores_df['participant_session'].tolist()

# Filter the entries_df to keep only rows with names that have scores >= 3
filtered_entries_df = entries_df[entries_df['participant_session'].isin(valid_entry_names)]

# Save the filtered result to a new CSV file
filtered_entries_df.to_csv('/home/INT/dienye.h/python_files/dhcp_dataset_info/dhcp_qc_filtered.csv', index=False)

print(f"Original entries: {len(entries_df)}")
print(f"Filtered entries: {len(filtered_entries_df)}")
print(f"Removed {len(entries_df) - len(filtered_entries_df)} entries with scores below 3")

# import pandas as pd

# # Read your CSV file (replace with your actual file path)
# df = pd.read_csv('/home/INT/dienye.h/python_files/dhcp_dataset_info/qc_scores/dHCP_fetal_BOUNTI_surfaces_left_qc.csv')

# df.columns = df.columns.str.lower()

# # Create the new column by combining subject and session id with the specified format
# df['participant_session'] = df['subject_id'].astype(str) + '_' + df['session_id'].astype(str) + '_left'

# # Display the result
# print(df.head())

# # Save to CSV if needed
# df.to_csv('/home/INT/dienye.h/python_files/dhcp_dataset_info/qc_scores/updated_columns/updated_dhcp_surfaces_left_qc.csv', index=False)