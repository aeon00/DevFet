import pandas as pd

# Recreate the table from the image
df = pd.read_csv("/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref/cv_results_summary.csv")

# Capitalise model labels: m1a -> M1a
df["Model"] = df["Model"].str.upper().str.replace("M1", "M1", regex=False)

# Pivot: rows = Feature, columns = Model, values = LogScore_mean
pivot = df.pivot(index="Feature", columns="Model", values="LogScore_mean")

# Rename column axis label for cleanliness
pivot.columns.name = None
pivot.index.name = "Feature"

# Reorder columns in case they are not sorted
col_order = [c for c in ["M1A", "M1B", "M1C", "M1D", "M1E"] if c in pivot.columns]
pivot = pivot[col_order]

# Add Best Model column: higher LogScore_mean is better
pivot["Best Model"] = pivot[col_order].idxmax(axis=1)

print(pivot.to_string())

# Save to CSV
pivot.to_csv("/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref/logscore_pivot.csv")
print("\nSaved to logscore_pivot.csv")