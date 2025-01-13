import pandas as pd

# Load the Excel file
file_path = "cleaned_file.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# Display the first few rows of the original DataFrame
print("Original DataFrame:")
print(df.head())

# Interpolate the DataFrame
df_interpolated = df.interpolate(method='linear', limit = 5)

df_cleaned = df_interpolated.dropna(axis=1)

# Save the cleaned DataFrame to a new Excel file
output_file_path = "cleaned_file_date_indexed.pkl"
df_cleaned.to_pickle(output_file_path)

print(f"Cleaned DataFrame saved to {output_file_path}")
