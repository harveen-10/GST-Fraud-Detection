import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Folder path 
folder_path = r"C:\Users\mouli\OneDrive\Desktop\Capstone_New\final-b2c&b2b(for diagrams&graph)\b2b_new_transactions"
all_data = []

print(f"Reading files from: {folder_path}")

# Read and clean all CSVs
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)
            
            if 'Invoice date' in df.columns and 'Taxable Value' in df.columns:
                # Convert and clean
                df['Invoice date'] = pd.to_datetime(df['Invoice date'], dayfirst=True, errors='coerce')
                df['Taxable Value'] = pd.to_numeric(df['Taxable Value'], errors='coerce')
                df = df.dropna(subset=['Invoice date', 'Taxable Value'])
                all_data.append(df[['Invoice date', 'Taxable Value']])
            else:
                print(f"Missing columns in {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Combine and aggregate monthly
combined_df = pd.concat(all_data, ignore_index=True)
monthly_series = combined_df.resample('ME', on='Invoice date')['Taxable Value'].sum()

print("\nMonthly aggregated taxable values (first 5 rows):")
print(monthly_series.head())

# Set max lags safely
max_lags = min(20, len(monthly_series) // 2 - 1)
print(f"\nUsing max_lags = {max_lags} for ACF and PACF")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_acf(monthly_series.dropna(), ax=axes[0], lags=max_lags)
axes[0].set_title('ACF - Monthly Taxable Value (B2B Transactions)')
plot_pacf(monthly_series.dropna(), ax=axes[1], lags=max_lags)
axes[1].set_title('PACF - Monthly Taxable Value (B2B Transactions)')
plt.tight_layout()
plt.show()
