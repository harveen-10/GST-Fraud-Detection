import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

# Set folder path
folder_path = r"C:\Users\mouli\OneDrive\Desktop\Capstone_New\final-b2c&b2b(for diagrams&graph)\b2b_new_transactions"

print("Reading files from:", folder_path)
all_data = []

# Read all CSVs in the folder
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(path)

            # Parse invoice date
            df['Invoice date'] = pd.to_datetime(df['Invoice date'], dayfirst=True, errors='coerce')
            df.dropna(subset=['Invoice date'], inplace=True)

            # Ensure taxable value is numeric
            df['Taxable Value'] = pd.to_numeric(df['Taxable Value'], errors='coerce')
            df.dropna(subset=['Taxable Value'], inplace=True)

            all_data.append(df[['Invoice date', 'Taxable Value']])
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Combine all data
combined_df = pd.concat(all_data)
combined_df = combined_df.sort_values('Invoice date')

# Aggregate taxable value by date
ts = combined_df.groupby('Invoice date')['Taxable Value'].sum()

# ========================
# ADF Test 
# ========================
print("\nRunning ADF Test on B2B Time Series")
result = adfuller(ts.dropna())
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
print("Critical Values:")
for key, value in result[4].items():
    print(f"   {key}: {value}")
if result[1] < 0.05:
    print("ADF: Series is stationary")
else:
    print("ADF: Series is non-stationary")

# ========================
# KPSS Test 
# ========================
print("\nRunning KPSS Test on B2B Time Series")
kpss_result = kpss(ts.dropna(), regression='c', nlags='auto')
print(f"KPSS Statistic: {kpss_result[0]}")
print(f"p-value: {kpss_result[1]}")
print("Critical Values:")
for key, value in kpss_result[3].items():
    print(f"   {key}: {value}")
if kpss_result[1] > 0.05:
    print("KPSS: Series is stationary")
else:
    print("KPSS: Series is non-stationary")

