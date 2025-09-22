import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Path to the folder
folder_path = r"C:\Users\mouli\OneDrive\Desktop\Capstone_New\final-b2c&b2b(for diagrams&graph)\b2b_new_transactions"

all_data = []

# Read and combine all CSVs
for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        try:
            df = pd.read_csv(os.path.join(folder_path, file))
            df.columns = df.columns.str.strip()
            df['Invoice date'] = pd.to_datetime(df['Invoice date'], dayfirst=True, errors='coerce')
            df['Taxable Value'] = pd.to_numeric(df['Taxable Value'], errors='coerce')
            df = df.dropna(subset=['Invoice date', 'Taxable Value'])
            all_data.append(df[['Invoice date', 'Taxable Value']])
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Combine all data
if not all_data:
    print("No valid data found.")
    exit()

combined = pd.concat(all_data, ignore_index=True)
combined = combined.sort_values('Invoice date')

# Monthly aggregation
monthly = combined.resample('M', on='Invoice date')['Taxable Value'].sum()
monthly = monthly.asfreq('M').fillna(0)

# Grid search for ARIMA(p,1,q)
results = []

for p in range(4):
    for q in range(4):
        try:
            model = ARIMA(monthly, order=(p, 1, q)).fit()
            results.append({
                'ARIMA(p,d,q)': f'ARIMA({p},1,{q})',
                'AIC': model.aic,
                'BIC': model.bic
            })
        except:
            continue

# Sort by AIC and display top 10
result_df = pd.DataFrame(results).sort_values(by='AIC').reset_index(drop=True)

print("\n Top ARIMA(p,1,q) Models Based on AIC:")
print(result_df.head(10))
