import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Path to the folder
folder_path = r"C:\Users\mouli\OneDrive\Desktop\Capstone_New\final-b2c&b2b(for diagrams&graph)\b2b_new_transactions"

# ARIMA models to compare
models = [(2,1,0), (2,1,1)]
all_results = []

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)
            # Parse 'Invoice date'
            df['Invoice date'] = pd.to_datetime(df['Invoice date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Invoice date'])
            # Convert Taxable Value
            df['Taxable Value'] = pd.to_numeric(df['Taxable Value'], errors='coerce')
            df = df.dropna(subset=['Taxable Value'])
            # Resample monthly
            monthly = df.resample('M', on='Invoice date')['Taxable Value'].sum()
            
            # Filter: skip if not enough data or all values are 0
            if len(monthly) < 6 or monthly.sum() == 0:
                print(f" Skipping file (zero or insufficient data): {file}")
                continue
            
            # Train-test split (80% train, 20% test)
            split_point = int(len(monthly) * 0.8)
            train, test = monthly[:split_point], monthly[split_point:]
            
            # Test each ARIMA model
            for order in models:
                try:
                    model = ARIMA(train, order=order)
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=len(test))
                    rmse = np.sqrt(mean_squared_error(test, forecast))
                    
                    all_results.append({
                        'File': file,
                        'Model': f'ARIMA{order}',
                        'RMSE': rmse
                    })
                except:
                    continue
                    
        except Exception as e:
            print(f" Error in file {file}: {e}")
            continue

# Create DataFrame and analyze results
results_df = pd.DataFrame(all_results)

# Show best model for each file
best_models = results_df.loc[results_df.groupby('File')['RMSE'].idxmin()]
print("\n Best Model per File:")
print(best_models[['File', 'Model', 'RMSE']].head(10))

# Show overall model performance
print("\n Average RMSE by Model:")
print(results_df.groupby('Model')['RMSE'].mean().round(2).to_string())