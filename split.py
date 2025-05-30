import pandas as pd

# --- 1. Load Your Data ---
# Replace 'your_msft_data.csv' with the actual path to your MSFT stock data file.
# Make sure your CSV has a 'Date' column and it's properly formatted.
try:
    df = pd.read_csv('your_msft_data.csv')
except FileNotFoundError:
    print("Error: 'your_msft_data.csv' not found. Please provide the correct path to your data.")
    exit() # Exit if the file isn't found

# --- 2. Prepare the Data (Crucial for Time Series) ---
# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort the DataFrame by date to ensure chronological order
df = df.sort_values(by='Date').reset_index(drop=True)

# --- 3. Define Split Ratios ---
# You can adjust these percentages based on your dataset size and needs.
# Common ratios: 70/15/15, 80/10/10, etc.
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15 # The sum of ratios should be 1 or close to 1

# --- 4. Perform the Chronological Split ---
total_rows = len(df)

# Calculate the number of rows for each set
train_rows = int(total_rows * train_ratio)
val_rows = int(total_rows * val_ratio)

# Split the DataFrame
train_df = df.iloc[:train_rows]
val_df = df.iloc[train_rows : train_rows + val_rows]
test_df = df.iloc[train_rows + val_rows :]

# --- 5. Save to Separate CSV Files ---
# Using index=False prevents pandas from writing the DataFrame index as a column
# in the CSV file, which is generally desired for clean data.
try:
    train_df.to_csv('train_clean.csv', index=False)
    val_df.to_csv('val_clean.csv', index=False)
    test_df.to_csv('test_clean.csv', index=False)

    print("\nData successfully split and saved:")
    print(f"- 'train_clean.csv' created with {len(train_df)} rows.")
    print(f"- 'val_clean.csv' created with {len(val_df)} rows.")
    print(f"- 'test_clean.csv' created with {len(test_df)} rows.")
    print("\nDate ranges for each set:")
    print(f"  Train: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Validation: {val_df['Date'].min().strftime('%Y-%m-%d')} to {val_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Test: {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}")

except Exception as e:
    print(f"An error occurred while saving files: {e}")
