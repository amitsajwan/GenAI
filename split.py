import pandas as pd
from pathlib import Path

def split_and_save_timeseries_data(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    output_dir: str = "data",
    output_prefix: str = "cleaned",
    date_column_for_range_info: str = "Date" # Column used for printing date ranges
) -> bool:
    """
    Splits a time-series DataFrame into train, validation, and test sets sequentially
    and saves them to CSV files, preserving the index.

    Args:
        df (pd.DataFrame): The input DataFrame, assumed to be sorted by date.
        train_ratio (float): Proportion of data for the training set (e.g., 0.7).
        val_ratio (float): Proportion of data for the validation set (e.g., 0.15).
                           The test set will be the remainder.
        output_dir (str): Directory to save the output CSV files.
        output_prefix (str): Prefix for the output filenames (e.g., "cleaned", "featured").
        date_column_for_range_info (str): Name of the column containing dates
                                          for printing range information. If your date is
                                          the index itself and no such column exists,
                                          you'll need to adjust the print statements.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and (train_ratio + val_ratio) < 1):
        print("Error: train_ratio and val_ratio must be between 0 and 1, and their sum must be less than 1.")
        return False

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_rows = len(df)
    train_rows = int(total_rows * train_ratio)
    val_rows = int(total_rows * val_ratio)
    # test_rows will be total_rows - train_rows - val_rows

    print(f"Input DataFrame has {total_rows} rows.")
    print(f"The DataFrame's index is named: {df.index.name}") # Good to check index name

    # --- Data Splitting (Sequential for Time Series) ---
    train_df = df.iloc[0:train_rows]
    val_df = df.iloc[train_rows : train_rows + val_rows]
    test_df = df.iloc[train_rows + val_rows :]

    if train_df.empty or val_df.empty or test_df.empty:
        print("Error: One or more of the data splits is empty. Check DataFrame length and ratios.")
        print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")
        return False

    # --- Save to Separate CSV Files, Preserving the Index ---
    train_file = Path(output_dir) / f"{output_prefix}_train.csv"
    val_file = Path(output_dir) / f"{output_prefix}_val.csv"
    test_file = Path(output_dir) / f"{output_prefix}_test.csv"

    try:
        # Key change: index=True to save the DataFrame's index
        train_df.to_csv(train_file, index=True)
        val_df.to_csv(val_file, index=True)
        test_df.to_csv(test_file, index=True)

        print("\nData successfully split and saved with index:")
        print(f"- '{train_file}' created with {len(train_df)} rows.")
        print(f"- '{val_file}' created with {len(val_df)} rows.")
        print(f"- '{test_file}' created with {len(test_df)} rows.")

        # Print date ranges for verification
        # This assumes 'date_column_for_range_info' column exists and contains datetime-like objects
        # If your date information is solely in the index (and it's a DatetimeIndex):
        # You would use: train_df.index.min(), train_df.index.max(), etc.
        print(f"\nDate ranges for each set (using column: '{date_column_for_range_info}'):")
        if date_column_for_range_info in train_df.columns:
            print(f"  Train: {pd.to_datetime(train_df[date_column_for_range_info]).min().strftime('%Y-%m-%d')} to {pd.to_datetime(train_df[date_column_for_range_info]).max().strftime('%Y-%m-%d')}")
            print(f"  Val:   {pd.to_datetime(val_df[date_column_for_range_info]).min().strftime('%Y-%m-%d')} to {pd.to_datetime(val_df[date_column_for_range_info]).max().strftime('%Y-%m-%d')}")
            print(f"  Test:  {pd.to_datetime(test_df[date_column_for_range_info]).min().strftime('%Y-%m-%d')} to {pd.to_datetime(test_df[date_column_for_range_info]).max().strftime('%Y-%m-%d')}")
        elif isinstance(train_df.index, pd.DatetimeIndex):
             print(f"  Train: {train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')} (from index)")
             print(f"  Val:   {val_df.index.min().strftime('%Y-%m-%d')} to {val_df.index.max().strftime('%Y-%m-%d')} (from index)")
             print(f"  Test:  {test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')} (from index)")
        else:
            print(f"  Warning: Column '{date_column_for_range_info}' not found and index is not DatetimeIndex. Cannot print date ranges.")

        return True

    except Exception as e:
        print(f"An error occurred while saving files: {e}")
        return False

if __name__ == '__main__':
    # --- Example Usage ---
    # Create a dummy DataFrame (replace with your actual data loading)
    data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                 '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10']),
        'Price': [100, 101, 102, 103, 102, 104, 105, 106, 107, 108],
        'Volume': [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1350, 1400, 1380]
    }
    sample_df = pd.DataFrame(data)

    # IMPORTANT: Ensure your DataFrame is sorted by date before splitting
    sample_df = sample_df.sort_values(by='Date').reset_index(drop=True) # Reset index if sorting changes it

    # Set a meaningful index (e.g., the 'Date' column itself or a numerical representation)
    # Option 1: Use Date as DatetimeIndex
    sample_df.set_index('Date', inplace=True)
    # Option 2: Create and use a numerical index (if you did this in a previous EDA step)
    # sample_df['numeric_date_idx'] = sample_df['Date'].astype(np.int64) // 10**9
    # sample_df.set_index('numeric_date_idx', inplace=True)
    # If using Option 2, you might still want 'Date' as a regular column for range printing,
    # or adjust date_column_for_range_info in the function call.

    print("--- Original DataFrame with Date Index ---")
    print(sample_df.head())
    print(f"Index name: {sample_df.index.name}")


    # Define ratios
    train_p = 0.7
    val_p = 0.15
    # Test ratio will be 1.0 - train_p - val_p = 0.15

    # Specify the output directory
    output_directory = "data_timeseries_split" # Will be created if it doesn't exist

    # Call the function
    # If 'Date' is now the index, the original 'Date' column is gone.
    # The function will try to use the index for date range printing if it's a DatetimeIndex.
    # If you kept 'Date' as a regular column AND set a different index,
    # you could pass date_column_for_range_info="Date".
    # Since 'Date' is the index here, we don't need to specify date_column_for_range_info,
    # as the function will check if the index is a DatetimeIndex.
    success = split_and_save_timeseries_data(
        df=sample_df,
        train_ratio=train_p,
        val_ratio=val_p,
        output_dir=output_directory,
        output_prefix="stock_data_cleaned"
        # date_column_for_range_info="Date" # Not needed if 'Date' is the index
                                            # and index is pd.DatetimeIndex
    )

    if success:
        print(f"\nSplitting successful. Files saved in '{output_directory}' directory.")
        # You can now load them back to verify the index:
        # loaded_train_df = pd.read_csv(Path(output_directory) / "stock_data_cleaned_train.csv",
        # index_col=sample_df.index.name) # Use the original index name
        # print("\n--- Loaded Train DF (first 5 rows) ---")
        # print(loaded_train_df.head())
    else:
        print("\nSplitting failed.")
