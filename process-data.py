import pandas as pd
import numpy as np
import os # Import os for path manipulation

# --- Configuration ---
# Define the directory containing the input file
data_directory = 'data'
input_filename = 'math-500.parquet'
output_filename = 'math-500_reindexed.parquet' # Use a different name for the output

# Construct full paths
input_parquet_path = os.path.join(data_directory, input_filename)
output_parquet_path = os.path.join(data_directory, output_filename)

# --- Processing ---
try:
    # Create the data directory if it doesn't exist (for example purposes)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Created directory: {data_directory}")
        # Create a dummy input file if it doesn't exist for the code to run
        print(f"Creating a dummy input file: {input_parquet_path}")
        dummy_df = pd.DataFrame({
            'col_a': np.random.rand(500),
            'col_b': np.random.randint(0, 100, 500),
            'original_index_col': range(1000, 1500) # Example column
        })
        # Set a non-sequential index to demonstrate the change
        dummy_df.index = pd.Index(range(0, 1000, 2), name='old_index')
        dummy_df.to_parquet(input_parquet_path)


    # Read the Parquet file into a pandas DataFrame
    # We don't need the original index, so we can reset it immediately if needed,
    # but setting df.index directly below overwrites it anyway.
    print(f"Reading Parquet file from: {input_parquet_path}")
    df = pd.read_parquet(input_parquet_path)
    print("Original DataFrame info:")
    df.info()
    print("\nOriginal first 5 rows:")
    print(df.head())


    # Get the number of rows in the DataFrame
    num_rows = len(df)
    print(f"\nDataFrame has {num_rows} rows.")

    # Create a new sequential index starting from 1 up to the number of rows
    # Name the new index 'extra_info' as requested
    print("Generating new sequential index named 'extra_info' from 1...")
    new_index = pd.RangeIndex(start=1, stop=num_rows + 1, step=1, name='extra_info')

    # Set the new index for the DataFrame, replacing the old one
    df.index = new_index
    print("New index assigned.")

    # Write the modified DataFrame back to a new Parquet file
    # index=True ensures the new index ('extra_info') is written to the file
    print(f"Writing modified DataFrame to: {output_parquet_path}")
    df.to_parquet(output_parquet_path, index=True)

    print("\n--- Success ---")
    print(f"Successfully processed '{input_parquet_path}'.")
    print(f"Created new index named 'extra_info' from 1 to {num_rows}.")
    print(f"Output saved to '{output_parquet_path}'.")

    # Display the first few rows with the new index to verify
    print("\nFirst 5 rows of the modified DataFrame:")
    print(df.head())
    print("\nModified DataFrame info:")
    df.info()


except FileNotFoundError:
    print(f"Error: Input file not found at '{input_parquet_path}'. Please ensure the file exists.")
except Exception as e:
    # Catch any other potential errors during processing
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
