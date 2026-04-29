import pandas as pd
import numpy as np
import os # Import os for path manipulation


def build_extra_info(value: object, index: int) -> dict[str, object]:
    if isinstance(value, dict):
        extra_info = dict(value)
    else:
        extra_info = {}
    extra_info["index"] = index
    return extra_info

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


    # Read the Parquet file into a pandas DataFrame.
    print(f"Reading Parquet file from: {input_parquet_path}")
    df = pd.read_parquet(input_parquet_path)
    print("Original DataFrame info:")
    df.info()
    print("\nOriginal first 5 rows:")
    print(df.head())


    # Get the number of rows in the DataFrame
    num_rows = len(df)
    print(f"\nDataFrame has {num_rows} rows.")

    # RLHFDataset reads row_dict["extra_info"]["index"], so store the repeat
    # index inside the extra_info column rather than as a pandas index.
    print("Generating 0-based extra_info.index values...")
    if "extra_info" in df.columns:
        existing_extra_info = df["extra_info"].tolist()
    else:
        existing_extra_info = [None] * num_rows

    df["extra_info"] = [
        build_extra_info(value=value, index=index)
        for index, value in enumerate(existing_extra_info)
    ]
    df = df.reset_index(drop=True)
    print("extra_info.index assigned.")

    # Write the modified DataFrame back to a new Parquet file
    print(f"Writing modified DataFrame to: {output_parquet_path}")
    df.to_parquet(output_parquet_path, index=False)

    print("\n--- Success ---")
    print(f"Successfully processed '{input_parquet_path}'.")
    if num_rows:
        print(f"Created 0-based extra_info.index values from 0 to {num_rows - 1}.")
    else:
        print("Created empty extra_info.index values.")
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
