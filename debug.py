import pandas as pd
import os
import glob

data_dir = '/cluster/scratch/nbaruffol/airborne_dataset'
csv_path = f"{data_dir}/part1/ImageSets/groundtruth.csv"

# Load the CSV
df = pd.read_csv(csv_path)
print("CSV Columns:", df.columns.tolist())

# Pick a folder that we KNOW is on your hard drive
folders = glob.glob(f"{data_dir}/part1/Images/*")
test_flight = os.path.basename(folders[0])
print(f"\nTesting flight: {test_flight}")

# Grab the first row from the CSV for this flight
flight_df = df[df['flight_id'] == test_flight]
if len(flight_df) > 0:
    first_row = flight_df.iloc[0]
    print("\nFirst row data for this flight:")
    print(first_row)
    
    # Let's see what is actually physically on the disk in that folder
    actual_files = glob.glob(f"{data_dir}/part1/Images/{test_flight}/*.jpg")
    print(f"\nFiles found on disk in this folder: {len(actual_files)}")
    if len(actual_files) > 0:
        print(f"Example file on disk: {actual_files[0]}")