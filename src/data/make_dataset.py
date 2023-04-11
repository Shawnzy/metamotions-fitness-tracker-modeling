import pandas as pd
from glob import glob


# --------------------------------------------------------------
# Load raw data and create accelerometer and gyroscope dataframes
# --------------------------------------------------------------

# Get path and list of filenames
data_path = "../../data/raw/MetaMotion/"
files = glob("../../data/raw/MetaMotion/*.csv")


def read_data_from_files(files):
    # Create empty dataframes
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    # Initialize set counter
    acc_set = 1
    gyr_set = 1

    for f in files:
        # Extract features from filename
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].split("_")[0].rstrip("123")

        # Open file and read data
        df = pd.read_csv(f)

        # Create 3 new columns
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        # Concatenate dataframes based on if it is accelerometer or gyroscope data
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    # Convert "epoch (ms)" column to datetime and set it as index
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    # Delete unnecessary columns
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    # Return accelerometer and gyroscope dataframes
    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merge the 2 datasets
# --------------------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

# List of functions to apply to each column
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "label": "last",
    "category": "last",
    "participant": "last",
    "set": "last",
}

# Group data by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

# Resample data and drop NaN values for each day
data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

# Format "set" column
data_resampled["set"] = data_resampled["set"].astype("int")

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
