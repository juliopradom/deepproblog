import pandas as pd
import numpy as np


def create_matrix(probability, rows, columns):
    # Calculate the number of mixed rows
    num_mixed_rows = int(probability * rows)

    # Initialize the matrix with zeros
    matrix = np.zeros((rows, columns), dtype=int)

    # Randomly choose which rows will be mixed
    mixed_rows_indices = np.random.choice(rows, num_mixed_rows, replace=False)

    # Fill the mixed rows with random 1s
    for row_index in mixed_rows_indices:
        # Determine the number of 1s in this row
        num_ones = np.random.randint(1, columns + 1)
        # Randomly choose positions for the 1s in this row
        one_positions = np.random.choice(columns, num_ones, replace=False)
        # Assign 1s to the chosen positions
        matrix[row_index, one_positions] = 1

    return matrix


def add_missing_labels(df, df_standard, percentage):

    columns = list(df.columns)[1:]
    grid = create_matrix(percentage, df.shape[0], df.shape[1])

    index_label = columns.index("is_fraud")
    df["noisy"] = False
    for i in range(len(df)):
        has_noise = False
        for j in range(len(columns)):
            if grid[i][j] == 1 and j != index_label:
                has_noise = True
                if isinstance(df[columns[j]][i], np.int64):
                    df.loc[i, columns[j]] = -1
                elif isinstance(df[columns[j]][i], np.float64):
                    df.loc[i, columns[j]] = -1
                df_standard.loc[i, columns[j]] = 0.0
        if has_noise:
            df.loc[i, "noisy"] = True

    return df, df_standard


if __name__ == "__main__":

    df = pd.read_csv(
        "../data_downsampler/fraudTrainPreprocessedDownsampled.csv", index_col=False
    )
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df_standard = pd.read_csv(
        "../data_downsampler/fraudTrainPreprocessedScaledDownsampled.csv",
        index_col=False,
    )
    df_standard = df_standard.loc[:, ~df_standard.columns.str.contains("^Unnamed")]
    for i in range(1, 10):
        print(f"Creating training missing {i}0%")
        df_missing, df_standard_missing = add_missing_labels(
            df, df_standard, float(f"0.{i}")
        )
        df_missing.to_csv(f"fraudTrainPreprocessedDownsampledMissing{i}0.csv")
        df_standard_missing.to_csv(
            f"fraudTrainPreprocessedScaledDownsampledMissing{i}0.csv"
        )

    df = pd.read_csv(
        "../data_downsampler/fraudTestPreprocessedOriginal.csv", index_col=False
    )
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df_standard = pd.read_csv(
        "../data_downsampler/fraudTestPreprocessedScaledOriginal.csv", index_col=False
    )
    df_standard = df_standard.loc[:, ~df_standard.columns.str.contains("^Unnamed")]
    for i in range(1, 10):
        print(f"Creating test missing {i}0%")
        df_missing, df_standard_missing = add_missing_labels(
            df, df_standard, float(f"0.{i}")
        )
        df_missing.to_csv(f"fraudTestPreprocessedOriginalMissing{i}0.csv")
        df_standard_missing.to_csv(
            f"fraudTestPreprocessedScaledOriginalMissing{i}0.csv"
        )
