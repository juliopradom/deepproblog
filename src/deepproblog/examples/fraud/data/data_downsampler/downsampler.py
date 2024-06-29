import pandas as pd
import numpy as np


def build_dataset(df, df_standard, multiplier, out_name, out_name_standard):

    # Main file
    df_pos = df[df["is_fraud"] == 1]
    # Generate a random set of indices
    np.random.seed(42)  # Set a random seed for reproducibility
    subset_indices_pos = np.random.choice(df_pos.index, size=len(df_pos), replace=False)
    df_pos = df_pos.loc[subset_indices_pos]
    df_pos_standard = df_standard[df_standard["is_fraud"] == 1]
    df_pos_standard = df_pos_standard.loc[subset_indices_pos]
    df_neg = df[df["is_fraud"] == 0]
    subset_indices_neg = np.random.choice(
        df_neg.index, size=int(len(df_neg) * multiplier), replace=False
    )
    df_neg = df_neg.loc[subset_indices_neg]
    df_neg_standard = df_standard[df_standard["is_fraud"] == 0]
    df_neg_standard = df_neg_standard.loc[subset_indices_neg]
    df = pd.concat([df_pos, df_neg]).reset_index()
    df_standard = pd.concat([df_pos_standard, df_neg_standard]).reset_index()
    subset_indices_final = np.random.choice(df.index, size=len(df), replace=False)
    df = df.loc[subset_indices_final]
    df_standard = df_standard.loc[subset_indices_final]
    print(f"Total length scaled train: {len(df)} -> {len(df_pos)}/{len(df_neg)}")
    df.to_csv(out_name, index=False)
    print(
        f"Total length standard train: {len(df_standard)} -> {len(df_pos_standard)}/{len(df_neg_standard)}"
    )
    df_standard.to_csv(out_name_standard, index=False)

    # 10% version
    df_10 = df.sample(frac=0.1, random_state=42)
    df_10.to_csv(f"{out_name.split('.csv')[0]}10.csv")
    df_10_standard = df_standard.sample(frac=0.1, random_state=42)
    df_10_standard.to_csv(f"{out_name_standard.split('.csv')[0]}10.csv")

    # 1% version
    df_1 = df.sample(frac=0.01, random_state=42)
    df_1.to_csv(f"{out_name.split('.csv')[0]}1.csv")
    df_1_standard = df_standard.sample(frac=0.01, random_state=42)
    df_1_standard.to_csv(f"{out_name_standard.split('.csv')[0]}1.csv")


def build_original_test_dataset(df, df_standard, size, out_name, out_name_standard):

    # Generate a random set of indices
    np.random.seed(42)  # Set a random seed for reproducibility
    subset_indices_final = np.random.choice(df.index, size=size, replace=False)
    df = df.loc[subset_indices_final]
    df_standard = df_standard.loc[subset_indices_final]
    print(
        f"Total length scaled test: {len(df)} -> {len(df[df['is_fraud'] == 1])}/{len(df[df['is_fraud'] == 0])}"
    )
    df.to_csv(out_name, index=False)
    print(
        f"Total length standard test: {len(df_standard)} -> {len(df_standard[df_standard['is_fraud'] == 1])}/{len(df_standard[df_standard['is_fraud'] == 0])}"
    )
    df_standard.to_csv(out_name_standard, index=False)


def build_test_dataset(df, df_standard, multiplier, out_name, out_name_standard):

    # Generate a random set of indices
    df_pos = df[df["is_fraud"] == 1]
    df_pos_standard = df_standard[df_standard["is_fraud"] == 1]
    # Generate a random set of indices
    np.random.seed(42)  # Set a random seed for reproducibility
    subset_indices_pos = np.random.choice(df_pos.index, size=len(df_pos), replace=False)
    df_pos = df_pos.loc[subset_indices_pos]
    df_pos_standard = df_pos_standard.loc[subset_indices_pos]

    df_neg = df[df["is_fraud"] == 0]
    subset_indices_neg = np.random.choice(
        df_neg.index, size=int(len(df_pos)) * multiplier, replace=False
    )
    df_neg = df_neg.loc[subset_indices_neg]
    df_neg_standard = df_standard[df_standard["is_fraud"] == 0]
    df_neg_standard = df_neg_standard.loc[subset_indices_neg]
    df = pd.concat([df_pos, df_neg]).reset_index()
    df_standard = pd.concat([df_pos_standard, df_neg_standard]).reset_index()

    subset_indices_final = np.random.choice(df.index, size=len(df), replace=False)
    df = df.loc[subset_indices_final]
    df_standard = df_standard.loc[subset_indices_final]

    print(
        f"Total length scaled test: {len(df)} -> {len(df[df['is_fraud'] == 1])}/{len(df[df['is_fraud'] == 0])}"
    )
    df.to_csv(out_name, index=False)
    print(
        f"Total length standard test: {len(df_standard)} -> {len(df_standard[df_standard['is_fraud'] == 1])}/{len(df_standard[df_standard['is_fraud'] == 0])}"
    )
    df_standard.to_csv(out_name_standard, index=False)


def training_datasets_generator():

    df = pd.read_csv("../preprocessed/fraudTrainPreprocessedScaled.csv")
    df_standard = pd.read_csv("../preprocessed/fraudTrainPreprocessed.csv")

    build_dataset(
        df,
        df_standard,
        0.058,
        "fraudTrainPreprocessedScaledDownsampled.csv",
        "fraudTrainPreprocessedDownsampled.csv",
    )
    build_dataset(
        df,
        df_standard,
        0.0058,
        "fraudTrainPreprocessedScaledBalanced.csv",
        "fraudTrainPreprocessedBalanced.csv",
    )


def test_datasets_generator():

    df = pd.read_csv("../preprocessed/fraudTestPreprocessedScaled.csv")
    df_standard = pd.read_csv("../preprocessed/fraudTestPreprocessed.csv")

    build_original_test_dataset(
        df,
        df_standard,
        40000,
        "fraudTestPreprocessedScaledOriginal.csv",
        "fraudTestPreprocessedOriginal.csv",
    )
    build_test_dataset(
        df,
        df_standard,
        10,
        "fraudTestPreprocessedScaledDownsampled.csv",
        "fraudTestPreprocessedDownsampled.csv",
    )
    build_test_dataset(
        df,
        df_standard,
        1,
        "fraudTestPreprocessedScaledBalanced.csv",
        "fraudTestPreprocessedBalanced.csv",
    )


if __name__ == "__main__":
    training_datasets_generator()
    test_datasets_generator()
