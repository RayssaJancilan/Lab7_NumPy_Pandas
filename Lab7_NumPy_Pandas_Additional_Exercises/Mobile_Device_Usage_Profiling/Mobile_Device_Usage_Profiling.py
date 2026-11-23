import numpy as np
import pandas as pd


STUDENT_ID = "TUPM-26-3228"
SEED_DIGIT = 8

ID_SUM = 23

SAMPLE_PER_GROUP = 20 + ID_SUM

FEATURE_COUNT = 5

RANDOM_SEED = SEED_DIGIT

MEASUREMENT_FILE = r"C:\Users\Rayssa Jancilan\Lab7_NumPy_Pandas_Jancilan\Lab7_NumPy_Pandas_Additional_Exercises\data\secom.data"
LABEL_FILE = r"C:\Users\Rayssa Jancilan\Lab7_NumPy_Pandas_Jancilan\Lab7_NumPy_Pandas_Additional_Exercises\data\secom_labels.data"

BALANCED_OUTPUT = "balanced_sample.csv"
COMPARISON_OUTPUT = "feature_comparison.csv"

rng = np.random.default_rng(RANDOM_SEED)

print("=" * 70)
print("MANUFACTURING OUTCOME COMPARISON")
print("=" * 70)

print("Student ID:", STUDENT_ID)
print("ID_SUM:", ID_SUM)
print("Seed digit:", SEED_DIGIT)
print("Sample size per group:", SAMPLE_PER_GROUP)


measurements = pd.read_csv(
    MEASUREMENT_FILE,
    sep=r"\s+",
    header=None,
    na_values=["?", "NaN", "nan"]
)

measurements.columns = [
    f"Feature_{i+1}" for i in range(measurements.shape[1])
]

measurements = measurements.apply(
    pd.to_numeric,
    errors="coerce"
)

measurements.insert(
    0,
    "Record_ID",
    np.arange(len(measurements))
)

print("\nMeasurement dataset loaded.")
print("Records:", len(measurements))
print("Measurement features:", measurements.shape[1] - 1)


labels = pd.read_csv(
    LABEL_FILE,
    sep=r"\s+",
    header=None
)

labels.columns = ["Outcome", "Timestamp"]

labels["Outcome"] = pd.to_numeric(
    labels["Outcome"],
    errors="coerce"
)


labels.insert(
    0,
    "Record_ID",
    np.arange(len(labels))
)

if len(measurements) != len(labels):
    raise ValueError(
        "Measurement and label record counts do not match!"
    )

data = pd.merge(
    measurements,
    labels,
    on="Record_ID",
    how="inner",
    validate="one_to_one"
)

print("\nLabeled DataFrame created.")
print("Total records:", len(data))


pass_records = data[data["Outcome"] == -1].copy()
fail_records = data[data["Outcome"] == 1].copy()

pass_count = len(pass_records)
fail_count = len(fail_records)

print("\nOUTCOME COUNTS")
print("-" * 70)
print("Passing records:", pass_count)
print("Failing records:", fail_count)

balanced_size = min(
    SAMPLE_PER_GROUP,
    pass_count,
    fail_count
)

if balanced_size == 0:
    raise ValueError(
        "One outcome group has no records."
        "Cannot create a balanced sample."
    )

print("Balanced sample per group:", balanced_size)
print("Total balanced sample:", balanced_size * 2)


balanced_pass = pass_records.sample(
    n=balanced_size,
    random_state=RANDOM_SEED
)

balanced_fail = fail_records.sample(
    n=balanced_size,
    random_state=RANDOM_SEED
)

balanced_sample = pd.concat(
    [balanced_pass, balanced_fail],
    ignore_index=True
)

balanced_sample = balanced_sample.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

print("\nBALANCED SAMPLE")
print("-" * 70)
print("Passing samples:", len(balanced_pass))
print("Failing samples:", len(balanced_fail))
print("Total samples:", len(balanced_sample))


measurement_columns = [
    col for col in data.columns
    if col.startswith("Feature_")
]

usable_features = [
    col for col in measurement_columns
    if data[col].notna().any()
    and data[col].nunique(dropna=True) > 1
]

if len(usable_features) < FEATURE_COUNT:
    raise ValueError(
        "Not enough usable features for selection."
    )


feature_rng = np.random.default_rng(RANDOM_SEED)

selected_features = feature_rng.choice(
    usable_features,
    size=FEATURE_COUNT,
    replace=False
).tolist()

print("\nSELECTED FEATURES")
print("-" * 70)

for feature in selected_features:
    print(feature)


full_dataset_medians = data[selected_features].median(
    skipna=True
)

balanced_sample[selected_features] = (
    balanced_sample[selected_features]
    .fillna(full_dataset_medians)
)

remaining_missing = (
    balanced_sample[selected_features]
    .isna()
    .sum()
    .sum()
)

print("\nMEDIAN IMPUTATION")
print("-" * 70)
print("Remaining missing measurements:", remaining_missing)


grouped = balanced_sample.groupby("Outcome")[selected_features]

group_statistics = grouped.agg(
    ["count", "mean", "median", "std"]
)

print("\nGROUPED STATISTICS")
print("-" * 70)
print(group_statistics)


pass_stats = group_statistics.loc[-1]
fail_stats = group_statistics.loc[1]

pass_means = pass_stats.xs("mean", level=1)
fail_means = fail_stats.xs("mean", level=1)

pass_medians = pass_stats.xs("median", level=1)
fail_medians = fail_stats.xs("median", level=1)

pass_std = pass_stats.xs("std", level=1)
fail_std = fail_stats.xs("std", level=1)

n_pass = len(balanced_sample[balanced_sample["Outcome"] == -1])
n_fail = len(balanced_sample[balanced_sample["Outcome"] == 1])

absolute_mean_difference = np.abs(
    fail_means - pass_means
)

pooled_variance = (
    ((n_pass - 1) * pass_std**2)
    + ((n_fail - 1) * fail_std**2)
) / (n_pass + n_fail - 2)

pooled_std = np.sqrt(pooled_variance)

standardized_mean_difference = np.divide(
    absolute_mean_difference,
    pooled_std,
    out=np.zeros(len(selected_features), dtype=float),
    where=pooled_std != 0
)


comparison = pd.DataFrame({
    "Feature": selected_features,
    "Pass_Count": n_pass,
    "Fail_Count": n_fail,
    "Pass_Mean": pass_means.values,
    "Fail_Mean": fail_means.values,
    "Pass_Median": pass_medians.values,
    "Fail_Median": fail_medians.values,
    "Pass_Std": pass_std.values,
    "Fail_Std": fail_std.values,
    "Absolute_Mean_Difference": absolute_mean_difference.values,
    "Pooled_Std": pooled_std.values,
    "Standardized_Mean_Difference": standardized_mean_difference
})

comparison = comparison.sort_values(
    by="Standardized_Mean_Difference",
    ascending=False
).reset_index(drop=True)

comparison.insert(
    0,
    "Rank",
    np.arange(1, len(comparison) + 1)
)

print("\nFEATURE COMPARISON")
print("-" * 70)
print(comparison.to_string(index=False))


verification_results = []

for outcome in [-1, 1]:

    group_data = balanced_sample[
        balanced_sample["Outcome"] == outcome
    ][selected_features]

    pandas_means = group_data.mean().to_numpy()

    numpy_means = np.nanmean(
        group_data.to_numpy(),
        axis=0
    )

    pandas_medians = group_data.median().to_numpy()

    numpy_medians = np.nanmedian(
        group_data.to_numpy(),
        axis=0
    )

    pandas_std = group_data.std(ddof=1).to_numpy()

    numpy_std = np.nanstd(
        group_data.to_numpy(),
        axis=0,
        ddof=1
    )

    means_match = np.allclose(
        pandas_means,
        numpy_means,
        equal_nan=True
    )

    medians_match = np.allclose(
        pandas_medians,
        numpy_medians,
        equal_nan=True
    )

    std_match = np.allclose(
        pandas_std,
        numpy_std,
        equal_nan=True
    )

    verification_results.append(
        means_match and medians_match and std_match
    )

    print(f"\nOutcome {outcome} Verification")
    print("Means match:", means_match)
    print("Medians match:", medians_match)
    print("Standard deviations match:", std_match)

numpy_pandas_match = all(verification_results)


balanced_sample.to_csv(
    BALANCED_OUTPUT,
    index=False
)

comparison.to_csv(
    COMPARISON_OUTPUT,
    index=False
)

print("\nCSV FILES CREATED")
print("-" * 70)
print(BALANCED_OUTPUT)
print(COMPARISON_OUTPUT)


highest_ranked_feature = comparison.iloc[0]["Feature"]

highest_standardized_difference = comparison.iloc[0][
    "Standardized_Mean_Difference"
]

print("\n" + "=" * 70)
print("FINAL ASSESSMENT REPORT")
print("=" * 70)

print("Student ID:", STUDENT_ID)
print("ID_SUM:", ID_SUM)
print("Balanced Sample Size per Group:", balanced_size)
print("Total Balanced Sample Size:", len(balanced_sample))

print("\nSelected Features:")
print(selected_features)

print("\nHighest-Ranked Feature:")
print(highest_ranked_feature)

print("\nAbsolute Standardized Mean Difference:")
print(f"{highest_standardized_difference:.6f}")

print("\nNumPy-Pandas Statistics Match:")
print(numpy_pandas_match)

print("\nTop Ranked Feature Comparisons:")
print(comparison.head(5).to_string(index=False))

print("\nPROGRAM COMPLETED SUCCESSFULLY")
print("=" * 70)

