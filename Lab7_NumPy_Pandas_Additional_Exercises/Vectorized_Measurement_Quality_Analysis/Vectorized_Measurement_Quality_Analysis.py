import numpy as np
import pandas as pd


STUDENT_ID = "TUPM-26-3228"
SEED_DIGIT = 8

FEATURE_COUNT = 5 + (SEED_DIGIT * 3)

DATASET_FILE = r"C:\Users\Rayssa Jancilan\Lab7_NumPy_Pandas_Jancilan\Lab7_NumPy_Pandas_Additional_Exercises\data\secom.data"
OUTPUT_FILE = "measurement_quality.csv"

ROBUST_Z_THRESHOLD = 3.5


rng = np.random.default_rng(SEED_DIGIT)


print("=" * 65)
print("VECTORIZED MEASUREMENT QUALITY ANALYSIS")
print("=" * 65)

df = pd.read_csv(
    DATASET_FILE,
    sep=r"\s+",
    header=None,
    na_values=["?", "NaN", "nan"]
)

df.columns = [f"Feature_{i+1}" for i in range(df.shape[1])]

df = df.apply(pd.to_numeric, errors="coerce")

print("\nDataset successfully loaded.")
print("Number of records:", df.shape[0])
print("Number of features:", df.shape[1])


features_with_missing = df.columns[df.isna().any()].tolist()

constant_features = [
    col for col in df.columns
    if df[col].nunique(dropna=True) <= 1
]

all_missing_features = [
    col for col in df.columns
    if df[col].isna().all()
]

unusable_features = list(set(constant_features + all_missing_features))

usable_features = [
    col for col in df.columns
    if col not in unusable_features
]

print("\nFEATURE QUALITY CHECK")
print("-" * 65)
print("Features containing missing values:", len(features_with_missing))
print("Constant features:", len(constant_features))
print("All-missing features:", len(all_missing_features))
print("Usable features:", len(usable_features))


if len(usable_features) < FEATURE_COUNT:
    raise ValueError(
        f"Not enough usable features. Required: {FEATURE_COUNT}, "
        f"Available: {len(usable_features)}"
    )

selected_features = rng.choice(
    usable_features,
    size=FEATURE_COUNT,
    replace=False
)

selected_features = list(selected_features)

selected_data = df[selected_features]

data = selected_data.to_numpy(dtype=float)

print("\nSELECTED FEATURES")
print("-" * 65)
print("Number of selected features:", len(selected_features))

for feature in selected_features:
    print(feature)


missing_percent = np.mean(np.isnan(data), axis=0) * 100

feature_means = np.nanmean(data, axis=0)

feature_medians = np.nanmedian(data, axis=0)

feature_std = np.nanstd(data, axis=0, ddof=1)


deviation = np.abs(data - feature_medians)

mad = np.nanmedian(deviation, axis=0)

safe_mad = np.where(mad == 0, np.nan, mad)


robust_z = (
    0.6745 * (data - feature_medians)
    / safe_mad
)

robust_z = np.nan_to_num(
    robust_z,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


outlier_mask = np.abs(robust_z) > ROBUST_Z_THRESHOLD

flagged_counts = np.sum(outlier_mask, axis=0)

valid_counts = np.sum(~np.isnan(data), axis=0)

flagged_percent = np.divide(
    flagged_counts,
    valid_counts,
    out=np.zeros_like(flagged_counts, dtype=float),
    where=valid_counts != 0
) * 100

total_flagged = np.sum(outlier_mask)


summary = pd.DataFrame({
    "Feature": selected_features,
    "Mean": feature_means,
    "Median": feature_medians,
    "Sample_Std": feature_std,
    "Missing_Percent": missing_percent,
    "MAD": mad,
    "Flagged_Count": flagged_counts,
    "Valid_Count": valid_counts,
    "Flagged_Percent": flagged_percent
})

summary = summary.sort_values(
    by="Flagged_Percent",
    ascending=False
).reset_index(drop=True)

summary.insert(
    0,
    "Rank",
    np.arange(1, len(summary) + 1)
)


verified_counts = []

for i in range(data.shape[1]):
    count = 0

    for j in range(data.shape[0]):
        if abs(robust_z[j, i]) > ROBUST_Z_THRESHOLD:
            count += 1

    verified_counts.append(count)

verified_counts = np.array(verified_counts)

vectorized_verified = np.array_equal(
    flagged_counts,
    verified_counts
)

print("\nVERIFICATION")
print("-" * 65)
print("Vectorized flagged counts:", flagged_counts)
print("Loop-verified flagged counts:", verified_counts)
print("Results match:", vectorized_verified)


summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nCSV successfully generated:", OUTPUT_FILE)


highest_missing_index = np.argmax(missing_percent)

highest_missing_feature = selected_features[
    highest_missing_index
]

highest_missing_value = missing_percent[
    highest_missing_index
]

highest_flagged_index = np.argmax(flagged_percent)

highest_flagged_feature = selected_features[
    highest_flagged_index
]

highest_flagged_value = flagged_percent[
    highest_flagged_index
]


print("\n" + "=" * 65)
print("FINAL MEASUREMENT QUALITY REPORT")
print("=" * 65)

print("\nStudent ID:", STUDENT_ID)
print("SEED_DIGIT:", SEED_DIGIT)
print("Selected Feature Count:", FEATURE_COUNT)

print("\nSelected Features:")
print(selected_features)

print("\nHighest Missing Percentage:")
print(f"{highest_missing_feature}: {highest_missing_value:.2f}%")

print("\nFeature with Highest Flagged Percentage:")
print(
    f"{highest_flagged_feature}: "
    f"{highest_flagged_value:.2f}%"
)

print("\nTotal Flagged Observations:")
print(total_flagged)

print("\nVectorized Results Verified:")
print(vectorized_verified)

print("\nTop 10 Features by Flagged Percentage:")
print(summary.head(10).to_string(index=False))

print("\nFull Summary:")
print(summary.to_string(index=False))

print("\nOutput file:", OUTPUT_FILE)

print("\nPROGRAM COMPLETED SUCCESSFULLY")
print("=" * 65)

