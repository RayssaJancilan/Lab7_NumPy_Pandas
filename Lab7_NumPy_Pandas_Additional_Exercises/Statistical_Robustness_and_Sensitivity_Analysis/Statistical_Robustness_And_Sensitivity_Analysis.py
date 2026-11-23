import numpy as np
import pandas as pd


STUDENT_ID = "TUPM-26-3228"
SEED_DIGIT = 8

FEATURE_COUNT = 5
BOOTSTRAP_REPS = 200 + (10 * SEED_DIGIT)  # 280
RANDOM_SEED = SEED_DIGIT

MEASUREMENT_FILE = r"C:\Users\Rayssa Jancilan\Lab7_NumPy_Pandas_Jancilan\Lab7_NumPy_Pandas_Additional_Exercises\data\secom.data"
LABEL_FILE = r"C:\Users\Rayssa Jancilan\Lab7_NumPy_Pandas_Jancilan\Lab7_NumPy_Pandas_Additional_Exercises\data\secom_labels.data"

CSV_OUTPUT = "robustness_comparison.csv"
REPORT_OUTPUT = "sensitivity_report.txt"

print("=" * 70)
print("STATISTICAL ROBUSTNESS AND SENSITIVITY ANALYSIS")
print("=" * 70)
print("Student ID:", STUDENT_ID)
print("Seed digit:", SEED_DIGIT)
print("Selected features:", FEATURE_COUNT)
print("Bootstrap repetitions:", BOOTSTRAP_REPS)


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
    0, "Record_ID", np.arange(len(measurements))
)

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
    0, "Record_ID", np.arange(len(labels))
)

if len(measurements) != len(labels):
    raise ValueError("Measurement and label counts do not match!")

data = pd.merge(
    measurements,
    labels,
    on="Record_ID",
    how="inner",
    validate="one_to_one"
)

pass_data = data[data["Outcome"] == -1].copy()
fail_data = data[data["Outcome"] == 1].copy()

print("\nDataset loaded successfully.")
print("Total records:", len(data))
print("Pass records:", len(pass_data))
print("Fail records:", len(fail_data))


measurement_columns = [
    col for col in data.columns
    if col.startswith("Feature_")
]

usable_features = []

for feature in measurement_columns:

    pass_observed = pass_data[feature].notna().sum()
    fail_observed = fail_data[feature].notna().sum()

    if data[feature].nunique(dropna=True) <= 1:
        continue

    if pass_observed >= 10 and fail_observed >= 10:
        usable_features.append(feature)

if len(usable_features) < FEATURE_COUNT:
    raise ValueError(
        "Not enough usable features to select five."
    )



selection_rng = np.random.default_rng(RANDOM_SEED)

selected_features = selection_rng.choice(
    usable_features,
    size=FEATURE_COUNT,
    replace=False
).tolist()

print("\nSELECTED FEATURES")
print("-" * 70)

for feature in selected_features:
    print(feature)


available_pass = pass_data[selected_features].copy()
available_fail = fail_data[selected_features].copy()

full_medians = data[selected_features].median()

imputed_pass = pass_data[selected_features].fillna(full_medians)
imputed_fail = fail_data[selected_features].fillna(full_medians)



def calculate_mean_differences(pass_df, fail_df):
    """
    Calculates Fail mean - Pass mean for each feature.
    """

    pass_means = pass_df.mean()
    fail_means = fail_df.mean()

    return fail_means - pass_means


available_difference = calculate_mean_differences(
    available_pass,
    available_fail
)

imputed_difference = calculate_mean_differences(
    imputed_pass,
    imputed_fail
)


def bootstrap_mean_difference(
    pass_df,
    fail_df,
    repetitions,
    seed
):
    """
    Resample Pass and Fail groups independently with replacement.

    Each feature is bootstrapped using its valid values.
    Returns:
        Bootstrap mean differences
        95% percentile confidence intervals
    """

    rng = np.random.default_rng(seed)

    bootstrap_results = {
        feature: []
        for feature in selected_features
    }

    for _ in range(repetitions):

        for feature in selected_features:

            pass_values = pass_df[feature].dropna().to_numpy()
            fail_values = fail_df[feature].dropna().to_numpy()

            if len(pass_values) == 0 or len(fail_values) == 0:
                bootstrap_results[feature].append(np.nan)
                continue

            boot_pass = rng.choice(
                pass_values,
                size=len(pass_values),
                replace=True
            )

            boot_fail = rng.choice(
                fail_values,
                size=len(fail_values),
                replace=True
            )

            difference = np.mean(boot_fail) - np.mean(boot_pass)

            bootstrap_results[feature].append(difference)

    confidence_intervals = {}

    for feature in selected_features:

        values = np.array(
            bootstrap_results[feature],
            dtype=float
        )

        values = values[~np.isnan(values)]

        if len(values) == 0:
            confidence_intervals[feature] = (np.nan, np.nan)
        else:
            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)

            confidence_intervals[feature] = (lower, upper)

    return bootstrap_results, confidence_intervals


available_bootstrap, available_ci = bootstrap_mean_difference(
    available_pass,
    available_fail,
    repetitions=BOOTSTRAP_REPS,
    seed=RANDOM_SEED
)

imputed_bootstrap, imputed_ci = bootstrap_mean_difference(
    imputed_pass,
    imputed_fail,
    repetitions=BOOTSTRAP_REPS,
    seed=RANDOM_SEED
)


comparison = pd.DataFrame({
    "Feature": selected_features,

    "Available_Case_Mean_Difference":
        available_difference.values,

    "Available_Case_CI_Lower": [
        available_ci[f][0] for f in selected_features
    ],

    "Available_Case_CI_Upper": [
        available_ci[f][1] for f in selected_features
    ],

    "Median_Imputed_Mean_Difference":
        imputed_difference.values,

    "Median_Imputed_CI_Lower": [
        imputed_ci[f][0] for f in selected_features
    ],

    "Median_Imputed_CI_Upper": [
        imputed_ci[f][1] for f in selected_features
    ]
})


comparison["Absolute_Change_in_Difference"] = np.abs(
    comparison["Median_Imputed_Mean_Difference"]
    - comparison["Available_Case_Mean_Difference"]
)

comparison["Direction_Changed"] = (
    np.sign(comparison["Available_Case_Mean_Difference"])
    != np.sign(comparison["Median_Imputed_Mean_Difference"])
)

comparison["Available_Case_Rank"] = (
    comparison["Available_Case_Mean_Difference"]
    .abs()
    .rank(ascending=False, method="min")
)

comparison["Median_Imputed_Rank"] = (
    comparison["Median_Imputed_Mean_Difference"]
    .abs()
    .rank(ascending=False, method="min")
)

comparison["Ranking_Changed"] = (
    comparison["Available_Case_Rank"]
    != comparison["Median_Imputed_Rank"]
)

comparison = comparison.sort_values(
    by="Median_Imputed_Rank"
).reset_index(drop=True)

available_bootstrap_repeat, available_ci_repeat = (
    bootstrap_mean_difference(
        available_pass,
        available_fail,
        repetitions=BOOTSTRAP_REPS,
        seed=RANDOM_SEED
    )
)

imputed_bootstrap_repeat, imputed_ci_repeat = (
    bootstrap_mean_difference(
        imputed_pass,
        imputed_fail,
        repetitions=BOOTSTRAP_REPS,
        seed=RANDOM_SEED
    )
)

reproducibility_verified = True

for feature in selected_features:

    first_available = np.array(
        available_bootstrap[feature]
    )

    second_available = np.array(
        available_bootstrap_repeat[feature]
    )

    first_imputed = np.array(
        imputed_bootstrap[feature]
    )

    second_imputed = np.array(
        imputed_bootstrap_repeat[feature]
    )

    if not np.allclose(
        first_available,
        second_available,
        equal_nan=True
    ):
        reproducibility_verified = False

    if not np.allclose(
        first_imputed,
        second_imputed,
        equal_nan=True
    ):
        reproducibility_verified = False


comparison.to_csv(
    CSV_OUTPUT,
    index=False
)

largest_change_row = comparison.loc[
    comparison["Absolute_Change_in_Difference"].idxmax()
]

largest_change_feature = largest_change_row["Feature"]

largest_change_value = largest_change_row[
    "Absolute_Change_in_Difference"
]

changed_sign_features = comparison.loc[
    comparison["Direction_Changed"],
    "Feature"
].tolist()

changed_ranking_features = comparison.loc[
    comparison["Ranking_Changed"],
    "Feature"
].tolist()

available_ci_cross_zero = []
imputed_ci_cross_zero = []

for feature in selected_features:

    a_lower, a_upper = available_ci[feature]
    i_lower, i_upper = imputed_ci[feature]

    if a_lower <= 0 <= a_upper:
        available_ci_cross_zero.append(feature)

    if i_lower <= 0 <= i_upper:
        imputed_ci_cross_zero.append(feature)


report_lines = [
    "STATISTICAL ROBUSTNESS AND SENSITIVITY ANALYSIS",
    "=" * 60,
    f"Student ID: {STUDENT_ID}",
    f"Seed digit: {SEED_DIGIT}",
    f"Selected features: {FEATURE_COUNT}",
    f"Bootstrap repetitions: {BOOTSTRAP_REPS}",
    "",
    "Selected Features:",
    ", ".join(selected_features),
    "",
    f"Largest change in mean difference: {largest_change_value:.6f}",
    f"Feature with largest change: {largest_change_feature}",
    "",
    "Features with changed difference sign:",
    str(changed_sign_features),
    "",
    "Features with changed ranking:",
    str(changed_ranking_features),
    "",
    "Available-case confidence intervals crossing zero:",
    str(available_ci_cross_zero),
    "",
    "Median-imputed confidence intervals crossing zero:",
    str(imputed_ci_cross_zero),
    "",
    f"Reproducibility verified: {reproducibility_verified}",
    "",
    "Feature Comparison Table:",
    comparison.to_string(index=False)
]

with open(REPORT_OUTPUT, "w", encoding="utf-8") as file:
    file.write("\n".join(report_lines))


print("\n" + "=" * 70)
print("FINAL SENSITIVITY ANALYSIS")
print("=" * 70)

print("\nSelected Features:")
print(selected_features)

print("\nBootstrap Repetitions:")
print(BOOTSTRAP_REPS)

print("\nLargest Change in Mean Difference:")
print(f"{largest_change_feature}: {largest_change_value:.6f}")

print("\nFeatures with Changed Difference Sign:")
print(changed_sign_features)

print("\nFeatures with Changed Ranking:")
print(changed_ranking_features)

print("\nReproducibility Verified:")
print(reproducibility_verified)

print("\nComparison Table:")
print(comparison.to_string(index=False))

print("\nOutput Files:")
print(CSV_OUTPUT)
print(REPORT_OUTPUT)

print("\nPROGRAM COMPLETED SUCCESSFULLY")
print("=" * 70)

