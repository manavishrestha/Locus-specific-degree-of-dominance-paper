# xgboost LOEO yield
# hyperopt 10.08.2026
# MAster CV created with outputs from this file.

from pathlib import Path
import time
import random

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope


###############################################################################
# Settings
###############################################################################

random.seed(42)
np.random.seed(42)

NUM_ITERATIONS = 5
N_GROUPS = 5
GROUP_SOURCE_COLUMN = "Field_Location"

MAX_EVALS = 20
MAX_N_ESTIMATORS = 8000
EARLY_STOPPING_ROUNDS = 200

DATA_PATH = (
    "/Users/manavi/Desktop/Manuscript_review/"
    "Datasets/Final_merged_March19.csv"
)

OUTPUT_FOLDER = Path(
    "/Users/manavi/Desktop/Manuscript_review/"
    "xgboost_optimized_leave_group_env_out_yield"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

total_start_time = time.time()


###############################################################################
# Load data
###############################################################################

Final_dataset = pd.read_csv(DATA_PATH)

print("Original dataset shape:", Final_dataset.shape)

Final_dataset = (
    Final_dataset
    .dropna(subset=["Average_BLUE"])
    .reset_index(drop=True)
)

print(
    "Dataset shape after removing missing responses:",
    Final_dataset.shape
)

X = Final_dataset.drop(
    columns=[
        "Year",
        "Treatment",
        "Hybrid",
        "Field_Location",
        "Env",
        "Average_BLUE"
    ]
)

y = Final_dataset["Average_BLUE"].copy()

non_numeric_columns = (
    X.select_dtypes(exclude=np.number)
    .columns
    .tolist()
)

if non_numeric_columns:
    raise TypeError(
        "These predictor columns are not numeric:\n"
        f"{non_numeric_columns}"
    )

X = X.astype(np.float32)
y = y.astype(np.float32)

field_locations = (
    Final_dataset[GROUP_SOURCE_COLUMN]
    .dropna()
    .unique()
)

if len(field_locations) < N_GROUPS:
    raise ValueError(
        f"Only {len(field_locations)} unique field locations were found, "
        f"but N_GROUPS is {N_GROUPS}."
    )

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Unique field locations:", len(field_locations))
print("Results folder:", OUTPUT_FOLDER)


###############################################################################
# Randomly assign field locations to groups
###############################################################################

def assign_locations_to_groups(field_locations, n_groups, random_seed):
    rng = np.random.default_rng(random_seed)
    shuffled_locations = rng.permutation(np.asarray(field_locations))
    split_locations = np.array_split(shuffled_locations, n_groups)

    location_to_group = {}

    for group_index, group_locations in enumerate(
        split_locations,
        start=1
    ):
        for location in group_locations:
            location_to_group[location] = f"Group_{group_index}"

    return location_to_group


###############################################################################
# Hyperparameter search space
###############################################################################

space = {
    "learning_rate": hp.loguniform(
        "learning_rate",
        np.log(0.005),
        np.log(0.30)
    ),

    "max_depth": scope.int(
        hp.quniform("max_depth", 2, 15, 1)
    ),

    "min_child_weight": hp.loguniform(
        "min_child_weight",
        np.log(1),
        np.log(20)
    ),

    "subsample": hp.uniform(
        "subsample",
        0.10,
        1.00
    ),

    "colsample_bytree": hp.uniform(
        "colsample_bytree",
        0.10,
        1.00
    ),

    "gamma": hp.loguniform(
        "gamma",
        np.log(0.01),
        np.log(0.50)
    ),

    "reg_alpha": hp.loguniform(
        "reg_alpha",
        np.log(10),
        np.log(200)
    ),

    "reg_lambda": hp.loguniform(
        "reg_lambda",
        np.log(1),
        np.log(10)
    )
}


###############################################################################
# Storage
###############################################################################

all_fold_results = []
all_iteration_results = []
best_parameters_all_folds = []
all_predictions = []


###############################################################################
# Repeated random group assignments
###############################################################################

for iteration in range(1, NUM_ITERATIONS + 1):

    iteration_start_time = time.time()

    print("\n" + "#" * 80)
    print(f"ITERATION {iteration}")
    print("#" * 80)

    location_to_group = assign_locations_to_groups(
        field_locations=field_locations,
        n_groups=N_GROUPS,
        random_seed=42 + iteration
    )

    Final_dataset["Group"] = (
        Final_dataset[GROUP_SOURCE_COLUMN]
        .map(location_to_group)
    )

    if Final_dataset["Group"].isna().any():
        raise ValueError(
            "Some observations were not assigned to a group."
        )

    group_labels = Final_dataset["Group"].to_numpy()

    # Save group assignments
    group_assignment_df = pd.DataFrame({
        GROUP_SOURCE_COLUMN: list(location_to_group.keys()),
        "Group": list(location_to_group.values())
    })

    group_assignment_df = (
        group_assignment_df
        .sort_values(["Group", GROUP_SOURCE_COLUMN])
        .reset_index(drop=True)
    )

    group_assignment_df.to_csv(
        OUTPUT_FOLDER
        / f"iteration_{iteration}_location_group_assignments.csv",
        index=False
    )

    group_text = ""

    for group_number in range(1, N_GROUPS + 1):
        current_group = f"Group_{group_number}"

        locations = [
            str(location)
            for location, group in location_to_group.items()
            if group == current_group
        ]

        group_text += (
            f"{current_group}: {', '.join(locations)}\n"
        )

    with open(
        OUTPUT_FOLDER / f"iteration_{iteration}_groups.txt",
        "w"
    ) as file:
        file.write(group_text)

    # Outer CV: hold out one complete group of field locations
    outer_group_kfold = GroupKFold(n_splits=N_GROUPS)

    iteration_y_test = []
    iteration_y_pred = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_group_kfold.split(
            X,
            y,
            groups=group_labels
        ),
        start=1
    ):

        fold_start_time = time.time()

        print("\n" + "=" * 75)
        print(
            f"ITERATION {iteration} - OUTER FOLD {outer_fold}"
        )
        print("=" * 75)

        X_train = X.iloc[outer_train_idx].copy()
        X_test = X.iloc[outer_test_idx].copy()

        y_train = y.iloc[outer_train_idx].copy()
        y_test = y.iloc[outer_test_idx].copy()

        train_groups = group_labels[outer_train_idx]
        test_groups = np.unique(group_labels[outer_test_idx])

        if len(test_groups) != 1:
            raise ValueError(
                "Each outer test fold should contain one group."
            )

        test_group = str(test_groups[0])

        print(
            "Outer training groups:",
            np.unique(train_groups)
        )
        print("Outer test group:", test_group)
        print("Outer training shape:", X_train.shape)
        print("Outer test shape:", X_test.shape)

        with open(
            OUTPUT_FOLDER
            / (
                f"iteration_{iteration}_fold_{outer_fold}_"
                "train_test_groups.txt"
            ),
            "w"
        ) as file:
            file.write(
                "Train groups: "
                + ", ".join(map(str, np.unique(train_groups)))
                + f"\nTest group: {test_group}\n"
            )


        #######################################################################
        # Hyperopt objective
        #
        # The inner CV uses the four outer-training groups.
        # One complete training group is held out in each inner fold.
        #######################################################################

        def objective(params):

            params = params.copy()
            params["max_depth"] = int(params["max_depth"])

            inner_fold_mse = []
            inner_best_iterations = []

            unique_inner_groups = np.unique(train_groups)

            inner_group_kfold = GroupKFold(
                n_splits=len(unique_inner_groups)
            )

            for inner_train_idx, inner_validation_idx in (
                inner_group_kfold.split(
                    X_train,
                    y_train,
                    groups=train_groups
                )
            ):

                X_inner_train = X_train.iloc[inner_train_idx]
                X_inner_validation = X_train.iloc[
                    inner_validation_idx
                ]

                y_inner_train = y_train.iloc[inner_train_idx]
                y_inner_validation = y_train.iloc[
                    inner_validation_idx
                ]

                temporary_model = xgb.XGBRegressor(
                    n_estimators=MAX_N_ESTIMATORS,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    early_stopping_rounds=(
                        EARLY_STOPPING_ROUNDS
                    ),
                    tree_method="hist",
                    random_state=42,
                    verbosity=0,
                    **params
                )

                temporary_model.fit(
                    X_inner_train,
                    y_inner_train,
                    eval_set=[
                        (
                            X_inner_validation,
                            y_inner_validation
                        )
                    ],
                    verbose=False
                )

                validation_predictions = (
                    temporary_model.predict(
                        X_inner_validation
                    )
                )

                validation_mse = mean_squared_error(
                    y_inner_validation,
                    validation_predictions
                )

                inner_fold_mse.append(validation_mse)

                best_iteration = getattr(
                    temporary_model,
                    "best_iteration",
                    MAX_N_ESTIMATORS - 1
                )

                inner_best_iterations.append(
                    int(best_iteration + 1)
                )

            return {
                "loss": float(np.mean(inner_fold_mse)),
                "status": STATUS_OK,
                "best_iteration": int(
                    np.median(inner_best_iterations)
                )
            }


        #######################################################################
        # Run Hyperopt
        #######################################################################

        trials = Trials()

        best_raw = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=trials,
            rstate=np.random.default_rng(
                42 + iteration * 100 + outer_fold
            )
        )

        best_params = space_eval(space, best_raw)
        best_params["max_depth"] = int(
            best_params["max_depth"]
        )

        best_trial = trials.best_trial

        best_n_estimators = int(
            best_trial["result"]["best_iteration"]
        )

        best_inner_mse = float(
            best_trial["result"]["loss"]
        )

        print("\nBest XGBoost parameters:")
        print(best_params)
        print(
            "Maximum trees allowed:",
            MAX_N_ESTIMATORS
        )
        print(
            "Early-stopping selected trees:",
            best_n_estimators
        )
        print(
            "Best mean inner group-CV MSE:",
            best_inner_mse
        )


        #######################################################################
        # Fit final outer model
        #######################################################################

        final_model = xgb.XGBRegressor(
            n_estimators=best_n_estimators,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            **best_params
        )

        final_model.fit(
            X_train,
            y_train
        )

        model_path = (
            OUTPUT_FOLDER
            / (
                "xgboost_optimized_leave_group_out_yield_"
                f"iteration_{iteration}_"
                f"fold_{outer_fold}_"
                f"test_{test_group}.json"
            )
        )

        final_model.save_model(str(model_path))

        y_pred = final_model.predict(X_test)

        outer_mse = mean_squared_error(
            y_test,
            y_pred
        )

        outer_rmse = np.sqrt(outer_mse)

        if (
            len(y_test) > 1
            and np.std(y_test) > 0
            and np.std(y_pred) > 0
        ):
            outer_correlation = float(
                pearsonr(y_test, y_pred)[0]
            )
        else:
            outer_correlation = np.nan

        print(
            f"\nPerformance for held-out group {test_group}:"
        )
        print(f"MSE = {outer_mse:.4f}")
        print(f"RMSE = {outer_rmse:.4f}")
        print(
            f"Pearson r = {outer_correlation:.4f}"
        )


        #######################################################################
        # Save metrics and parameters
        #######################################################################

        all_fold_results.append({
            "Iteration": iteration,
            "Fold": outer_fold,
            "Test_group": test_group,
            "MSE": outer_mse,
            "RMSE": outer_rmse,
            "Correlation": outer_correlation,
            "Best_inner_MSE": best_inner_mse,
            "Maximum_estimators": MAX_N_ESTIMATORS,
            "Best_n_estimators": best_n_estimators
        })

        best_parameters_all_folds.append({
            "Iteration": iteration,
            "Fold": outer_fold,
            "Test_group": test_group,
            "Best_inner_MSE": best_inner_mse,
            "Maximum_estimators": MAX_N_ESTIMATORS,
            "Best_n_estimators": best_n_estimators,
            **best_params
        })


        #######################################################################
        # Save fold predictions
        #######################################################################

        test_metadata = Final_dataset.iloc[
            outer_test_idx
        ]

        hybrid_env = (
            test_metadata[["Hybrid", "Env"]]
            .astype(str)
            .agg("_".join, axis=1)
            .to_numpy()
        )

        fold_predictions = pd.DataFrame({
            "Hybrid_Env": hybrid_env,
            "Hybrid": (
                test_metadata["Hybrid"]
                .astype(str)
                .to_numpy()
            ),
            "Env": (
                test_metadata["Env"]
                .astype(str)
                .to_numpy()
            ),
            "Field_Location": (
                test_metadata[GROUP_SOURCE_COLUMN]
                .astype(str)
                .to_numpy()
            ),
            "Year": test_metadata["Year"].to_numpy(),
            "Iteration": iteration,
            "Fold": outer_fold,
            "Test_group": test_group,
            "Actual": y_test.to_numpy(),
            "Predicted": y_pred
        })

        fold_predictions.to_csv(
            OUTPUT_FOLDER
            / (
                f"iteration_{iteration}_"
                f"fold_{outer_fold}_"
                f"test_{test_group}_predictions.csv"
            ),
            index=False
        )

        all_predictions.append(fold_predictions)

        iteration_y_test.append(
            y_test.to_numpy()
        )
        iteration_y_pred.append(y_pred)


        #######################################################################
        # Save checkpoints after each completed outer fold
        #######################################################################

        pd.DataFrame(all_fold_results).to_csv(
            OUTPUT_FOLDER
            / "leave_group_out_results_checkpoint.csv",
            index=False
        )

        pd.DataFrame(
            best_parameters_all_folds
        ).to_csv(
            OUTPUT_FOLDER
            / (
                "leave_group_out_"
                "best_parameters_checkpoint.csv"
            ),
            index=False
        )

        pd.concat(
            all_predictions,
            ignore_index=True
        ).to_csv(
            OUTPUT_FOLDER
            / "leave_group_out_predictions_checkpoint.csv",
            index=False
        )

        fold_runtime_minutes = (
            time.time() - fold_start_time
        ) / 60

        print(
            f"\nOuter-fold runtime: "
            f"{fold_runtime_minutes:.2f} minutes"
        )


    ###########################################################################
    # Iteration summary
    ###########################################################################

    current_fold_results = pd.DataFrame(
        [
            result
            for result in all_fold_results
            if result["Iteration"] == iteration
        ]
    )

    mean_fold_mse = current_fold_results["MSE"].mean()
    mean_fold_rmse = current_fold_results["RMSE"].mean()
    mean_fold_correlation = (
        current_fold_results["Correlation"].mean()
    )

    iteration_true = np.concatenate(
        iteration_y_test
    )
    iteration_pred = np.concatenate(
        iteration_y_pred
    )

    pooled_iteration_mse = mean_squared_error(
        iteration_true,
        iteration_pred
    )

    pooled_iteration_rmse = np.sqrt(
        pooled_iteration_mse
    )

    if (
        np.std(iteration_true) > 0
        and np.std(iteration_pred) > 0
    ):
        pooled_iteration_correlation = float(
            pearsonr(
                iteration_true,
                iteration_pred
            )[0]
        )
    else:
        pooled_iteration_correlation = np.nan

    all_iteration_results.append({
        "Iteration": iteration,
        "Mean_fold_MSE": mean_fold_mse,
        "Mean_fold_RMSE": mean_fold_rmse,
        "Mean_fold_Correlation": (
            mean_fold_correlation
        ),
        "Pooled_prediction_MSE": (
            pooled_iteration_mse
        ),
        "Pooled_prediction_RMSE": (
            pooled_iteration_rmse
        ),
        "Pooled_prediction_Correlation": (
            pooled_iteration_correlation
        )
    })

    pd.DataFrame(
        all_iteration_results
    ).to_csv(
        OUTPUT_FOLDER
        / (
            "leave_group_out_"
            "iteration_summary_checkpoint.csv"
        ),
        index=False
    )

    iteration_runtime_minutes = (
        time.time() - iteration_start_time
    ) / 60

    print("\n" + "-" * 75)
    print(f"ITERATION {iteration} SUMMARY")
    print("-" * 75)
    print(
        f"Mean fold RMSE: "
        f"{mean_fold_rmse:.4f}"
    )
    print(
        f"Mean fold correlation: "
        f"{mean_fold_correlation:.4f}"
    )
    print(
        f"Pooled prediction RMSE: "
        f"{pooled_iteration_rmse:.4f}"
    )
    print(
        f"Pooled prediction correlation: "
        f"{pooled_iteration_correlation:.4f}"
    )
    print(
        f"Iteration runtime: "
        f"{iteration_runtime_minutes:.2f} minutes"
    )


###############################################################################
# Final files
###############################################################################

fold_results_df = pd.DataFrame(
    all_fold_results
)

iteration_results_df = pd.DataFrame(
    all_iteration_results
)

best_parameters_df = pd.DataFrame(
    best_parameters_all_folds
)

all_predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

fold_results_df.to_csv(
    OUTPUT_FOLDER
    / (
        "leave_group_out_fold_results_"
        "optimized_xgboost_yield.csv"
    ),
    index=False
)

iteration_results_df.to_csv(
    OUTPUT_FOLDER
    / (
        "leave_group_out_iteration_summary_"
        "optimized_xgboost_yield.csv"
    ),
    index=False
)

best_parameters_df.to_csv(
    OUTPUT_FOLDER
    / (
        "leave_group_out_"
        "best_parameters_each_fold.csv"
    ),
    index=False
)

all_predictions_df.to_csv(
    OUTPUT_FOLDER
    / (
        "leave_group_out_all_predictions_"
        "optimized_xgboost_yield.csv"
    ),
    index=False
)


###############################################################################
# Overall summary
###############################################################################

mean_iteration_pooled_rmse = (
    iteration_results_df[
        "Pooled_prediction_RMSE"
    ].mean()
)

mean_iteration_pooled_correlation = (
    iteration_results_df[
        "Pooled_prediction_Correlation"
    ].mean()
)

overall_true = all_predictions_df["Actual"].to_numpy()
overall_pred = all_predictions_df["Predicted"].to_numpy()

overall_pooled_rmse = np.sqrt(
    mean_squared_error(
        overall_true,
        overall_pred
    )
)

if (
    np.std(overall_true) > 0
    and np.std(overall_pred) > 0
):
    overall_pooled_correlation = float(
        pearsonr(
            overall_true,
            overall_pred
        )[0]
    )
else:
    overall_pooled_correlation = np.nan

overall_summary_df = pd.DataFrame([
    {
        "Summary": (
            "Mean_across_iteration_pooled_metrics"
        ),
        "RMSE": mean_iteration_pooled_rmse,
        "Correlation": (
            mean_iteration_pooled_correlation
        )
    },
    {
        "Summary": (
            "Pooled_all_predictions_all_iterations"
        ),
        "RMSE": overall_pooled_rmse,
        "Correlation": overall_pooled_correlation
    }
])

overall_summary_df.to_csv(
    OUTPUT_FOLDER
    / (
        "leave_group_out_overall_summary_"
        "optimized_xgboost_yield.csv"
    ),
    index=False
)


###############################################################################
# Print final results and runtime
###############################################################################

print("\n" + "=" * 80)
print("FINAL ITERATION SUMMARIES")
print("=" * 80)
print(iteration_results_df)

print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)
print(overall_summary_df)

total_runtime_minutes = (
    time.time() - total_start_time
) / 60

print(
    f"\nTotal runtime: "
    f"{total_runtime_minutes:.2f} minutes"
)

print("\nResults saved in:")
print(OUTPUT_FOLDER)