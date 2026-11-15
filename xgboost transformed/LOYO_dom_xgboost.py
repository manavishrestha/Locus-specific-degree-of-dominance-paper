
# yield dom LOYO hyperopt 14.08.2026
# read thisone
# use this version


##############################################################################
# Libraries
###############################################################################

from pathlib import Path
import time
import random

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from hyperopt import (
    hp,
    fmin,
    tpe,
    Trials,
    STATUS_OK,
    space_eval
)

from hyperopt.pyll import scope


###############################################################################
# Main settings
###############################################################################

random.seed(42)
np.random.seed(42)

# Number of Hyperopt parameter combinations tested for every outer test year
MAX_EVALS = 20 #change to 20 later

# Maximum number of boosting rounds allowed
MAX_N_ESTIMATORS = 8000

# Stop when validation RMSE has not improved for this many rounds
EARLY_STOPPING_ROUNDS = 200

# Number of SNP columns at the beginning of X
# The remaining predictor columns are environmental covariates
N_SNP_COLUMNS = 2397


###############################################################################
# Start measuring total runtime
###############################################################################

total_start_time = time.time()


###############################################################################
# Load complete yield dataset
###############################################################################

Final_dataset = pd.read_csv(
    "/Users/user/Desktop/Manuscript_review/"
    "Datasets/Final_merged_March19.csv"
)

print(
    "Original dataset shape:",
    Final_dataset.shape
)


###############################################################################
# Remove rows with missing response
###############################################################################

Final_dataset = (
    Final_dataset
    .dropna(
        subset=["Average_BLUE"]
    )
    .reset_index(drop=True)
)

print(
    "Dataset shape after removing missing responses:",
    Final_dataset.shape
)

print("\nObservations per year:")

print(
    Final_dataset["Year"]
    .value_counts()
    .sort_index()
)


###############################################################################
# Create grouping variable, predictors and response
###############################################################################

years = Final_dataset[
    "Year"
].to_numpy()

unique_years = sorted(
    Final_dataset[
        "Year"
    ].unique()
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


y = Final_dataset[
    "Average_BLUE"
].copy()


###############################################################################
# Check that all predictor columns are numeric
###############################################################################

non_numeric_columns = (
    X.select_dtypes(
        exclude=np.number
    )
    .columns
    .tolist()
)

if len(non_numeric_columns) > 0:

    raise TypeError(
        "These predictor columns are not numeric:\n"
        f"{non_numeric_columns}"
    )


###############################################################################
# Convert to float32 to reduce memory use
###############################################################################

X = X.astype(
    np.float32
)

y = y.astype(
    np.float32
)


print(
    "\nX shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "Years:",
    np.unique(years)
)


###############################################################################
# Output folder
###############################################################################

output_folder = Path(
    "/Users/user/Desktop/Manuscript_review/"
    "xgboost_optimized_LOYO_dominance_yield"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

print(
    "\nResults will be saved in:"
)

print(
    output_folder
)


###############################################################################
# Hyperparameter search space
#
# n_estimators is not optimized here.
# It has a fixed upper limit of 8000.
# Early stopping determines the effective number of trees.
###############################################################################

space = {

    # Continuous, log-uniform
    "learning_rate": hp.loguniform(
        "learning_rate",
        np.log(0.005),
        np.log(0.30)
    ),

    # Integer
    "max_depth": scope.int(
        hp.quniform(
            "max_depth",
            2,
            15,
            1
        )
    ),

    # Continuous, log-uniform
    "min_child_weight": hp.loguniform(
        "min_child_weight",
        np.log(1),
        np.log(20)
    ),

    # Continuous, uniform
    "subsample": hp.uniform(
        "subsample",
        0.10,
        1.00
    ),

    # Continuous, uniform
    "colsample_bytree": hp.uniform(
        "colsample_bytree",
        0.10,
        1.00
    ),

    # Continuous, log-uniform
    "gamma": hp.loguniform(
        "gamma",
        np.log(0.01),
        np.log(0.50)
    ),

    # Continuous, log-uniform
    "reg_alpha": hp.loguniform(
        "reg_alpha",
        np.log(10),
        np.log(200)
    ),

    # Continuous, log-uniform
    "reg_lambda": hp.loguniform(
        "reg_lambda",
        np.log(1),
        np.log(10)
    )
}



###############################################################################
# Outer leave-one-year-out cross-validation
#
# Each unique year is held out once as the outer test set.
# All remaining years are used as the outer training set.
#
# Nested leave-one-year-out optimization requires at least three unique years:
# one outer test year and at least two years remaining for inner year-based CV.
###############################################################################

if len(unique_years) < 3:

    raise ValueError(
        "At least three unique years are required for nested leave-one-year-out "
        "cross-validation."
    )


###############################################################################
# Objects for storing results
###############################################################################

results = []

best_parameters_all_folds = []

all_y_test = []

all_y_pred = []

all_hybrid_env = []

all_test_years = []


###############################################################################
# Begin outer leave-one-year-out loop
###############################################################################

leave_one_year_out = LeaveOneGroupOut()

for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
    leave_one_year_out.split(
        X,
        y,
        groups=years
    ),
    start=1
):

    fold_start_time = time.time()

    print("\n" + "=" * 75)

    print(
        f"OUTER FOLD {outer_fold}"
    )

    print("=" * 75)


    ###########################################################################
    # Outer training years and held-out test year
    ###########################################################################

    outer_training_years = np.unique(
        years[
            outer_train_idx
        ]
    )

    test_year = np.unique(
        years[
            outer_test_idx
        ]
    )[0]


    ###########################################################################
    # Create outer training and test datasets
    ###########################################################################

    X_train = X.iloc[
        outer_train_idx
    ].copy()

    X_test = X.iloc[
        outer_test_idx
    ].copy()


    y_train = y.iloc[
        outer_train_idx
    ].copy()

    y_test = y.iloc[
        outer_test_idx
    ].copy()

    ###########################################################################
    # Dominance transformation
    #
    # d is calculated using the COMPLETE OUTER-TRAINING SET only.
    # The same SNP-specific d values are then applied to both:
    #   1. outer-training SNPs
    #   2. held-out outer-test-year SNPs
    #
    # Environmental covariates are not transformed.
    ###########################################################################

    # Extract the SNP part of the outer-training and outer-test matrices
    xtrain_snp = X_train.iloc[
        :,
        :N_SNP_COLUMNS
    ].to_numpy(
        copy=True
    )

    xtest_snp = X_test.iloc[
        :,
        :N_SNP_COLUMNS
    ].to_numpy(
        copy=True
    )

    # Keep original SNP matrices for the final 1 -> d replacement,
    # matching the dominance code used in the original script
    features_train = X_train.iloc[
        :,
        :N_SNP_COLUMNS
    ].to_numpy(
        copy=True
    )

    features_test = X_test.iloc[
        :,
        :N_SNP_COLUMNS
    ].to_numpy(
        copy=True
    )

    # Create matrix for mean phenotype of genotype 0, 1 and 2
    snp_mean = np.zeros(
        (
            xtrain_snp.shape[1],
            3
        )
    )

    print(
        "Number of SNPs for dominance transformation:",
        xtrain_snp.shape[1]
    )

    # Loop over each SNP
    for snp in range(
        xtrain_snp.shape[1]
    ):

        for genotype in range(3):

            # Calculate mean outer-training phenotype
            # for each genotype category
            snp_mean[
                snp,
                genotype
            ] = np.nanmean(
                y_train[
                    xtrain_snp[:, snp]
                    ==
                    genotype
                ]
            )

            # Replace NaN values with 0
            snp_mean[
                np.isnan(
                    snp_mean
                )
            ] = 0.0


    # Create condition where genotype-0 mean > genotype-2 mean
    my_choose = (
        snp_mean[:, 0]
        >
        snp_mean[:, 2]
    )


    # Swap genotype-0 and genotype-2 phenotype means safely
    temp = snp_mean[
        my_choose,
        0
    ].copy()

    snp_mean[
        my_choose,
        0
    ] = snp_mean[
        my_choose,
        2
    ]

    snp_mean[
        my_choose,
        2
    ] = temp


    # Flip genotype coding in the temporary outer-training SNP matrix
    # exactly as in the original dominance code
    xtrain_snp[
        :,
        my_choose
    ] = np.abs(
        xtrain_snp[
            :,
            my_choose
        ]
        -
        2
    )


    # Calculate d, the degree of dominance
    d = (
        (
            snp_mean[:, 1]
            -
            snp_mean[:, 0]
        )
        /
        (
            snp_mean[:, 2]
            -
            snp_mean[:, 0]
        )
        *
        2
    )


    ###########################################################################
    # Replace missing d values with 1
    ###########################################################################

    my_choose_na = np.isnan(
        d
    )

    d[
        my_choose_na
    ] = 1


    ###########################################################################
    # Save dominance values BEFORE capping to 0-2
    #
    # At this point:
    #   - missing genotype means have already been changed to 0
    #   - d has been calculated
    #   - missing d values have already been changed to 1
    #   - d has NOT yet been capped to 0-2
    ###########################################################################

    d_before_capping = d.copy()

    dominance_test_year = np.unique(
        years[
            outer_test_idx
        ]
    )[0]

    dominance_before_capping = pd.DataFrame({
        "SNP": X_train.columns[:N_SNP_COLUMNS],
        "Mean_genotype_0": snp_mean[:, 0],
        "Mean_genotype_1": snp_mean[:, 1],
        "Mean_genotype_2": snp_mean[:, 2],
        "d_before_capping": d_before_capping
    })

    dominance_before_capping.to_csv(
        output_folder
        /
        (
            f"Dominance_Calculation_Before_Capping_"
            f"Fold_{outer_fold}_Test_Year_{dominance_test_year}.csv"
        ),
        index=False
    )


    ###########################################################################
    # Set boundaries for d
    ###########################################################################

    my_choose1 = (
        d >= 2
    )

    d[
        my_choose1
    ] = 2


    my_choose2 = (
        d <= 0
    )

    d[
        my_choose2
    ] = 0


    ###########################################################################
    # Save dominance values AFTER capping to 0-2
    ###########################################################################

    dominance_after_capping = pd.DataFrame({
        "SNP": X_train.columns[:N_SNP_COLUMNS],
        "Mean_genotype_0": snp_mean[:, 0],
        "Mean_genotype_1": snp_mean[:, 1],
        "Mean_genotype_2": snp_mean[:, 2],
        "d_after_capping": d
    })

    dominance_after_capping.to_csv(
        output_folder
        /
        (
            f"Dominance_Calculation_After_Capping_"
            f"Fold_{outer_fold}_Test_Year_{dominance_test_year}.csv"
        ),
        index=False
    )


    # Weighted heterozygous genotypes in outer-training SNP matrix
    for snp in range(
        features_train.shape[1]
    ):

        my_choose = (
            features_train[
                :,
                snp
            ]
            ==
            1
        )

        features_train[
            my_choose,
            snp
        ] = d[
            snp
        ]


    # Apply the SAME d values to the held-out outer-test-year SNP matrix
    for snp in range(
        features_test.shape[1]
    ):

        my_choose = (
            features_test[
                :,
                snp
            ]
            ==
            1
        )

        features_test[
            my_choose,
            snp
        ] = d[
            snp
        ]


    # Put the transformed SNPs back into X_train and X_test.
    # Environmental columns after the first 2397 columns remain unchanged.
    X_train.iloc[
        :,
        :N_SNP_COLUMNS
    ] = features_train

    X_test.iloc[
        :,
        :N_SNP_COLUMNS
    ] = features_test


    ###########################################################################
    # Outer-training years and held-out test year
    ###########################################################################

    train_years = years[
        outer_train_idx
    ]


    test_year = np.unique(
        years[
            outer_test_idx
        ]
    )[0]


    print(
        "Outer training years:",
        np.unique(train_years)
    )

    print(
        "Outer test year:",
        test_year
    )

    print(
        "Outer training shape:",
        X_train.shape
    )

    print(
        "Outer test shape:",
        X_test.shape
    )


    ###########################################################################
    # Objective function
    #
    # Hyperopt supplies one parameter combination.
    # It is evaluated using INNER LEAVE-ONE-YEAR-OUT CV.
    ###########################################################################

    def objective(params):

        # Create a copy so Hyperopt's original dictionary is not changed
        params = params.copy()

        # max_depth must be an integer
        params["max_depth"] = int(
            params["max_depth"]
        )


        #######################################################################
        # Inner leave-one-year-out setup
        #######################################################################

        inner_leave_one_year_out = LeaveOneGroupOut()


        #######################################################################
        # Store inner-fold results
        #######################################################################

        inner_fold_mse = []

        inner_best_iterations = []


        #######################################################################
        # Inner leave-one-year-out loop
        #
        # For each outer fold, every year remaining in the outer-training set
        # is held out once as the inner validation year.
        #######################################################################

        for inner_fold, (
            inner_train_idx,
            inner_validation_idx
        ) in enumerate(

            inner_leave_one_year_out.split(
                X_train,
                y_train,
                groups=train_years
            ),

            start=1
        ):

            inner_validation_year = np.unique(
                train_years[
                    inner_validation_idx
                ]
            )[0]


            ###################################################################
            # Inner training data
            ###################################################################

            X_inner_train = X_train.iloc[
                inner_train_idx
            ]

            y_inner_train = y_train.iloc[
                inner_train_idx
            ]


            ###################################################################
            # Inner validation data
            ###################################################################

            X_inner_validation = X_train.iloc[
                inner_validation_idx
            ]

            y_inner_validation = y_train.iloc[
                inner_validation_idx
            ]


            ###################################################################
            # Temporary XGBoost model
            ###################################################################

            temporary_model = xgb.XGBRegressor(

                n_estimators=MAX_N_ESTIMATORS,

                objective="reg:squarederror",

                eval_metric="rmse",

                early_stopping_rounds=
                    EARLY_STOPPING_ROUNDS,

                tree_method="hist",

                random_state=42,

                verbosity=0,

                **params
            )


            ###################################################################
            # Train temporary model
            #
            # The held-out inner year controls early stopping.
            ###################################################################

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


            ###################################################################
            # Predict inner validation year
            ###################################################################

            inner_validation_predictions = (
                temporary_model.predict(
                    X_inner_validation
                )
            )


            ###################################################################
            # Calculate inner validation MSE
            ###################################################################

            validation_mse = mean_squared_error(
                y_inner_validation,
                inner_validation_predictions
            )


            inner_fold_mse.append(
                validation_mse
            )


            ###################################################################
            # Obtain number of trees selected by early stopping
            #
            # best_iteration is zero-based, so add 1.
            ###################################################################

            model_best_iteration = getattr(
                temporary_model,
                "best_iteration",
                MAX_N_ESTIMATORS - 1
            )


            number_of_trees = int(
                model_best_iteration + 1
            )


            inner_best_iterations.append(
                number_of_trees
            )


        #######################################################################
        # Mean MSE across inner leave-one-year-out validation folds
        #######################################################################

        mean_inner_mse = float(
            np.mean(
                inner_fold_mse
            )
        )


        #######################################################################
        # Median selected tree number across inner folds
        #######################################################################

        median_best_iteration = int(
            np.median(
                inner_best_iterations
            )
        )


        #######################################################################
        # Return results to Hyperopt
        #######################################################################

        return {

            "loss":
                mean_inner_mse,

            "status":
                STATUS_OK,

            "best_iteration":
                median_best_iteration
        }
    ###########################################################################
    # Run Hyperopt for the current outer fold
    ###########################################################################

    trials = Trials()


    best_raw = fmin(

        fn=objective,

        space=space,

        algo=tpe.suggest,

        max_evals=MAX_EVALS,

        trials=trials,

        rstate=np.random.default_rng(
            42 + outer_fold
        )
    )


    ###########################################################################
    # Convert Hyperopt result into usable parameter values
    ###########################################################################

    best_params = space_eval(
        space,
        best_raw
    )


    # Ensure max_depth is stored as an integer
    best_params["max_depth"] = int(
        best_params["max_depth"]
    )


    ###########################################################################
    # Retrieve best Hyperopt trial
    ###########################################################################

    best_trial = trials.best_trial


    ###########################################################################
    # Retrieve number of trees selected by early stopping
    ###########################################################################

    best_n_estimators = int(
        best_trial[
            "result"
        ][
            "best_iteration"
        ]
    )


    ###########################################################################
    # Retrieve best inner leave-one-year-out MSE
    ###########################################################################

    best_inner_mse = float(
        best_trial[
            "result"
        ][
            "loss"
        ]
    )


    ###########################################################################
    # The fixed candidate maximum is 8000 trees
    ###########################################################################

    candidate_max_estimators = (
        MAX_N_ESTIMATORS
    )


    ###########################################################################
    # Parameters used in the final model
    ###########################################################################

    best_params_final = best_params.copy()


    print("\nBest XGBoost parameters:")

    print(
        best_params_final
    )

    print(
        "Maximum trees allowed:",
        candidate_max_estimators
    )

    print(
        "Early-stopping selected trees:",
        best_n_estimators
    )

    print(
        "Best mean inner leave-one-year-out MSE:",
        best_inner_mse
    )


    ###########################################################################
    # Fit final model using all outer-training years
    ###########################################################################

    final_model = xgb.XGBRegressor(

        n_estimators=
            best_n_estimators,

        objective=
            "reg:squarederror",

        eval_metric=
            "rmse",

        tree_method=
            "hist",

        random_state=
            42,

        verbosity=
            0,

        **best_params_final
    )


    final_model.fit(
        X_train,
        y_train
    )


    ###########################################################################
    # Save fitted final model immediately
    ###########################################################################

    model_path = (
        output_folder
        /
        (
            f"xgboost_optimized_LOYO_dominance_yield_"
            f"test_year_{test_year}.json"
        )
    )


    final_model.save_model(
        str(model_path)
    )


    print(
        "Fitted model saved:",
        model_path
    )


    ###########################################################################
    # Predict untouched outer-test year
    ###########################################################################

    y_pred = final_model.predict(
        X_test
    )


    ###########################################################################
    # Calculate outer-test RMSE
    ###########################################################################

    outer_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )


    ###########################################################################
    # Calculate outer-test Pearson correlation
    ###########################################################################

    if (
        len(y_test) > 1
        and np.std(y_test) > 0
        and np.std(y_pred) > 0
    ):

        outer_correlation = float(
            pearsonr(
                y_test,
                y_pred
            )[0]
        )

    else:

        outer_correlation = np.nan


    print(
        f"\nFinal performance for held-out year {test_year}:"
    )

    print(
        f"RMSE = {outer_rmse:.3f}"
    )

    print(
        f"Pearson r = {outer_correlation:.3f}"
    )


    ###########################################################################
    # Obtain Hybrid and Env for outer-test observations
    ###########################################################################

    hybrid_env = (
        Final_dataset.iloc[
            outer_test_idx
        ][
            [
                "Hybrid",
                "Env"
            ]
        ]
        .astype(str)
        .agg(
            "_".join,
            axis=1
        )
        .to_numpy()
    )


    ###########################################################################
    # Store outer-fold results
    ###########################################################################

    results.append({

        "Year":
            test_year,

        "RMSE":
            outer_rmse,

        "Correlation":
            outer_correlation,

        "Best_inner_MSE":
            best_inner_mse,

        "Maximum_estimators":
            candidate_max_estimators,

        "Best_n_estimators":
            best_n_estimators
    })


    ###########################################################################
    # Store best parameters for current outer fold
    ###########################################################################

    best_parameters_all_folds.append({

        "Test_year":
            test_year,

        "Best_inner_MSE":
            best_inner_mse,

        "Maximum_estimators":
            candidate_max_estimators,

        "Best_n_estimators":
            best_n_estimators,

        **best_params_final
    })


    ###########################################################################
    # Create and save outer-fold predictions
    ###########################################################################

    fold_predictions = pd.DataFrame({

        "Hybrid_Env":
            hybrid_env,

        "Year":
            test_year,

        "Actual":
            y_test.to_numpy(),

        "Predicted":
            y_pred
    })


    fold_predictions.to_csv(

        output_folder
        /
        (
            f"fold_{test_year}_yield_"
            "optimized_xgboost_LOYO_dominance.csv"
        ),

        index=False
    )


    ###########################################################################
    # Store values for pooled calculations
    ###########################################################################

    all_y_test.append(
        y_test.to_numpy()
    )

    all_y_pred.append(
        y_pred
    )

    all_hybrid_env.append(
        hybrid_env
    )

    all_test_years.append(
        np.repeat(
            test_year,
            len(y_test)
        )
    )


    ###########################################################################
    # Save checkpoint results after every completed LOYO outer fold
    ###########################################################################

    pd.DataFrame(
        results
    ).to_csv(

        output_folder
        /
        "LOYO_dominance_results_checkpoint.csv",

        index=False
    )


    pd.DataFrame(
        best_parameters_all_folds
    ).to_csv(

        output_folder
        /
        "LOYO_dominance_best_parameters_checkpoint.csv",

        index=False
    )


    ###########################################################################
    # Print outer-fold runtime
    ###########################################################################

    fold_runtime_minutes = (
        time.time()
        -
        fold_start_time
    ) / 60


    print(
        f"\nOuter-fold runtime: "
        f"{fold_runtime_minutes:.2f} minutes"
    )


###############################################################################
# Convert outer-fold results to DataFrame
###############################################################################

results_df = pd.DataFrame(
    results
)


###############################################################################
# Calculate mean across year-specific metrics
###############################################################################

mean_rmse = results_df[
    "RMSE"
].mean()


mean_correlation = results_df[
    "Correlation"
].mean()


mean_row = pd.DataFrame(
    [
        {
            "Year":
                "Mean_across_years",

            "RMSE":
                mean_rmse,

            "Correlation":
                mean_correlation,

            "Best_inner_MSE":
                np.nan,

            "Maximum_estimators":
                np.nan,

            "Best_n_estimators":
                np.nan
        }
    ]
)


###############################################################################
# Combine all outer-test predictions
###############################################################################

final_y_true = np.concatenate(
    all_y_test
)


final_y_pred = np.concatenate(
    all_y_pred
)


final_hybrid_env = np.concatenate(
    all_hybrid_env
)


final_test_years = np.concatenate(
    all_test_years
)


###############################################################################
# Calculate pooled RMSE
###############################################################################

pooled_rmse = np.sqrt(
    mean_squared_error(
        final_y_true,
        final_y_pred
    )
)


###############################################################################
# Calculate pooled Pearson correlation
###############################################################################

if (
    np.std(final_y_true) > 0
    and np.std(final_y_pred) > 0
):

    pooled_correlation = float(
        pearsonr(
            final_y_true,
            final_y_pred
        )[0]
    )

else:

    pooled_correlation = np.nan


pooled_row = pd.DataFrame(
    [
        {
            "Year":
                "Pooled_predictions",

            "RMSE":
                pooled_rmse,

            "Correlation":
                pooled_correlation,

            "Best_inner_MSE":
                np.nan,

            "Maximum_estimators":
                np.nan,

            "Best_n_estimators":
                np.nan
        }
    ]
)


###############################################################################
# Add mean and pooled rows
###############################################################################

results_df = pd.concat(
    [
        results_df,
        mean_row,
        pooled_row
    ],
    ignore_index=True
)


###############################################################################
# Save final LOYO outer-fold results
###############################################################################

results_df.to_csv(

    output_folder
    /
    "LOYO_dominance_results_yield_optimized_xgboost.csv",

    index=False
)


###############################################################################
# Save best parameters for every LOYO outer fold
###############################################################################

best_parameters_df = pd.DataFrame(
    best_parameters_all_folds
)


best_parameters_df.to_csv(

    output_folder
    /
    "best_parameters_each_LOYO_dominance_fold.csv",

    index=False
)


###############################################################################
# Create final prediction DataFrame
###############################################################################

final_predictions_df = pd.DataFrame({

    "Hybrid_Env":
        final_hybrid_env,

    "Year":
        final_test_years,

    "Actual":
        final_y_true,

    "Predicted":
        final_y_pred
})


###############################################################################
# Save all outer-test predictions
###############################################################################

final_predictions_df.to_csv(

    output_folder
    /
    "LOYO_dominance_final_predictions_optimized_xgboost_yield.csv",

    index=False
)


###############################################################################
# Print final results
###############################################################################

print("\n" + "=" * 75)

print("FINAL RESULTS")

print("=" * 75)

print(
    results_df
)


###############################################################################
# Print total runtime
###############################################################################

total_runtime_minutes = (
    time.time()
    -
    total_start_time
) / 60


print(
    f"\nTotal runtime: "
    f"{total_runtime_minutes:.2f} minutes"
)



print(
    "\nResults saved in:"
)

print(
    output_folder
)