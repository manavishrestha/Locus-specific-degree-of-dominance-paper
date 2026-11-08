#25.08.2026
# Split for LOEO


from pathlib import Path
import pandas as pd
import re


###############################################################################
# Folder containing the files from your original Yield CV
###############################################################################

SPLIT_FOLDER = Path(
    "/Users/manavi/Desktop/Manuscript_review/"
    "xgboost_optimized_leave_group_env_out_yield"
)

NUM_ITERATIONS = 5
N_GROUPS = 5


###############################################################################
# Storage
###############################################################################

all_splits = []


###############################################################################
# Loop over iterations
###############################################################################

for iteration in range(1, NUM_ITERATIONS + 1):

    print(f"Processing iteration {iteration}")

    ###########################################################################
    # 1. Read Field_Location -> Group assignment
    ###########################################################################

    location_file = (
        SPLIT_FOLDER
        / f"iteration_{iteration}_location_group_assignments.csv"
    )

    df_iter = pd.read_csv(location_file)

    # should contain:
    # Field_Location
    # Group

    df_iter["Iteration"] = iteration


    ###########################################################################
    # 2. Loop over folds
    ###########################################################################

    for fold in range(1, N_GROUPS + 1):

        train_test_file = (
            SPLIT_FOLDER
            / (
                f"iteration_{iteration}_fold_{fold}_"
                "train_test_groups.txt"
            )
        )

        #######################################################################
        # Read train/test information
        #######################################################################

        with open(train_test_file, "r") as file:
            lines = file.read()

        print(
            f"\nIteration {iteration}, Fold {fold}"
        )
        print(lines)


        #######################################################################
        # Extract test group
        #
        # Example file:
        #
        # Train groups: Group_2, Group_3, Group_4, Group_5
        # Test group: Group_1
        #######################################################################

        test_match = re.search(
            r"Test group:\s*(Group_\d+)",
            lines
        )

        if test_match is None:
            raise ValueError(
                f"Could not identify test group in:\n"
                f"{train_test_file}"
            )

        test_group = test_match.group(1)


        #######################################################################
        # 3. Make one copy of location assignments for this fold
        #######################################################################

        df_fold = df_iter.copy()

        df_fold["Fold"] = fold

        df_fold["Split"] = df_fold["Group"].apply(
            lambda group:
                "test" if group == test_group else "train"
        )

        df_fold["Test_group"] = test_group


        #######################################################################
        # Store
        #######################################################################

        all_splits.append(df_fold)


###############################################################################
# Combine everything
###############################################################################

master_splits = pd.concat(
    all_splits,
    ignore_index=True
)


###############################################################################
# Arrange columns
###############################################################################

master_splits = master_splits[
    [
        "Field_Location",
        "Iteration",
        "Fold",
        "Group",
        "Split",
        "Test_group"
    ]
]


###############################################################################
# Sort
###############################################################################

master_splits = (
    master_splits
    .sort_values(
        [
            "Iteration",
            "Fold",
            "Group",
            "Field_Location"
        ]
    )
    .reset_index(drop=True)
)


###############################################################################
# Check
###############################################################################

print(master_splits)

print(
    master_splits.groupby(
        ["Iteration", "Fold", "Split"]
    )["Field_Location"]
    .nunique()
)


###############################################################################
# Save ONE master file
###############################################################################

master_splits.to_csv(
    SPLIT_FOLDER / "MASTER_CV_LOCATION_SPLITS.csv",
    index=False
)

print(
    "\nSaved master split file:"
)

print(
    SPLIT_FOLDER / "MASTER_CV_LOCATION_SPLITS.csv"
)