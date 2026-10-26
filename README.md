# Genomic Prediction Using Locus-Specific Degree of Dominance Transformation

**Author:** Manavi Shrestha  
**Date:** October 23, 2026

## Dataset

The dataset used in this study was obtained from:

Genomes To Fields (2025). *Genomes to Fields 2024 Maize Genotype by Environment Prediction Competition*. CyVerse Data Commons.

**DOI:** https://doi.org/10.25739/78mn-4394

---

## 1. Data Preprocessing

### 1.1 Trait Data

Download the trait dataset:

```bash
wget "https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/1_Training_Trait_Data_2014_2023.csv"
```

Run the preprocessing script:

```bash
python Trait_data_processing.py
```

The script performs the following preprocessing steps:

- Filtering data for the years 2019–2023
- Removal of missing data
- Z-score-based outlier removal

**Output file:** `trait_clean_noOut.csv`

### 1.2 Genotype Data

Download the genotype dataset:

```bash
wget "https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt"
```

Run the preprocessing script:

```bash
python Genotype_preprocessing.py
```

The script performs the following preprocessing steps:

- Removal of columns with a high proportion of missing values
- Imputation of missing genotype values using K-nearest neighbors (KNN)

**Output file:** `Final_filtered_Recoded_imputed_SNP.csv`

### 1.3 Environmental Covariate Data

Download the environmental covariate dataset:

```bash
wget "https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/6_Training_EC_Data_2014_2023.csv"
```

Run the preprocessing script:

```bash
python "Environmental covariates.py"
```

The script performs:

- Filtering data for the years 2019–2023

**Output file:** `Env_clean.csv`

### 1.4 Metadata

Download the metadata:

```bash
wget "https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/2_Training_Meta_Data_2014_2023.csv"
```

Run the preprocessing script:

```bash
python "Metadata processing.py"
```

The script performs the following preprocessing steps:

- Filtering data for the years 2019–2023
- Removal of columns with a high proportion of missing values
- Manual imputation of missing values

**Output file:** `metadata_clean_imputed.csv`

### 1.5 Weather Data

Download the weather dataset:

```bash
wget "https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/4_Training_Weather_Data_2014_2023_full_year.csv"
```

Run the preprocessing script:

```bash
python "Weather data.py"
```

The script performs:

- Filtering data for the years 2019–2023

**Output file:** `weather_clean.csv`

---

## 2. BLUEs Calculation

Best Linear Unbiased Estimates (BLUEs) are calculated separately for grain yield and plant height.

**Input file:** `trait_clean_noOut.csv`

Run the script:

```bash
python BLUEs.py
```

The script contains code for calculating BLUEs for both traits.

**Output files:**

- Grain yield: `agg_BLUE_yield.csv`
- Plant height: `agg_BLUE_plant_height.csv`

---

## 3. Data Merging

The preprocessed trait, genotype, environmental covariate, metadata, and weather datasets are merged to generate the final datasets for genomic prediction.

The merging procedure is performed separately for grain yield and plant height.

### 3.1 Grain Yield

Use the following BLUEs file:

`agg_BLUE_yield.csv`

Run the merging script:

```bash
python Merging.py
```

**Output file:** `Final_merged_yield.csv`

### 3.2 Plant Height

Use the following BLUEs file:

`agg_BLUE_plant_height.csv`

Run the merging script again using the plant height BLUEs file.

```bash
python Merging.py
```

**Output file:** `Final_merged_plant_height.csv`

---

## 4. Genomic Prediction Models

The final merged datasets are used for genomic prediction.

**Input files:**

- Grain yield: `Final_merged_yield.csv`
- Plant height: `Final_merged_plant_height.csv`

Four genomic prediction models are evaluated in this study:

1. Bayesian model
2. XGBoost model
3. Bayesian model with locus-specific degree of dominance transformation
4. XGBoost model with locus-specific degree of dominance transformation

The locus-specific degree of dominance transformation is applied only to the transformed models.

### 4.1 Bayesian Model

The Bayesian model uses the original genotype information without the locus-specific degree of dominance transformation.

### 4.2 XGBoost Model

The XGBoost model uses the original genotype information without the locus-specific degree of dominance transformation.

### 4.3 Bayesian Model with Dominance Transformation

The Bayesian transformed model incorporates the locus-specific degree of dominance transformation of genotype information.

### 4.4 XGBoost Model with Dominance Transformation

The XGBoost transformed model incorporates the locus-specific degree of dominance transformation of genotype information.

---

## 5. Cross-Validation Strategies

All four genomic prediction models are evaluated under three cross-validation strategies.

### 5.1 Leave-One-Year-Out (LOYO)

In Leave-One-Year-Out cross-validation, one year is held out as the test set, while the remaining years are used for model training.

### 5.2 Rolling-Year Validation

In Rolling-Year cross-validation, models are trained using data from previous years and evaluated on a subsequent year.

The training dataset expands as additional years become available.

### 5.3 Leave-One-Environment-Out (LOEO)

In Leave-One-Environment-Out cross-validation, one environment is held out for testing, while the remaining environments are used for model training.

### Model Implementation

The code for all four genomic prediction models and their respective cross-validation strategies is available in the corresponding model folders.

Each model is evaluated separately for grain yield and plant height using the respective merged datasets.

---

## 6. Training Set Size and Overlapping Hybrids

The training set size and the number of overlapping hybrids between the training and test sets are calculated for the year-based cross-validation strategies.

These include:

- Leave-One-Year-Out (LOYO)
- Rolling-Year Validation

Run the R script:

```bash
Rscript train_test_2025Oct.R
```

The script calculates the following information:

- Training years
- Test year
- Training set size
- Test set size
- Number of unique hybrids in the test set
- Number of overlapping hybrids between the training and test sets

These summaries are used to evaluate the composition of the training and test sets under different year-based validation strategies.

---

## 7. Summary of Output Files

### 7.1 Preprocessed Data

| Data Type | Output File |
|-----------|-------------|
| Trait data | `trait_clean_noOut.csv` |
| Genotype data | `Final_filtered_Recoded_imputed_SNP.csv` |
| Environmental covariates | `Env_clean.csv` |
| Metadata | `metadata_clean_imputed.csv` |
| Weather data | `weather_clean.csv` |

### 7.2 BLUEs

| Trait | Output File |
|-------|-------------|
| Grain yield | `agg_BLUE_yield.csv` |
| Plant height | `agg_BLUE_plant_height.csv` |

### 7.3 Final Merged Datasets

| Trait | Output File |
|-------|-------------|
| Grain yield | `Final_merged_yield.csv` |
| Plant height | `Final_merged_plant_height.csv` |

---

## 8. Workflow Summary

The overall workflow consists of the following steps:

1. Download the raw datasets from the Genomes to Fields repository.
2. Preprocess the trait, genotype, environmental covariate, metadata, and weather datasets.
3. Calculate BLUEs separately for grain yield and plant height.
4. Merge the processed datasets separately for both traits.
5. Implement the four genomic prediction models.
6. Evaluate model performance using LOYO, Rolling-Year, and LOEO cross-validation.
7. Calculate training set sizes and overlapping hybrids for year-based cross-validation strategies.

---

## 9. Citation

If you use the dataset, please cite:

Genomes To Fields (2025). *Genomes to Fields 2024 Maize Genotype by Environment Prediction Competition*. CyVerse Data Commons.

https://doi.org/10.25739/78mn-4394

---

## Author

**Manavi Shrestha**
