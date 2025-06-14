import numpy as np
import scipy.io
import time
import warnings
# Updated imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# sklearn.metrics imports updated
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, make_scorer, roc_curve
from sklearn.utils import shuffle, resample # Added resample
from sklearn.feature_selection import SelectKBest, f_classif # Example feature selection
from tqdm.auto import tqdm # Import tqdm for progress bars

# Additional imports for new features
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import shap # Added for SHAP plots

# --- 1. Configuration Parameters ---
# File Paths and Data Variables
# *** IMPORTANT: Specify the variable name within the MAT file that holds the feature matrix ***
FEATURE_VAR_NAME = 'power_data' # MODIFY THIS if needed
# *** IMPORTANT: Specify the variable name for the subject/group identifier ***
SUBJECT_MAP_VAR_NAME = 'subject_file_map' # MODIFY THIS if needed
# *** IMPORTANT: Specify the variable name for the feature names (list of strings) ***
# Set to None or empty string if feature names are not available in the MAT file
FEATURE_NAMES_VAR_NAME = 'feature_names_list' # MODIFY THIS: e.g., 'feature_names_list'

# Data Splitting Method ('automatic' or 'manual_file')
SPLIT_METHOD = 'automatic' # Choose 'automatic' or 'manual_file'
# Add 'age_based' as an option for SPLIT_METHOD
# SPLIT_METHOD = 'age_based' # Example: Choose 'automatic', 'manual_file', or 'age_based'


# --- Settings for 'automatic' split ---
# Used only if SPLIT_METHOD = 'automatic'
MAT_FILE_ASD_AUTO = r"/Users/lightman/Code/Github/Code/MachineLearning/New_Epoch/DATA/ASD_TOM division BASE.mat" # Path to combined ASD data file
MAT_FILE_TD_AUTO = r"/Users/lightman/Code/Github/Code/MachineLearning/New_Epoch/DATA/TD_TOM division BASE.mat"  # Path to combined TD data file
TEST_SIZE = 0.2          # Proportion of data for test set

# --- Settings for 'manual_file' split ---
# Used only if SPLIT_METHOD = 'manual_file'
TRAIN_MAT_FILE_ASD = r"D:\Power\result\ASD_TOM division BASE.mat" # Path to TRAINING ASD data
TRAIN_MAT_FILE_TD = r"D:\Power\result\TD_TOM division BASE.mat"   # Path to TRAINING TD data
TEST_MAT_FILE_ASD = r"D:\Power\result\ASD_TOM_Rate.mat"   # Path to TESTING ASD data
TEST_MAT_FILE_TD = r"D:\Power\result\TD_TOM_Rate.mat"    # Path to TESTING TD data

# --- Settings for 'age_based' split ---
# Used only if SPLIT_METHOD = 'age_based'
# Assumes MAT files contain an 'AGE' variable (or as specified by AGE_VAR_NAME)
# with age in months for each trial/sample.
AGE_VAR_NAME = 'AGE' # MODIFY THIS: Variable name in MAT file for age data (e.g., 'AGE')
MAT_FILE_ASD_AGE_BASED = r"/Users/lightman/Code/Github/Code/MachineLearning/New_Epoch/DATA/ASD_TOM division BASE.mat"  # Path to ASD data file with age info
MAT_FILE_TD_AGE_BASED = r"/Users/lightman/Code/Github/Code/MachineLearning/New_Epoch/DATA/TD_TOM division BASE.mat"    # Path to TD data file with age info
# Define age ranges in months (inclusive)
TRAIN_AGE_RANGE = (49,132)  # Example: Train on subjects aged 36 to 72 months
TEST_AGE_RANGE = (36,48) # Example: Test on subjects aged 73 to 120 months


# Cross-Validation and Random Seed
CV_FOLDS = 5             # Number of folds for cross-validation
RANDOM_STATE = 42       # Global random seed for reproducibility
SCORING_METRIC = 'roc_auc'
SCORER = make_scorer(roc_auc_score, needs_proba=True) if SCORING_METRIC == 'roc_auc' else SCORING_METRIC

# New Configuration for Bootstrap and Visualization
N_BOOTSTRAPS = 1000
VISUALIZATION_OUTPUT_DIR = Path("ml_visualizations") # Directory to save plots and ROC params

# SHAP Plot Configuration
ENABLE_SHAP_PLOTS = True  # Set to True to generate SHAP summary plots
# Number of background samples for SHAP KernelExplainer (if used).
# Reduce if X_train is very large and SHAP calculation is too slow.
SHAP_BACKGROUND_DATA_SIZE = 2500
TOP_N_FEATURES_SHAP = 20 # Number of top features to display in SHAP summary plot


# --- 2. Algorithm Definitions and Hyperparameter Grids ---
# Define the classifiers and their parameter grids to search.
# Pipeline step names MUST be used as prefixes in the param_grid keys (e.g., 'classifier__C')
# Add or remove classifiers here to extend/reduce the search.

# Example Feature Selection configuration (tune 'k')
# If you don't want feature selection, remove 'feature_selector' from the pipeline steps below
# and remove 'feature_selector__k' from param_grids.
FEATURE_SELECTOR = SelectKBest(score_func=f_classif)
FEATURE_SELECTOR_PARAM_GRID = {
    # Select top N features.
    # Using a range allows tuning the number of features.
    # Note: 'all' is not directly supported by SelectKBest's 'k'.
    # If you want to compare *with* and *without* feature selection,
    # you might need separate pipeline definitions or more complex grid search logic.
    # Here, we tune 'k' within a range. Set a single value if you want fixed k.
    'feature_selector__k': [30,40,50,60,70,80,90,100,110,120,130,140,150,160,170], # Adjust K values based on your total features
}
# Set to True to include feature selection, False to exclude it
USE_FEATURE_SELECTION = False

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB # Example: Add Naive Bayes
from xgboost import XGBClassifier # Example: Add XGBoost (requires installation)
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

CLASSIFIERS_CONFIG = {
    # 'Linear Discriminant Analysis': {
    #     'estimator': LinearDiscriminantAnalysis(), # LDA 估计器实例
    #     'param_grid': {
    #         'classifier__solver': ['lsqr', 'eigen'],
    #         'classifier__shrinkage': [None, 'auto', 0.1, 0.3, 0.5, 0.7, 0.9]
    #         # shrinkage 参数:
    #         # None: 不使用收缩 (仅适用于 lsqr/eigen)
    #         # 'auto': 使用 Ledoit-Wolf 引理自动确定收缩强度 (仅适用于 lsqr/eigen)
    #         # float (0到1之间): 手动指定收缩强度 (仅适用于 lsqr/eigen)
    #     }
    # },
    # 'Logistic Regression': {
    #     'estimator': LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver='liblinear'),
    #     'param_grid': {
    #         # **FEATURE_SELECTOR_PARAM_GRID if USE_FEATURE_SELECTION else {}, # Combine feature selection params conditionally
    #         'classifier__C': [0.01, 0.1, 1, 10, 100],
    #         'classifier__penalty': ['l1', 'l2']
    #     }
    # },
    # 'Support Vector Machine': {
    #     'estimator': SVC(probability=True, random_state=RANDOM_STATE), # probability=True needed for roc_auc
    #     'param_grid': {
    #         # **FEATURE_SELECTOR_PARAM_GRID if USE_FEATURE_SELECTION else {},
    #         'classifier__C': [0.1, 1],
    #         'classifier__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
    #         'classifier__kernel': ['rbf', 'linear'] # 'poly' can be slow
    #     }
    # },
    # 'Random Forest': {
    #     'estimator': RandomForestClassifier(random_state=RANDOM_STATE),
    #     'param_grid': {
    #         # **FEATURE_SELECTOR_PARAM_GRID if USE_FEATURE_SELECTION else {},
    #         'classifier__n_estimators': [50, 100, 200],
    #         'classifier__max_depth': [None, 10, 20, 30],
    #         'classifier__min_samples_split': [2, 5, 10],
    #         'classifier__min_samples_leaf': [1, 3, 5]
    #     }
    # },
    # 'K-Nearest Neighbors': {
    #     'estimator': KNeighborsClassifier(),
    #     'param_grid': {
    #         # **FEATURE_SELECTOR_PARAM_GRID if USE_FEATURE_SELECTION else {},
    #         'classifier__n_neighbors': [3, 5, 7, 9, 11, 15],
    #         'classifier__weights': ['uniform', 'distance'],
    #         'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
    #     }
    # },
    # 'Gaussian Naive Bayes': {
    #     'estimator': GaussianNB(),
    #     'param_grid': {
    #         # **FEATURE_SELECTOR_PARAM_GRID if USE_FEATURE_SELECTION else {},
    #         # GaussianNB typically has few hyperparameters to tune via grid search
    #         'classifier__var_smoothing': np.logspace(0,-9, num=10)
    #     }
    # },
    'XGBoost': { # Requires xgboost library: pip install xgboost
       'estimator': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
       'param_grid': {
           # **FEATURE_SELECTOR_PARAM_GRID if USE_FEATURE_SELECTION else {},
           'classifier__n_estimators': [50, 100, 200],
           'classifier__learning_rate': [0.01, 0.1, 0.2],
           'classifier__max_depth': [3, 5, 7],
           'classifier__subsample': [0.8, 1.0],
           'classifier__colsample_bytree': [0.8, 1.0]
       }
    },
}

# Dynamically add feature selection parameters if enabled
if USE_FEATURE_SELECTION:
    for name in CLASSIFIERS_CONFIG:
        # Check if k needs adjustment based on total features later if needed
        # For now, merge the dictionaries
        CLASSIFIERS_CONFIG[name]['param_grid'] = {
            **FEATURE_SELECTOR_PARAM_GRID,
            **CLASSIFIERS_CONFIG[name]['param_grid']
        }

# --- 3. Helper Functions ---

def _load_and_validate_mat(filepath, feature_var_name, subject_map_var_name, description, age_var_name=None, feature_names_var_name=None):
    """Internal helper to load and validate a single MAT file, including subject map, optional age data, and optional feature names."""
    X = None
    subject_map = None
    age_data = None # Initialize age_data
    feature_names = None # Initialize feature_names
    try:
        mat = scipy.io.loadmat(filepath)
    except FileNotFoundError:
        print(f"Error: {description} file not found at {filepath}")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred loading {description} file {filepath}: {e}")
        return None, None, None, None

    # Load Features (X)
    try:
        X = mat[feature_var_name]
    except KeyError:
        print(f"Error: Feature variable '{feature_var_name}' not found in {description} file {filepath}.")
        print(f"Variables found: {list(k for k in mat.keys() if not k.startswith('__'))}")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred accessing feature data within {description} file {filepath}: {e}")
        return None, None, None, None

    # Validate and Reshape Features
    if not isinstance(X, np.ndarray):
        print(f"Error: Feature data in {description} file {filepath} is not a NumPy array.")
        return None, None, None, None
    if X.ndim == 1: X = X.reshape(-1, 1)
    if X.ndim > 2:
        print(f"Warning: {description} feature data ({filepath}) has {X.ndim} dimensions. Flattening features.")
        X = X.reshape(X.shape[0], -1)
    if np.isnan(X).any() or np.isinf(X).any():
        print(f"Warning: {description} feature data ({filepath}) contains NaN or Inf values. Consider cleaning.")

    # Load Subject Map
    try:
        subject_map_raw = mat[subject_map_var_name]
        # Ensure subject_map is a 1D array-like structure (e.g., list or 1D numpy array)
        # .mat files often load cell arrays as object arrays of arrays
        if isinstance(subject_map_raw, np.ndarray) and subject_map_raw.ndim == 2 and subject_map_raw.shape[0] == 1:
             subject_map_raw = subject_map_raw[0] # Handle [[a],[b]] case
        if isinstance(subject_map_raw, np.ndarray) and subject_map_raw.dtype == 'object':
             # Flatten potential nested arrays if they contain single elements
             subject_map = [item[0] if isinstance(item, np.ndarray) and item.size == 1 else item for item in subject_map_raw.flatten()]
        else:
             subject_map = subject_map_raw.flatten() # Flatten just in case

        # Convert to numpy array for consistency if needed, ensure it's 1D
        subject_map = np.array(subject_map).flatten()

    except KeyError:
        print(f"Error: Subject map variable '{subject_map_var_name}' not found in {description} file {filepath}.")
        print(f"Variables found: {list(k for k in mat.keys() if not k.startswith('__'))}")
        return X, None, None, None # Return X even if subject map fails, but signal error
    except Exception as e:
        print(f"An error occurred accessing subject map data within {description} file {filepath}: {e}")
        return X, None, None, None

    # Validate Subject Map
    if subject_map is None: # Check if loading failed above
         return X, None, None, None

    if len(subject_map) != X.shape[0]:
        print(f"Error: Number of subject entries ({len(subject_map)}) does not match number of samples ({X.shape[0]}) in {description} file {filepath}.")
        return X, None, None, None

    # Load Age Data (Optional)
    if age_var_name:
        try:
            age_data_raw = mat[age_var_name]
            if not isinstance(age_data_raw, np.ndarray):
                print(f"Warning: Age data in {description} file {filepath} ('{age_var_name}') is not a NumPy array. Age data ignored.")
            else:
                if age_data_raw.ndim == 1:
                    age_data = age_data_raw.reshape(-1, 1)
                elif age_data_raw.ndim == 2 and age_data_raw.shape[1] == 1:
                    age_data = age_data_raw
                else: # Attempt to flatten if it's a row vector or other shape that can be N x 1
                    age_data_flat = age_data_raw.flatten()
                    if len(age_data_flat) == X.shape[0]:
                        age_data = age_data_flat.reshape(-1,1)
                        print(f"Warning: Age data in {description} file {filepath} ('{age_var_name}') was reshaped to N x 1. Original shape: {age_data_raw.shape}")
                    else:
                        print(f"Warning: Age data in {description} file {filepath} ('{age_var_name}') has an incompatible shape {age_data_raw.shape} for {X.shape[0]} samples. Age data ignored.")
                        age_data = None

                if age_data is not None and age_data.shape[0] != X.shape[0]:
                    print(f"Warning: Age data rows ({age_data.shape[0]}) in {description} file {filepath} ('{age_var_name}') do not match feature data rows ({X.shape[0]}). Age data ignored.")
                    age_data = None
                elif age_data is not None and (np.isnan(age_data).any() or np.isinf(age_data).any()):
                    print(f"Warning: Age data in {description} file {filepath} ('{age_var_name}') contains NaN or Inf. Consider cleaning. Age data used as is (NaNs may cause issues).")
        except KeyError:
            print(f"Warning: Age variable '{age_var_name}' not found in {description} file {filepath}. Age data ignored.")
        except Exception as e:
            print(f"An error occurred accessing age data ('{age_var_name}') within {description} file {filepath}: {e}. Age data ignored.")

    # Load Feature Names (Optional)
    if feature_names_var_name and isinstance(feature_names_var_name, str) and feature_names_var_name.strip():
        try:
            names_raw = mat[feature_names_var_name]
            # Expecting a list/array of strings. Often loaded as object array of arrays from .mat
            if isinstance(names_raw, np.ndarray) and names_raw.dtype == 'object':
                # Handle cell arrays of strings: typically shape (1, N) or (N, 1) containing arrays
                feature_names_list = []
                for item_arr in names_raw.flatten():
                    if isinstance(item_arr, np.ndarray) and item_arr.size == 1 and isinstance(item_arr[0], str):
                        feature_names_list.append(item_arr[0])
                    elif isinstance(item_arr, str): # If already a flat list of strings in an object array
                        feature_names_list.append(item_arr)
                    else:
                        # print(f"Warning: Unexpected item type in feature names array: {type(item_arr)}. Item: {item_arr}")
                        pass # Or handle more specific cases
                feature_names = feature_names_list
            elif isinstance(names_raw, (list, np.ndarray)): # If it's already a simple list or 1D array
                feature_names = [str(fn) for fn in np.asarray(names_raw).flatten()]
            else:
                print(f"Warning: Feature names variable '{feature_names_var_name}' in {description} file {filepath} is not in a recognized list/array format. Type: {type(names_raw)}. Ignored.")

            if feature_names and len(feature_names) != X.shape[1]:
                print(f"Warning: Number of feature names ({len(feature_names)}) does not match number of features ({X.shape[1]}) in {description} file {filepath}. Feature names ignored.")
                feature_names = None
            elif feature_names:
                print(f"  Successfully loaded {len(feature_names)} feature names from '{feature_names_var_name}'.")

        except KeyError:
            print(f"Warning: Feature names variable '{feature_names_var_name}' not found in {description} file {filepath}. Feature names will be generic.")
        except Exception as e:
            print(f"An error occurred accessing feature names ('{feature_names_var_name}') within {description} file {filepath}: {e}. Feature names ignored.")
            feature_names = None

    return X, subject_map, age_data, feature_names

def load_asd_td_data(asd_file, td_file, feature_var_name, subject_map_var_name, age_var_name=None, feature_names_var_name=None):
    """
    Loads data, subject maps, optional age data, and optional feature names from combined ASD and TD MAT files.
    Performs basic validation and reshaping.
    Does NOT combine, label, or shuffle data here.
    """
    print(f"Loading data for automatic/age-based split from {asd_file} and {td_file}...")
    if age_var_name:
        print(f"  Attempting to load age data using variable: '{age_var_name}'")
    if feature_names_var_name:
        print(f"  Attempting to load feature names using variable: '{feature_names_var_name}'")

    X_asd, subject_map_asd, age_data_asd, feature_names_asd = _load_and_validate_mat(
        asd_file, feature_var_name, subject_map_var_name, "Combined ASD", age_var_name, feature_names_var_name
    )
    X_td, subject_map_td, age_data_td, feature_names_td = _load_and_validate_mat(
        td_file, feature_var_name, subject_map_var_name, "Combined TD", age_var_name, feature_names_var_name
    )

    if X_asd is None or X_td is None:
        print("Error loading feature data. Exiting.")
        return None, None, None, None, None, None, None
    if subject_map_asd is None or subject_map_td is None:
        print("Error loading subject map data. Exiting.")
        return None, None, None, None, None, None, None
    
    # If age_var_name was specified, age_data should also be present for age-based split
    if age_var_name and (age_data_asd is None or age_data_td is None):
        print(f"Error: Age data ('{age_var_name}') requested but not successfully loaded from one or both files. Exiting.")
        # Return None for age data specifically if it failed, other data might still be valid for other modes
        return X_asd, X_td, subject_map_asd, subject_map_td, None, None, None


    if X_asd.shape[1] != X_td.shape[1]:
        print(f"Error: Feature dimensions mismatch! ASD has {X_asd.shape[1]} features, TD has {X_td.shape[1]} features.")
        return None, None, None, None, None, None, None

    # Handle feature names: prefer ASD, then TD, ensure consistency if both present
    final_feature_names = None
    if feature_names_asd and feature_names_td:
        if feature_names_asd == feature_names_td:
            final_feature_names = feature_names_asd
            print("  Feature names loaded and consistent between ASD and TD files.")
        else:
            print("Warning: Feature names differ between ASD and TD files. Using names from ASD file.")
            final_feature_names = feature_names_asd
    elif feature_names_asd:
        final_feature_names = feature_names_asd
        print("  Feature names loaded from ASD file.")
    elif feature_names_td:
        final_feature_names = feature_names_td
        print("  Feature names loaded from TD file.")
    elif feature_names_var_name: # If var name was given but names weren't loaded
        print("  Feature names variable specified but names not loaded from either file.")


    print(f"Data loaded successfully:")
    print(f"  ASD samples: {X_asd.shape[0]}, Subjects: {len(np.unique(subject_map_asd))}")
    if age_data_asd is not None: print(f"  ASD age data shape: {age_data_asd.shape}")
    print(f"  TD samples: {X_td.shape[0]}, Subjects: {len(np.unique(subject_map_td))}")
    if age_data_td is not None: print(f"  TD age data shape: {age_data_td.shape}")
    print(f"  Number of features: {X_asd.shape[1]}")
    if final_feature_names: print(f"  Number of feature names loaded: {len(final_feature_names)}")


    return X_asd, X_td, subject_map_asd, subject_map_td, age_data_asd, age_data_td, final_feature_names


def load_and_split_by_age(asd_file, td_file, feature_var_name, subject_map_var_name, age_var_name,
                          train_age_range, test_age_range, random_state, feature_names_var_name=None):
    """
    Loads data, splits it into training and testing sets based on subject age ranges.
    Ensures subjects are unique to either train or test set.
    Also loads feature names if specified.
    """
    print(f"\n--- Loading and Splitting Data by Age ---")
    print(f"  Train Age Range (months): {train_age_range[0]}-{train_age_range[1]}")
    print(f"  Test Age Range (months): {test_age_range[0]}-{test_age_range[1]}")

    X_asd, X_td, sm_asd, sm_td, age_asd, age_td, feature_names = load_asd_td_data(
        asd_file, td_file, feature_var_name, subject_map_var_name, age_var_name, feature_names_var_name
    )

    if X_asd is None or age_asd is None or age_td is None: # Critical check for age data
        print("Exiting due to errors loading feature or age data for age-based split.")
        return None, None, None, None, None, None, None

    # Combine ASD and TD data
    X_all = np.vstack((X_asd, X_td))
    y_all = np.concatenate((np.ones(X_asd.shape[0], dtype=int), np.zeros(X_td.shape[0], dtype=int)))
    groups_all = np.concatenate((sm_asd, sm_td))
    ages_all_trials = np.concatenate((age_asd, age_td)).flatten() # Ensure 1D array of ages

    if len(ages_all_trials) != X_all.shape[0]:
        print(f"Error: Combined age data length ({len(ages_all_trials)}) does not match combined feature data length ({X_all.shape[0]}).")
        return None, None, None, None, None, None, None

    # Determine age for each unique subject
    subject_ages = {}
    unique_subject_ids = np.unique(groups_all)
    for subj_id in unique_subject_ids:
        # Find first trial for this subject to get their age
        # Assumes age is consistent for a subject across all their trials
        indices = np.where(groups_all == subj_id)[0]
        if len(indices) > 0:
            age_for_subj = ages_all_trials[indices[0]]
            if np.isnan(age_for_subj):
                print(f"Warning: Subject {subj_id} has NaN age. This subject will be excluded.")
                continue
            subject_ages[subj_id] = age_for_subj
            # Optional: Check for inconsistent ages for the same subject
            # if not np.all(ages_all_trials[indices] == age_for_subj):
            #     print(f"Warning: Inconsistent ages found for subject {subj_id}. Using first encountered age: {age_for_subj}.")
        else: # Should not happen if unique_subject_ids comes from groups_all
            print(f"Warning: Subject ID {subj_id} found in unique list but not in groups_all during age mapping. Skipping.")


    # Select subjects for training and testing sets based on age
    train_subjects = set()
    for subj_id, age in subject_ages.items():
        if train_age_range[0] <= age <= train_age_range[1]:
            train_subjects.add(subj_id)

    test_subjects = set()
    for subj_id, age in subject_ages.items():
        if test_age_range[0] <= age <= test_age_range[1]:
            # Ensure subject is not already in the training set
            if subj_id not in train_subjects:
                test_subjects.add(subj_id)
            else:
                print(f"Info: Subject {subj_id} (age {age}) qualifies for both train and test age ranges. Prioritizing training set. Subject excluded from test set.")


    if not train_subjects:
        print("Error: No subjects found for the training set based on the specified age range.")
        return None, None, None, None, None, None, None
    if not test_subjects:
        print("Error: No subjects found for the test set based on the specified age range (or all were assigned to train).")
        # Depending on requirements, this could be an error or a warning.
        # For now, let it proceed, but test set will be empty.
        # return None, None, None, None, None, None # Uncomment to make it an error

    # Create masks for selecting trials
    train_mask = np.array([gid in train_subjects for gid in groups_all])
    test_mask = np.array([gid in test_subjects for gid in groups_all])

    X_train, y_train, groups_train = X_all[train_mask], y_all[train_mask], groups_all[train_mask]
    X_test, y_test, groups_test = X_all[test_mask], y_all[test_mask], groups_all[test_mask]

    if X_train.shape[0] == 0:
        print("Error: Training set is empty after age-based filtering.")
        return None, None, None, None, None, None, None
    # It's possible X_test is empty if no subjects fit test criteria or all overlapped with train
    if X_test.shape[0] == 0:
        print("Warning: Test set is empty after age-based filtering. Evaluation on test set will be skipped.")


    # Shuffle the training set
    if X_train.shape[0] > 0:
        X_train, y_train, groups_train = shuffle(X_train, y_train, groups_train, random_state=random_state)

    print(f"  Age-based splitting complete:")
    print(f"    Training set: X={X_train.shape}, y={y_train.shape}, groups={groups_train.shape}")
    print(f"      Unique training subjects: {len(np.unique(groups_train)) if X_train.shape[0] > 0 else 0}")
    if X_train.shape[0] > 0: print(f"      Training class distribution: {np.bincount(y_train)}")
    print(f"    Test set: X={X_test.shape}, y={y_test.shape}, groups={groups_test.shape}")
    print(f"      Unique test subjects: {len(np.unique(groups_test)) if X_test.shape[0] > 0 else 0}")
    if X_test.shape[0] > 0: print(f"      Test class distribution: {np.bincount(y_test)}")
    
    common_subjects_final = set(np.unique(groups_train)).intersection(set(np.unique(groups_test)))
    if common_subjects_final:
        print(f"    CRITICAL WARNING: {len(common_subjects_final)} subjects found in BOTH final train and test sets: {common_subjects_final}. This indicates an issue in the splitting logic.")
    else:
        print("    Subject integrity verified: No subjects overlap between final train and test sets.")


    return X_train, X_test, y_train, y_test, groups_train, groups_test, feature_names


# Modified function to load groups as well
def load_from_manual_files(train_asd_file, train_td_file, test_asd_file, test_td_file,
                           feature_var_name, subject_map_var_name, random_state, feature_names_var_name=None):
    """
    Loads training and testing data directly from specified separate MAT files.
    Loads subject maps, labels the data (ASD=1, TD=0), and shuffles the training set.
    Returns X_train, X_test, y_train, y_test, groups_train, groups_test, and feature_names.
    Feature names are loaded from train_asd_file if specified.
    """
    print("\nLoading data from manually specified train/test files...")
    print(f"  Train ASD: {train_asd_file}")
    print(f"  Train TD: {train_td_file}")
    print(f"  Test ASD: {test_asd_file}")
    print(f"  Test TD: {test_td_file}")
    print(f"  Subject Map Variable: '{subject_map_var_name}'")
    if feature_names_var_name:
        print(f"  Feature Names Variable (from Train ASD): '{feature_names_var_name}'")


    # Load each file including subject maps and feature names from train_asd
    X_train_asd, groups_train_asd, _, feature_names = _load_and_validate_mat(
        train_asd_file, feature_var_name, subject_map_var_name, "Train ASD", feature_names_var_name=feature_names_var_name
    )
    X_train_td, groups_train_td, _, _ = _load_and_validate_mat(train_td_file, feature_var_name, subject_map_var_name, "Train TD") # No need to load feature_names again
    X_test_asd, groups_test_asd, _, _ = _load_and_validate_mat(test_asd_file, feature_var_name, subject_map_var_name, "Test ASD")
    X_test_td, groups_test_td, _, _ = _load_and_validate_mat(test_td_file, feature_var_name, subject_map_var_name, "Test TD")

    # Check if all files and subject maps loaded successfully
    if any(X is None for X in [X_train_asd, X_train_td, X_test_asd, X_test_td]):
        print("Exiting due to errors loading one or more feature data files.")
        return None, None, None, None, None, None, None
    if any(g is None for g in [groups_train_asd, groups_train_td, groups_test_asd, groups_test_td]):
        print("Exiting due to errors loading one or more subject map files.")
        return None, None, None, None, None, None, None

    # Validate feature consistency
    n_features = X_train_asd.shape[1]
    if not (X_train_td.shape[1] == n_features and X_test_asd.shape[1] == n_features and X_test_td.shape[1] == n_features):
        print("Error: Feature dimensions mismatch between the specified train/test files.")
        # ... (print feature counts) ...
        return None, None, None, None, None, None, None

    # Combine training and test sets for X
    X_train = np.vstack((X_train_asd, X_train_td))
    X_test = np.vstack((X_test_asd, X_test_td))

    # Create labels (ASD=1, TD=0)
    y_train = np.concatenate((np.ones(X_train_asd.shape[0], dtype=int),
                              np.zeros(X_train_td.shape[0], dtype=int)))
    y_test = np.concatenate((np.ones(X_test_asd.shape[0], dtype=int),
                             np.zeros(X_test_td.shape[0], dtype=int)))

    # Combine group identifiers
    groups_train = np.concatenate((groups_train_asd, groups_train_td))
    groups_test = np.concatenate((groups_test_asd, groups_test_td))

    # Shuffle ONLY the training set (X, y, and groups together)
    X_train, y_train, groups_train = shuffle(X_train, y_train, groups_train, random_state=random_state)

    print(f"  Manual file loading complete:")
    print(f"    Training set shape: X={X_train.shape}, y={y_train.shape}, groups={groups_train.shape}")
    print(f"    Test set shape: X={X_test.shape}, y={y_test.shape}, groups={groups_test.shape}")
    print(f"    Training set class distribution: {np.bincount(y_train)}")
    print(f"    Test set class distribution: {np.bincount(y_test)}")
    unique_train_groups = np.unique(groups_train)
    unique_test_groups = np.unique(groups_test)
    print(f"    Training subjects/groups: {len(unique_train_groups)}")
    print(f"    Test subjects/groups: {len(unique_test_groups)}")
    # Optional: Check for overlap (should not happen if files are distinct)
    common_manual_groups = set(unique_train_groups).intersection(set(unique_test_groups))
    if common_manual_groups:
        print(f"    Warning: {len(common_manual_groups)} subjects found in both manually specified train and test sets: {common_manual_groups}")


    # Basic validation after combining
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
         print("\nError: Manual file loading resulted in an empty training or test set.")
         return None, None, None, None, None, None, None
    if groups_train.shape[0] != X_train.shape[0] or groups_test.shape[0] != X_test.shape[0]:
         print("\nError: Group array length mismatch after loading manual files.")
         return None, None, None, None, None, None, None

    if feature_names:
        print(f"  Feature names loaded from Train ASD file: {len(feature_names)} names.")
        if len(feature_names) != X_train.shape[1]:
            print(f"  Warning: Loaded feature names count ({len(feature_names)}) does not match feature dimension ({X_train.shape[1]}). Ignoring names.")
            feature_names = None


    return X_train, X_test, y_train, y_test, groups_train, groups_test, feature_names


def train_evaluate_models(X_train, y_train, groups_train, classifiers_config, cv_folds, scoring_metric, scorer, use_feature_selection, random_state):
    """
    Trains multiple classifiers using GridSearchCV with GroupKFold cross-validation.
    Requires groups_train for CV splitting.
    """
    results = {}
    print(f"\n--- Starting Model Training & Hyperparameter Tuning ---")
    # Updated print statement for CV strategy
    print(f"Using {cv_folds}-fold GroupKFold Cross-Validation (respecting subject groups)")
    print(f"Optimizing for: {scoring_metric}")
    print(f"Feature selection enabled: {use_feature_selection}")

    # Define the cross-validation strategy using GroupKFold
    # GroupKFold ensures samples from the same group are not split across folds.
    # It does not use 'shuffle' or 'random_state'. Shuffling should happen before splitting if needed.
    # Note: Ensure groups_train is provided and matches X_train length.
    if groups_train is None:
        print("Error: groups_train is required for GroupKFold cross-validation but was not provided.")
        # Fallback or exit? For now, let's raise an error or return empty results.
        # Alternatively, could fall back to StratifiedKFold with a warning, but that violates the goal.
        raise ValueError("groups_train must be provided when using GroupKFold.")
    if len(groups_train) != X_train.shape[0]:
         raise ValueError(f"Length of groups_train ({len(groups_train)}) does not match X_train samples ({X_train.shape[0]}).")

    cv_strategy = GroupKFold(n_splits=cv_folds)

    total_start_time = time.time()

    # Wrap the loop with tqdm for a progress bar over models
    for name, config in tqdm(classifiers_config.items(), desc="Training Models"):
        model_start_time = time.time()
        # Removed print(f"\nTraining {name}...") as tqdm shows the model name

        # Create the pipeline
        pipeline_steps = []
        pipeline_steps.append(('scaler', StandardScaler())) # Always scale
        if use_feature_selection:
            # Adjust SelectKBest 'k' if requested k > number of features
            # This needs to be done carefully within GridSearchCV or beforehand.
            # A simple approach: adjust the grid search space if needed.
            max_k = X_train.shape[1]
            current_param_grid = config['param_grid'].copy() # Start with a copy
            if 'feature_selector__k' in current_param_grid:
                 original_k_values = current_param_grid['feature_selector__k']
                 # Filter k values to be <= max_k
                 valid_k_values = [k for k in original_k_values if isinstance(k, int) and k <= max_k]
                 if not valid_k_values: # If all k were > max_k, use max_k
                     valid_k_values = [max_k] if max_k > 0 else [] # Handle case where max_k is 0
                 # Update the grid for this specific model run only if necessary or values changed
                 if len(valid_k_values) < len(original_k_values) or not valid_k_values:
                     # tqdm.write(f"  Adjusted feature_selector__k for {name}: Max features is {max_k}. Using {valid_k_values}")
                     current_param_grid['feature_selector__k'] = valid_k_values
                 elif valid_k_values: # Ensure it's set if valid values exist
                     current_param_grid['feature_selector__k'] = valid_k_values
                 else: # No valid k values and max_k is 0 or less
                     tqdm.write(f"  Warning: No valid 'k' for SelectKBest for {name}. Max features {max_k}. Skipping feature selection tuning for 'k'.")
                     # Optionally remove 'feature_selector__k' from grid if empty
                     if 'feature_selector__k' in current_param_grid:
                         del current_param_grid['feature_selector__k']
                         # Need to ensure the pipeline step is still added if other FS params exist
                         # Or handle this more robustly depending on desired behavior

            # Add feature selector step only if there are valid parameters or it's intended
            # This logic might need refinement based on how FS params interact
            if valid_k_values or 'feature_selector__k' not in FEATURE_SELECTOR_PARAM_GRID: # Add if k is valid or k wasn't the only FS param
                 pipeline_steps.append(('feature_selector', FEATURE_SELECTOR))
            else: # No valid k, don't add the step or adjust pipeline
                 tqdm.write(f"  Skipping feature selector step for {name} due to invalid 'k' values.")
                 # Remove feature selector params from the grid to avoid errors
                 current_param_grid = {k: v for k, v in current_param_grid.items() if not k.startswith('feature_selector__')}

        else: # No feature selection enabled globally
            current_param_grid = config['param_grid'] # Use original grid

        pipeline_steps.append(('classifier', config['estimator']))
        pipeline = Pipeline(steps=pipeline_steps)

        # Perform Grid Search with Cross-Validation
        try:
            if not current_param_grid:
                 tqdm.write(f"\n--- Skipping {name}: Parameter grid is empty. ---")
                 results[name] = {'error': 'Empty parameter grid', 'training_time_sec': 0}
                 continue # Skip to the next model

            # n_jobs=-1 uses all available CPU cores
            # verbose=2 will print score for each fold and parameter combination,
            # providing progress within each model's grid search.
            # tqdm handles the outer loop progress.
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=current_param_grid,
                scoring=scorer, # Use the scorer object
                cv=cv_strategy, # Use GroupKFold strategy
                n_jobs=-1,
                verbose=0, # Changed from 2 to 0 to reduce excessive output with tqdm
                refit=True, # Refit the best model on the whole training set automatically
                error_score='raise' # Or use a numeric value like 0 or np.nan
            )
            # Pass groups to the fit method for GroupKFold
            grid_search.fit(X_train, y_train, groups=groups_train)

            model_time = time.time() - model_start_time
            # Use tqdm.write for messages that should persist without breaking the bar
            # Clearer separation for grid search output
            tqdm.write(f"\n--- Finished training {name} in {model_time:.2f} seconds. ---")
            tqdm.write(f"  Best CV Score ({scoring_metric}): {grid_search.best_score_:.4f}")
            tqdm.write(f"  Best Parameters: {grid_search.best_params_}")
            tqdm.write("-" * 50) # Separator after each model's details

            results[name] = {
                'best_score_cv': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'best_estimator': grid_search.best_estimator_, # The refitted pipeline
                'cv_results': grid_search.cv_results_, # Contains detailed CV results
                'training_time_sec': model_time
            }

        except Exception as e:
            model_time = time.time() - model_start_time
            tqdm.write(f"\n--- ERROR training {name} after {model_time:.2f} seconds: {e} ---")
            # Optionally log the error or store failure information
            results[name] = {
                'best_score_cv': -np.inf, # Indicate failure
                'best_params': None,
                'best_estimator': None,
                'cv_results': None,
                'error': str(e),
                'training_time_sec': model_time
            }
            tqdm.write("-" * 50) # Separator even on error
        # Suppress specific warnings from libraries if needed (e.g., convergence warnings)
        # warnings.filterwarnings('ignore', category=ConvergenceWarning)

    total_time = time.time() - total_start_time
    print(f"\n--- Completed Training and Tuning phase in {total_time:.2f} seconds ---")
    return results


# --- New Helper Functions for Bootstrap CI and Metrics ---
def _calculate_single_set_metrics(y_true_sample, y_pred_sample, y_prob_sample=None):
    """Calculates metrics for a single sample set."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true_sample, y_pred_sample)
    
    # Confusion matrix and derived metrics
    # Ensure there are predictions for both classes or handle appropriately
    # For bootstrap, samples might sometimes have only one class if small.
    unique_true = np.unique(y_true_sample)
    unique_pred = np.unique(y_pred_sample)

    if len(unique_true) < 2 and len(unique_pred) < 2 and unique_true[0] == unique_pred[0]:
        # All true and predicted are of the same single class
        if unique_true[0] == 1: # All TP
            tn, fp, fn, tp = 0, 0, 0, len(y_true_sample)
        else: # All TN
            tn, fp, fn, tp = len(y_true_sample), 0, 0, 0
    elif len(unique_true) < 2 and len(unique_pred) >=2 : # True is single class, pred is mixed
        # This case is tricky for CM based on standard ravel.
        # Fallback to calculating TP,FP,FN,TN manually or rely on sklearn's CM robustness
        # For simplicity, let sklearn handle it, but be aware of potential issues with tiny bootstrap samples.
        cm = confusion_matrix(y_true_sample, y_pred_sample, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    else: # Standard case or true is mixed
        cm = confusion_matrix(y_true_sample, y_pred_sample, labels=[0, 1])
        if cm.size == 4: # Ensure it's a 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
        else: # Handle cases where CM might not be 2x2 (e.g. only one class predicted)
              # This can happen in bootstrap samples if a class is missing.
              # Set to 0 to avoid errors, though this metric might be less reliable for such samples.
            tn, fp, fn, tp = 0,0,0,0
            if np.array_equal(unique_true, [0]) and np.array_equal(unique_pred, [0]): tn = len(y_true_sample)
            if np.array_equal(unique_true, [1]) and np.array_equal(unique_pred, [1]): tp = len(y_true_sample)


    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall for positive class
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # Precision for positive class
    # metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # If needed
    
    if y_prob_sample is not None and len(np.unique(y_true_sample)) > 1:
        try:
            metrics['auroc'] = roc_auc_score(y_true_sample, y_prob_sample)
        except ValueError: # Handles cases like only one class present in y_true_sample
            metrics['auroc'] = np.nan
    else:
        metrics['auroc'] = np.nan
        
    return metrics

def calculate_bootstrap_ci(y_true, y_pred, y_prob, n_bootstraps, random_state):
    """Calculates 95% CI for metrics using bootstrapping."""
    if len(y_true) == 0: # Handle empty test set case
        return {
            'accuracy_ci': (np.nan, np.nan), 
            'sensitivity_ci': (np.nan, np.nan), # Recall CI
            'precision_ci': (np.nan, np.nan),   # Precision CI
            'auroc_ci': (np.nan, np.nan)
        }

    np.random.seed(random_state)
    boot_metrics = {'accuracy': [], 'sensitivity': [], 'precision': [], 'auroc': []}
    n_samples = len(y_true)
    
    for _ in range(n_bootstraps):
        indices = resample(np.arange(n_samples), n_samples=n_samples, replace=True, random_state=random_state + _)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices] if y_prob is not None else None
        
        if len(y_true_boot) == 0: continue # Should not happen with replace=True if n_samples > 0

        current_metrics = _calculate_single_set_metrics(y_true_boot, y_pred_boot, y_prob_boot)
        
        for key in boot_metrics.keys():
            boot_metrics[key].append(current_metrics.get(key, np.nan))
            
    cis = {}
    for key, values in boot_metrics.items():
        # Filter out NaNs that might occur (e.g. AUROC with single class) before percentile calculation
        valid_values = [v for v in values if not np.isnan(v)]
        if not valid_values: # If all values were NaN
            cis[f'{key}_ci'] = (np.nan, np.nan)
        else:
            lower = np.percentile(valid_values, 2.5)
            upper = np.percentile(valid_values, 97.5)
            cis[f'{key}_ci'] = (lower, upper)
            
    return cis

# --- Modified evaluation function ---
def evaluate_on_test_set(best_pipeline, X_test, y_test, n_bootstraps_for_ci, random_state_for_ci):
    """Evaluates the final chosen pipeline on the independent test set, including CIs."""
    start_time = time.time()

    # Predict on the test set
    y_pred = best_pipeline.predict(X_test)
    try:
        # Probability for the positive class (ASD=1)
        y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
        has_proba = True
    except AttributeError:
        y_pred_proba = None
        has_proba = False
    except Exception: # Catch other potential errors during predict_proba
        y_pred_proba = None
        has_proba = False

    eval_time = time.time() - start_time

    # Calculate point estimate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=['TD (0)', 'ASD (1)'], output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, target_names=['TD (0)', 'ASD (1)'], zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    auc_score = None
    if has_proba and y_pred_proba is not None and len(np.unique(y_test)) > 1:
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc_score = None # Or np.nan
    
    # Calculate Bootstrap CIs
    # Ensure y_true, y_pred, y_prob are numpy arrays for bootstrapping
    y_test_np = np.array(y_test)
    y_pred_np = np.array(y_pred)
    y_pred_proba_np = np.array(y_pred_proba) if y_pred_proba is not None else None

    bootstrap_cis = calculate_bootstrap_ci(
        y_test_np, y_pred_np, y_pred_proba_np,
        n_bootstraps=n_bootstraps_for_ci,
        random_state=random_state_for_ci
    )
    
    # Extract sensitivity from classification report for point estimate
    # report_dict['ASD (1)']['recall'] is sensitivity for class 1
    sensitivity_point = report_dict.get('ASD (1)', {}).get('recall', 0.0)
    # Extract precision from classification report for point estimate
    precision_point = report_dict.get('ASD (1)', {}).get('precision', 0.0)


    return {
        'accuracy': accuracy,
        'roc_auc': auc_score,
        'sensitivity': sensitivity_point, # Point estimate for sensitivity
        'precision': precision_point,     # Point estimate for precision
        'classification_report_dict': report_dict,
        'classification_report_str': report_str,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred, # Keep for potential use
        'y_pred_proba': y_pred_proba, # Keep for potential use
        'evaluation_time_sec': eval_time,
        'has_proba': has_proba,
        **bootstrap_cis # Add CI results: accuracy_ci, sensitivity_ci, auroc_ci
    }

# --- New Visualization Functions ---
def plot_roc_curve_and_save_params(y_true, y_prob, model_name, auc_score_point, output_dir: Path, auroc_ci=None):
    """Plots ROC curve, saves it, and saves its parameters."""
    if y_prob is None or len(np.unique(y_true)) < 2:
        print(f"  Skipping ROC curve for {model_name} (no probability scores or single class in y_true).")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score_point:.2f})')
    if auroc_ci and not any(np.isnan(val) for val in auroc_ci):
        plt.plot([], [], ' ', label=f'AUC 95% CI: [{auroc_ci[0]:.2f}-{auroc_ci[1]:.2f}]')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    plot_save_path = output_dir / f"{model_name.replace(' ', '_')}_ROC_Curve.png"
    plt.savefig(plot_save_path,dpi=300)
    plt.close()
    print(f"  ROC curve for {model_name} saved to {plot_save_path}")

    # Save ROC parameters
    roc_params = {
        'model_name': model_name,
        'auc': auc_score_point,
        'auc_ci_lower': auroc_ci[0] if auroc_ci else np.nan,
        'auc_ci_upper': auroc_ci[1] if auroc_ci else np.nan,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }
    params_save_path = output_dir / "best_model_roc_params.json" # Overwrites for current best
    with open(params_save_path, 'w') as f:
        json.dump(roc_params, f, indent=4)
    print(f"  ROC parameters for {model_name} saved to {params_save_path}")


def plot_confusion_matrix_and_save(y_true, y_pred, model_name, conf_matrix, output_dir: Path):
    """Plots confusion matrix and saves it."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['TD (0)', 'ASD (1)'], yticklabels=['TD (0)', 'ASD (1)'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    plot_save_path = output_dir / f"{model_name.replace(' ', '_')}_Confusion_Matrix.png"
    plt.savefig(plot_save_path,dpi=300)
    plt.close()
    print(f"  Confusion matrix for {model_name} saved to {plot_save_path}")


# --- New SHAP Plotting Function ---
def generate_shap_summary_plot(pipeline, X_data, feature_names_original, model_name_full, output_dir: Path, top_n_features, background_data_size, random_state_shap):
    """
    Generates and saves a SHAP summary plot for the given model pipeline.
    X_data should be the training data (e.g., X_train) used to fit the pipeline.
    feature_names_original should correspond to the columns of X_data.
    """
    if not ENABLE_SHAP_PLOTS:
        print("  SHAP plot generation is disabled.")
        return

    print(f"\n  Generating SHAP summary plot for {model_name_full}...")
    try:
        model_to_explain = pipeline.named_steps['classifier']
        
        # Prepare data: apply scaling and feature selection if present in pipeline
        X_transformed = X_data.copy()
        current_feature_names = list(feature_names_original) # Make a mutable copy

        if 'scaler' in pipeline.named_steps:
            scaler = pipeline.named_steps['scaler']
            X_transformed = scaler.transform(X_transformed)
            print("    Data scaled for SHAP.")
        
        if 'feature_selector' in pipeline.named_steps:
            selector = pipeline.named_steps['feature_selector']
            # Check if selector is fitted and transform
            if hasattr(selector, 'get_support'): # For SelectKBest etc.
                X_transformed = selector.transform(X_transformed)
                selected_indices = selector.get_support(indices=True)
                current_feature_names = [current_feature_names[i] for i in selected_indices]
                print(f"    Feature selection applied for SHAP. {len(current_feature_names)} features selected.")
            else: # For selectors that might not have get_support but do transform
                X_transformed = selector.transform(X_transformed)
                print(f"    Feature transformation applied by selector. New shape: {X_transformed.shape}")
                if X_transformed.shape[1] != len(current_feature_names):
                    print(f"    Warning: Number of features after selection ({X_transformed.shape[1]}) doesn't match original names count ({len(current_feature_names)}). Using generic names for selected features.")
                    current_feature_names = [f"Selected_Feature_{i}" for i in range(X_transformed.shape[1])]


        # Choose SHAP explainer based on model type
        # For tree-based models that have native SHAP support
        if isinstance(model_to_explain, (RandomForestClassifier, XGBClassifier, LGBMClassifier)):
            explainer = shap.TreeExplainer(model_to_explain, data=X_transformed, model_output="probability" if hasattr(model_to_explain, 'predict_proba') else "raw")
            # For TreeExplainer, X_transformed can be the background data or None if model has internal background data handling
            # shap_values = explainer.shap_values(X_transformed) # This might be for all classes
            # For binary classification, often interested in SHAP values for the positive class
            # The structure of shap_values depends on the model and explainer.
            # For scikit-learn compatible models, explainer(X) often gives shap_values for each class.
            shap_values_obj = explainer(X_transformed) # This is the newer API returning Explanation object
            # If shap_values_obj is a list (multi-class), take the one for the positive class (usually index 1)
            if isinstance(shap_values_obj.values, list) and len(shap_values_obj.values) > 1:
                 shap_values_for_plot = shap_values_obj.values[1] # Assuming class 1 is positive
                 # The Explanation object might already handle this, check shap_values_obj.base_values structure
            else: # Single output or already correct structure
                 shap_values_for_plot = shap_values_obj.values


        # For Linear models
        elif isinstance(model_to_explain, (LogisticRegression, LinearDiscriminantAnalysis)):
            # KernelExplainer can also be used, but LinearExplainer is more direct
            # LinearExplainer needs a masker.
            # masker = shap.maskers.Independent(X_transformed, max_samples=min(background_data_size, X_transformed.shape[0]))
            # explainer = shap.LinearExplainer(model_to_explain, masker)
            # shap_values = explainer.shap_values(X_transformed)
            # Using KernelExplainer as a more general approach for now, can be slow
            print("    Using KernelExplainer for linear model (can be slow). Consider LinearExplainer if performance is an issue.")
            # KernelExplainer needs background data
            if X_transformed.shape[0] > background_data_size:
                background = shap.sample(X_transformed, background_data_size, random_state=random_state_shap)
            else:
                background = X_transformed
            explainer = shap.KernelExplainer(model_to_explain.predict_proba, background)
            # For KernelExplainer, shap_values for predict_proba will be a list [shap_values_class0, shap_values_class1]
            shap_values_raw = explainer.shap_values(X_transformed, nsamples='auto') # nsamples='auto' or a number
            shap_values_for_plot = shap_values_raw[1] # SHAP values for the positive class (ASD=1)

        # For other models (e.g., SVC, KNN, GaussianNB) use KernelExplainer (can be slow)
        else:
            print(f"    Using KernelExplainer for {type(model_to_explain).__name__} (can be slow).")
            if not hasattr(model_to_explain, 'predict_proba'):
                print(f"    Warning: Model {type(model_to_explain).__name__} does not have predict_proba. SHAP plot might not be meaningful for probability. Trying to use decision_function or predict.")
                # Fallback logic for models without predict_proba - this is more complex
                # For simplicity, we'll assume predict_proba or skip if not available.
                # If you need to support models without predict_proba, this part needs careful handling.
                print(f"    Skipping SHAP for {model_name_full} as predict_proba is not available and KernelExplainer needs it for probability-based explanation.")
                return

            # KernelExplainer needs background data
            if X_transformed.shape[0] > background_data_size:
                background = shap.sample(X_transformed, background_data_size, random_state=random_state_shap)
            else:
                background = X_transformed
            
            explainer = shap.KernelExplainer(model_to_explain.predict_proba, background)
            # For KernelExplainer, shap_values for predict_proba will be a list [shap_values_class0, shap_values_class1]
            shap_values_raw = explainer.shap_values(X_transformed, nsamples='auto') # nsamples='auto' or a number
            shap_values_for_plot = shap_values_raw[1] # SHAP values for the positive class (ASD=1)

        # Generate SHAP summary plot
        plt.figure()
        
        # Prepare feature names for the plot with percentages
        augmented_feature_names_for_plot = list(current_feature_names) # Default to current names
        if shap_values_for_plot is not None and current_feature_names is not None and len(current_feature_names) > 0:
            mean_abs_shap_per_feature = np.mean(np.abs(shap_values_for_plot), axis=0)
            sum_of_all_mean_abs_shap = np.sum(mean_abs_shap_per_feature)
            if sum_of_all_mean_abs_shap > 0:
                normalized_mean_abs_shap_all_features = mean_abs_shap_per_feature / sum_of_all_mean_abs_shap
            else:
                normalized_mean_abs_shap_all_features = np.zeros_like(mean_abs_shap_per_feature)

            # Create augmented names: "Feature Name (XX.XX%)"
            # These names correspond to the columns in X_transformed and shap_values_for_plot
            # shap.summary_plot will sort them based on importance.
            augmented_feature_names_for_plot = [
                f"{name} ({norm_val*100:.2f}%)" 
                for name, norm_val in zip(current_feature_names, normalized_mean_abs_shap_all_features)
            ]

        # Use the Explanation object directly if available and appropriate
        if 'shap_values_obj' in locals() and isinstance(shap_values_obj, shap.Explanation):
            # Create a new Explanation object with modified feature names if needed, or pass directly
            # For simplicity, we'll pass the augmented names to summary_plot, which should handle it.
            # If shap_values_obj.feature_names can be set, that's another option.
            # However, summary_plot's feature_names parameter usually takes precedence.
            shap.summary_plot(shap_values_obj, X_transformed, feature_names=augmented_feature_names_for_plot, max_display=top_n_features, show=False, plot_type="dot")
        else: # Fallback to passing shap_values array and X_transformed
            shap.summary_plot(shap_values_for_plot, X_transformed, feature_names=augmented_feature_names_for_plot, max_display=top_n_features, show=False, plot_type="dot")
        
        plt.title(f"SHAP Summary Plot - {model_name_full}\n(Top {top_n_features} features for predicting ASD)")
        plt.tight_layout()
        
        plot_save_path = output_dir / f"{model_name_full.replace(' ', '_')}_SHAP_Summary.png"
        plt.savefig(plot_save_path, bbox_inches='tight',dpi=300)
        plt.close()
        print(f"    SHAP summary plot saved to {plot_save_path}")

        # Calculate and print Average Normalized |SHAP| for the features shown in the plot
        if shap_values_for_plot is not None and current_feature_names is not None and len(current_feature_names) > 0:
            # Calculate mean absolute SHAP value for each feature
            mean_abs_shap_per_feature = np.mean(np.abs(shap_values_for_plot), axis=0) # Shape: (n_features,)

            # Normalize these mean absolute SHAP values globally
            # Sum of mean absolute SHAP values across all features
            sum_of_all_mean_abs_shap = np.sum(mean_abs_shap_per_feature)
            
            if sum_of_all_mean_abs_shap > 0:
                normalized_mean_abs_shap_all_features = mean_abs_shap_per_feature / sum_of_all_mean_abs_shap
            else:
                # Avoid division by zero; if sum is zero, all normalized values are zero.
                normalized_mean_abs_shap_all_features = np.zeros_like(mean_abs_shap_per_feature)

            # Determine feature order as in the plot (based on sum of abs SHAP values)
            # The summary_plot sorts features by the sum of absolute SHAP values for each feature.
            sum_abs_shap_values_per_feature = np.sum(np.abs(shap_values_for_plot), axis=0)
            
            # Get indices of features sorted by importance (descending)
            # This determines the order in which features appear on the dot plot (top to bottom)
            sorted_feature_indices_desc = np.argsort(sum_abs_shap_values_per_feature)[::-1]
            
            # Number of features to actually display and print (could be less than top_n_features if fewer features exist)
            num_features_on_plot = min(len(current_feature_names), top_n_features)
            
            if num_features_on_plot > 0:
                # Select the indices, names, and normalized values for the features that are on the plot
                indices_on_plot = sorted_feature_indices_desc[:num_features_on_plot]
                
                feature_names_on_plot = [current_feature_names[i] for i in indices_on_plot]
                normalized_values_for_plot_features = normalized_mean_abs_shap_all_features[indices_on_plot]

                print(f"    Average Normalized |SHAP| Values (Global Feature Importance Contribution - % of total) for Top {num_features_on_plot} Features (Order as in plot):")
                for i in range(num_features_on_plot):
                    print(f"      {i+1}. {feature_names_on_plot[i]}: {normalized_values_for_plot_features[i]*100:.2f}%")
            else:
                print("    No features available to display SHAP importance values.")

            # Calculate and print sum of normalized |SHAP| values per frequency band
            band_shap_sums = {"Theta": 0.0, "Alpha": 0.0, "Beta": 0.0, "Other": 0.0}
            if normalized_mean_abs_shap_all_features is not None and len(current_feature_names) == len(normalized_mean_abs_shap_all_features):
                for i, feature_name in enumerate(current_feature_names):
                    norm_shap_val = normalized_mean_abs_shap_all_features[i]
                    name_lower = feature_name.lower()
                    if "theta" in name_lower:
                        band_shap_sums["Theta"] += norm_shap_val
                    elif "alpha" in name_lower:
                        band_shap_sums["Alpha"] += norm_shap_val
                    elif "beta" in name_lower:
                        band_shap_sums["Beta"] += norm_shap_val
                    else:
                        band_shap_sums["Other"] += norm_shap_val
                
                print("\n    Sum of Normalized |SHAP| Values by Frequency Band:")
                for band, total_shap in band_shap_sums.items():
                    if total_shap > 0 or band in ["Theta", "Alpha", "Beta"]: # Print even if zero for main bands
                        print(f"      {band}: {total_shap*100:.2f}%")
            else:
                print("    Could not calculate band-specific SHAP sums due to mismatch or missing data.")

        elif len(current_feature_names) == 0:
            print("    Skipping Average Normalized |SHAP| calculation as there are no features.")


    except Exception as e:
        print(f"    Error generating SHAP plot for {model_name_full}: {e}")
        import traceback
        traceback.print_exc()


# --- 4. Main Execution ---
if __name__ == "__main__":
    print("--- Machine Learning Pipeline for ASD/TD Classification ---")
    # Create visualization output directory
    VISUALIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Visualizations will be saved to: {VISUALIZATION_OUTPUT_DIR.resolve()}")

    print(f"Configuration:")
    print(f"  Feature Variable: '{FEATURE_VAR_NAME}'")
    print(f"  Subject Map Variable: '{SUBJECT_MAP_VAR_NAME}'")
    print(f"  Split Method: {SPLIT_METHOD}")

    if SPLIT_METHOD == 'automatic':
        print(f"  ASD File (Combined): {MAT_FILE_ASD_AUTO}")
        print(f"  TD File (Combined): {MAT_FILE_TD_AUTO}")
        print(f"  Test Set Size: {TEST_SIZE:.2f}")
    elif SPLIT_METHOD == 'manual_file':
        print(f"  Train ASD File: {TRAIN_MAT_FILE_ASD}")
        print(f"  Train TD File: {TRAIN_MAT_FILE_TD}")
        print(f"  Test ASD File: {TEST_MAT_FILE_ASD}")
        print(f"  Test TD File: {TEST_MAT_FILE_TD}")
    elif SPLIT_METHOD == 'age_based':
        print(f"  ASD File (Age-based): {MAT_FILE_ASD_AGE_BASED}")
        print(f"  TD File (Age-based): {MAT_FILE_TD_AGE_BASED}")
        print(f"  Age Variable Name: '{AGE_VAR_NAME}'")
        print(f"  Training Age Range (months): {TRAIN_AGE_RANGE[0]}-{TRAIN_AGE_RANGE[1]}")
        print(f"  Test Age Range (months): {TEST_AGE_RANGE[0]}-{TEST_AGE_RANGE[1]}")
    else:
        print(f"Error: Invalid SPLIT_METHOD '{SPLIT_METHOD}'. Choose 'automatic', 'manual_file', or 'age_based'.")
        exit()

    print(f"  CV Folds: {CV_FOLDS}")
    print(f"  Random State: {RANDOM_STATE}")
    print(f"  Scoring Metric for Optimization: {SCORING_METRIC}")
    print(f"  Feature Selection Enabled: {USE_FEATURE_SELECTION}")
    print(f"  SHAP Plots Enabled: {ENABLE_SHAP_PLOTS}")
    if ENABLE_SHAP_PLOTS:
        print(f"    Feature Names Variable: '{FEATURE_NAMES_VAR_NAME}'")
        print(f"    SHAP Top N Features: {TOP_N_FEATURES_SHAP}")
        print(f"    SHAP Background Data Size: {SHAP_BACKGROUND_DATA_SIZE}")
    print("-" * 60)

    # Initialize variables, including groups
    X_train, X_test, y_train, y_test = None, None, None, None
    groups_train, groups_test = None, None # Initialize group variables
    feature_names = None # Initialize feature names

    # 1. Load and Split Data based on SPLIT_METHOD
    if SPLIT_METHOD == 'manual_file':
        try:
            # Update call to receive groups and feature_names
            X_train, X_test, y_train, y_test, groups_train, groups_test, feature_names = load_from_manual_files(
                TRAIN_MAT_FILE_ASD, TRAIN_MAT_FILE_TD,
                TEST_MAT_FILE_ASD, TEST_MAT_FILE_TD,
                FEATURE_VAR_NAME, SUBJECT_MAP_VAR_NAME, # Pass subject map var name
                RANDOM_STATE, FEATURE_NAMES_VAR_NAME # Pass feature names var name
            )
            if X_train is None:
                 print("\nExiting due to errors during manual file loading.")
                 exit()
        except Exception as e:
            print(f"\nAn unexpected error occurred during manual file loading: {e}")
            exit()

    elif SPLIT_METHOD == 'automatic':
        # Load data and subject maps
        X_asd, X_td, subject_map_asd, subject_map_td, _, _, feature_names = load_asd_td_data(
            MAT_FILE_ASD_AUTO, MAT_FILE_TD_AUTO, FEATURE_VAR_NAME, SUBJECT_MAP_VAR_NAME, age_var_name=None, feature_names_var_name=FEATURE_NAMES_VAR_NAME
        )
        if X_asd is None or X_td is None or subject_map_asd is None or subject_map_td is None:
            print("\nExiting due to data or subject map loading errors.")
            exit()

        print(f"\nCombining data and subject maps...")
        X = np.vstack((X_asd, X_td))
        y_asd = np.ones(X_asd.shape[0], dtype=int)
        y_td = np.zeros(X_td.shape[0], dtype=int)
        y = np.concatenate((y_asd, y_td))
        # Combine subject maps into a single 1D array (groups)
        groups = np.concatenate((subject_map_asd, subject_map_td))

        print(f"  Combined data shape: X={X.shape}, y={y.shape}, groups={groups.shape}")
        print(f"  Combined class distribution: {np.bincount(y)}")
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        print(f"  Total unique subjects/groups found: {len(unique_groups)}")
        # Optional: print min/max trails per subject
        # print(f"  Min/Max trails per subject: {np.min(group_counts)} / {np.max(group_counts)}")


        print(f"\nSplitting data using GroupShuffleSplit (Test Size: {TEST_SIZE:.0%})...")
        print(f"Ensuring subjects remain entirely in train or test set.")
        try:
            # Use GroupShuffleSplit for a single split respecting groups
            gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            # Get the indices for the train and test sets
            train_idx, test_idx = next(gss.split(X, y, groups=groups))

            # Create the train and test sets using the indices
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx] # Keep track of groups in splits

            # Validation: Check if any subject is in both sets
            train_subjects = set(np.unique(groups_train))
            test_subjects = set(np.unique(groups_test))
            common_subjects = train_subjects.intersection(test_subjects)

            if common_subjects:
                 print(f"\n--- WARNING: Group split failed! {len(common_subjects)} subjects found in both train and test sets: {common_subjects} ---")
                 # This shouldn't happen with GroupShuffleSplit but good to check.
            else:
                 print("  Group integrity verified: No subjects overlap between train and test sets.")


            print(f"  Split complete:")
            print(f"    Training set shape: X={X_train.shape}, y={y_train.shape}")
            print(f"    Test set shape: X={X_test.shape}, y={y_test.shape}")
            print(f"    Training subjects: {len(train_subjects)}")
            print(f"    Test subjects: {len(test_subjects)}")
            print(f"    Training set class distribution: {np.bincount(y_train)}")
            print(f"    Test set class distribution: {np.bincount(y_test)}")
            # Note: GroupShuffleSplit does not guarantee stratification by y.
            # Class distribution might be slightly different from the original TEST_SIZE ratio.

        except ValueError as e:
             print(f"\nError during GroupShuffleSplit: {e}")
             print("This might happen if test_size is too large/small relative to the number of groups,")
             print("or if a group is too large to fit entirely in train or test.")
             exit()
        except Exception as e:
            print(f"\nAn unexpected error occurred during automatic group splitting: {e}")
            exit()
    
    elif SPLIT_METHOD == 'age_based':
        try:
            X_train, X_test, y_train, y_test, groups_train, groups_test, feature_names = load_and_split_by_age(
                MAT_FILE_ASD_AGE_BASED, MAT_FILE_TD_AGE_BASED,
                FEATURE_VAR_NAME, SUBJECT_MAP_VAR_NAME, AGE_VAR_NAME,
                TRAIN_AGE_RANGE, TEST_AGE_RANGE,
                               RANDOM_STATE, FEATURE_NAMES_VAR_NAME # Pass feature names var name
            )
            if X_train is None: # load_and_split_by_age will print specific errors
                 print("\nExiting due to errors during age-based data loading or splitting.")
                 exit()
        except Exception as e:
            print(f"\nAn unexpected error occurred during age-based splitting: {e}")
            exit()


    if X_train is None or X_test is None or y_train is None or y_test is None:
        print("\nData loading/splitting failed. Cannot proceed to training.")
        exit()
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("\nError: Training or test set is empty after splitting.")
        exit()
    # Add check for groups_train needed for CV
    if groups_train is None:
        print("\nError: Training group information (groups_train) is missing. Cannot proceed with GroupKFold CV.")
        exit()

    # Generate default feature names if not loaded
    if feature_names is None and X_train is not None:
        print(f"Feature names not loaded. Generating default names: 'Feature 0' to 'Feature {X_train.shape[1]-1}'.")
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
    elif feature_names is not None and X_train is not None and len(feature_names) != X_train.shape[1]:
        print(f"Warning: Loaded feature names count ({len(feature_names)}) mismatches X_train columns ({X_train.shape[1]}). Using default names.")
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]


    # 3. Train Models and Find Best Hyperparameters using Cross-Validation
    # Pass groups_train to the training function
    cv_results = train_evaluate_models(
        X_train, y_train, groups_train, # Pass groups_train here
        CLASSIFIERS_CONFIG, CV_FOLDS, SCORING_METRIC, SCORER, USE_FEATURE_SELECTION, RANDOM_STATE
    )

    # 4. Select the Best Model based on CV Performance
    best_model_name_cv = None
    best_cv_score = -np.inf

    print("\n--- Cross-Validation Results Summary ---")
    print(f"{'Model':<25} | {'Best CV Score (' + SCORING_METRIC + ')':<25} | {'Training Time (s)':<18}")
    print("-" * 70)
    valid_results = {name: res for name, res in cv_results.items() if res.get('best_estimator') is not None}

    if not valid_results:
        print("Error: No models were successfully trained. Check configurations and data.")
        exit()

    sorted_cv_results = sorted(valid_results.items(), key=lambda item: item[1]['best_score_cv'], reverse=True)

    for name, result in sorted_cv_results:
         print(f"{name:<25} | {result['best_score_cv']:.4f}{' ':<20} | {result['training_time_sec']:.2f}")
         if result['best_score_cv'] > best_cv_score:
             best_cv_score = result['best_score_cv']
             best_model_name_cv = name

    print("-" * 70)
    print(f"\nBest Model based on CV ({SCORING_METRIC}): {best_model_name_cv}")
    print(f"Best CV Score: {best_cv_score:.4f}")
    if best_model_name_cv:
        print(f"Best Parameters found via CV: {valid_results[best_model_name_cv]['best_params']}")
    print("-" * 60)


    # 5. Evaluate ALL Models on the Independent Test Set
    print("\n--- Evaluating All Trained Models on Independent Test Set ---")
    test_set_performances = {}
    evaluation_start_time = time.time()

    if X_test.shape[0] == 0:
        print("Test set is empty. Skipping evaluation on test set.")
    else:
        # Ensure y_test is 1D array for consistency
        y_test_eval = np.array(y_test).ravel()
        for name, result in tqdm(valid_results.items(), desc="Evaluating on Test Set"):
            pipeline = result['best_estimator']
            if pipeline:
                try:
                    # Pass N_BOOTSTRAPS and RANDOM_STATE to evaluate_on_test_set
                    test_metrics = evaluate_on_test_set(pipeline, X_test, y_test_eval, N_BOOTSTRAPS, RANDOM_STATE)
                    test_set_performances[name] = test_metrics
                except Exception as e:
                    tqdm.write(f"\n--- ERROR evaluating {name} on test set: {e} ---")
                    test_set_performances[name] = {'error': str(e)}
            else:
                 tqdm.write(f"\n--- Skipping evaluation for {name} (no valid estimator) ---")

    evaluation_total_time = time.time() - evaluation_start_time
    print(f"\n--- Completed Test Set Evaluation for {len(test_set_performances)} models in {evaluation_total_time:.2f} seconds ---")

    # 6. Print Summary Table of Test Set Performance (Updated with CIs)
    print("\n--- Test Set Performance Summary (with 95% CIs) ---")
    if not test_set_performances and X_test.shape[0] == 0:
        print("No test set evaluation performed as the test set was empty.")
    elif not test_set_performances:
        print("No models were successfully evaluated on the test set.")
    else:
        sorted_test_results = sorted(
            test_set_performances.items(),
            key=lambda item: item[1].get('roc_auc', -1) if isinstance(item[1], dict) and 'error' not in item[1] else -1,
            reverse=True
        )

        header = f"{'Model':<25} | {'Accuracy (CI)':<25} | {'AUC (CI)':<25} | {'Sensitivity (CI)':<25} | {'Precision (CI)':<25} | {'Eval Time (s)':<15}"
        print(header)
        print("-" * len(header))
        
        for name, metrics in sorted_test_results:
            if isinstance(metrics, dict) and 'error' in metrics:
                print(f"{name:<25} | {'ERROR':<25} | {'ERROR':<25} | {'ERROR':<25} | {'ERROR':<25} | {'N/A':<15}")
            elif isinstance(metrics, dict):
                acc_ci = metrics.get('accuracy_ci', (np.nan, np.nan))
                auc_ci = metrics.get('auroc_ci', (np.nan, np.nan))
                sens_ci = metrics.get('sensitivity_ci', (np.nan, np.nan)) # Recall CI
                prec_ci = metrics.get('precision_ci', (np.nan, np.nan)) # Precision CI

                acc_str = f"{metrics.get('accuracy', np.nan):.3f} ({acc_ci[0]:.3f}-{acc_ci[1]:.3f})"
                auc_val = metrics.get('roc_auc')
                auc_str = "N/A"
                if auc_val is not None and not np.isnan(auc_val):
                    auc_str = f"{auc_val:.3f} ({auc_ci[0]:.3f}-{auc_ci[1]:.3f})"
                elif auc_val is None or np.isnan(auc_val): # If point AUC is NaN/None, CI is also likely NaN
                    auc_str = f"N/A ({auc_ci[0]:.3f}-{auc_ci[1]:.3f})"


                sens_val = metrics.get('sensitivity', np.nan) # Using point estimate from report
                sens_str = f"{sens_val:.3f} ({sens_ci[0]:.3f}-{sens_ci[1]:.3f})"

                prec_val = metrics.get('precision', np.nan) # Using point estimate from report
                prec_str = f"{prec_val:.3f} ({prec_ci[0]:.3f}-{prec_ci[1]:.3f})"
                
                eval_time_str = f"{metrics.get('evaluation_time_sec', 0):.2f}"
                print(f"{name:<25} | {acc_str:<25} | {auc_str:<25} | {sens_str:<25} | {prec_str:<25} | {eval_time_str:<15}")
            else:
                 print(f"{name:<25} | {'Unknown':<25} | {'Unknown':<25} | {'Unknown':<25} | {'Unknown':<25} | {'Unknown':<15}")
        print("-" * len(header))

    # 7. Detailed Report and Visualizations for the Best Model on Test Set
    if sorted_test_results: # Check if there are any results
        best_test_model_name = sorted_test_results[0][0]
        best_test_metrics = sorted_test_results[0][1]
        best_model_pipeline_for_shap = valid_results.get(best_test_model_name, {}).get('best_estimator')

        if isinstance(best_test_metrics, dict) and 'error' not in best_test_metrics:
            print(f"\n--- Detailed Report & Visualizations for Best Model on Test Set ({best_test_model_name}, based on Test AUC) ---")
            
            acc_ci = best_test_metrics.get('accuracy_ci', (np.nan, np.nan))
            auc_ci = best_test_metrics.get('auroc_ci', (np.nan, np.nan))
            sens_ci = best_test_metrics.get('sensitivity_ci', (np.nan, np.nan))
            prec_ci = best_test_metrics.get('precision_ci', (np.nan, np.nan))

            print(f"  Test Accuracy: {best_test_metrics['accuracy']:.4f} (95% CI: {acc_ci[0]:.4f}-{acc_ci[1]:.4f})")
            
            auc_val = best_test_metrics.get('roc_auc')
            if auc_val is not None and not np.isnan(auc_val):
                print(f"  Test AUC Score: {auc_val:.4f} (95% CI: {auc_ci[0]:.4f}-{auc_ci[1]:.4f})")
            else:
                print(f"  Test AUC Score: N/A (95% CI: {auc_ci[0]:.4f}-{auc_ci[1]:.4f}) if calculable")

            sens_val = best_test_metrics.get('sensitivity', np.nan)
            print(f"  Test Sensitivity (Recall): {sens_val:.4f} (95% CI: {sens_ci[0]:.4f}-{sens_ci[1]:.4f})")

            prec_val = best_test_metrics.get('precision', np.nan)
            print(f"  Test Precision: {prec_val:.4f} (95% CI: {prec_ci[0]:.4f}-{prec_ci[1]:.4f})")

            print("\nClassification Report:\n", best_test_metrics['classification_report_str'])
            conf_matrix = best_test_metrics['confusion_matrix']
            print("\nConfusion Matrix:")
            print("        Predicted TD | Predicted ASD")
            print(f"Actual TD |   {conf_matrix[0, 0]:<10} | {conf_matrix[0, 1]:<10}")
            print(f"Actual ASD|   {conf_matrix[1, 0]:<10} | {conf_matrix[1, 1]:<10}")

            # Generate and save visualizations for the best model
            print(f"\n  Generating visualizations for {best_test_model_name}...")
            plot_roc_curve_and_save_params(
                y_true=np.array(y_test).ravel(), # Ensure y_test is 1D
                y_prob=best_test_metrics['y_pred_proba'],
                model_name=best_test_model_name,
                auc_score_point=best_test_metrics.get('roc_auc', np.nan),
                auroc_ci=best_test_metrics.get('auroc_ci'),
                output_dir=VISUALIZATION_OUTPUT_DIR
            )
            plot_confusion_matrix_and_save(
                y_true=np.array(y_test).ravel(), # Ensure y_test is 1D
                y_pred=best_test_metrics['y_pred'],
                model_name=best_test_model_name,
                conf_matrix=conf_matrix,
                output_dir=VISUALIZATION_OUTPUT_DIR
            )

            # Generate SHAP summary plot for the best model using X_train
            if ENABLE_SHAP_PLOTS and best_model_pipeline_for_shap and X_train is not None and feature_names is not None:
                generate_shap_summary_plot(
                    pipeline=best_model_pipeline_for_shap,
                    X_data=X_train, # Use training data for explaining what model learned
                    feature_names_original=feature_names,
                    model_name_full=best_test_model_name,
                    output_dir=VISUALIZATION_OUTPUT_DIR,
                    top_n_features=TOP_N_FEATURES_SHAP,
                    background_data_size=SHAP_BACKGROUND_DATA_SIZE,
                    random_state_shap=RANDOM_STATE # For reproducibility of sampling in KernelExplainer
                )
            elif ENABLE_SHAP_PLOTS:
                print("  Skipping SHAP plot generation: Best model pipeline, X_train, or feature_names not available.")

    elif X_test.shape[0] > 0 and not test_set_performances:
        print("\nNo models were successfully evaluated to determine the best on the test set.")
    elif X_test.shape[0] == 0:
        print("\nDetailed report and visualizations for test set skipped as test set was empty.")


    print("\n--- Pipeline Execution Finished ---")
