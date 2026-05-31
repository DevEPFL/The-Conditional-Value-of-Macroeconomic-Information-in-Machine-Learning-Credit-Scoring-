"""
NOTE ON OPTBINNING LIBRARY:
If running this script on a new machine or a fresh .venv, the optbinning library 
will throw a TypeError due to a Scikit-Learn version mismatch. 

To fix it, manually edit: .venv/Lib/site-packages/optbinning/binning/metrics.py
Change: check_array(..., force_all_finite=True) 
To:     check_array(..., ensure_all_finite=True) (Lines 17 and 29)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import ks_2samp
from optbinning import BinningProcess
from imblearn.over_sampling import SMOTE
import shap
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------
# 0. Global Random Seed for Thesis Reproducibility
# ------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# 1. Load and prepare LendingClub data
# ------------------------------------------------------------
df = pd.read_csv('accepted_2007_to_2018Q4.csv', low_memory=False)
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')

# Define mature loans
mature_status = ['Fully Paid', 'Charged Off', 'Default']
df = df[df['loan_status'].isin(mature_status)].copy()

# Binary target
df['target'] = df['loan_status'].isin(['Charged Off', 'Default']).astype(int)

# Leakage columns (add any others you identified)
leakage_cols = [
    'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'last_pymnt_amnt',
    'out_prncp', 'out_prncp_inv', 'recoveries', 'collection_recovery_fee',
    'total_rec_late_fee', 'hardship_flag', 'settlement_status',
    'debt_settlement_flag', 'id', 'member_id', 'url', 'desc', 'emp_title',
    'funded_amnt', 'funded_amnt_inv', 'next_pymnt_d', 'last_pymnt_d',
    'last_credit_pull_d', 'loan_status',
    # new ones to consider:
    'hardship_amount', 'hardship_dpd', 'hardship_end_date', 'hardship_last_payment_amount', 
    'hardship_length', 'hardship_loan_status', 'hardship_payoff_balance_amount', 
    'hardship_reason', 'hardship_start_date', 'hardship_status', 'hardship_type',
    'debt_settlement_flag_date', 'settlement_amount', 'settlement_date', 'settlement_percentage',
    'settlement_term', 'total_rec_int', 'last_fico_range_high', 'last_fico_range_low',
    'pymnt_plan', 'payment_plan_start_date',
    'deferral_term', 'orig_projected_additional_accrued_interest',
    # too high cardinality or useless due to modeling choices
    'title', 'zip_code',
    # zip code is granular and has a lot of value but we are not using local macro data so we cannot use it correctly
    'policy_code', # this one is a constant column of 1 so we can remove it to free up memory
    "grade","sub_grade", "int_rate", "installment"
    ]

# ------------------------------------------------------------
# Feature Engineering: Make Employment Length Ordinal
# ------------------------------------------------------------
emp_map = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
    '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
    '8 years': 8, '9 years': 9, '10+ years': 10
}
df['emp_length'] = df['emp_length'].map(emp_map)

# ------------------------------------------------------------
# Feature Engineering: Length of Credit History
# ------------------------------------------------------------
# 1. Convert the text strings into Pandas datetime objects
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
df['sec_app_earliest_cr_line'] = pd.to_datetime(df['sec_app_earliest_cr_line'], format='%b-%Y', errors='coerce')

# 2. Calculate the number of months between their first credit line and the loan issue date
# For the primary applicant
df['credit_history_months'] = (
    (df['issue_d'].dt.year - df['earliest_cr_line'].dt.year) * 12 +
    (df['issue_d'].dt.month - df['earliest_cr_line'].dt.month)
).clip(lower=0)

# For the secondary applicant (co-signer)
df['sec_app_credit_history_months'] = (
    (df['issue_d'].dt.year - df['sec_app_earliest_cr_line'].dt.year) * 12 +
    (df['issue_d'].dt.month - df['sec_app_earliest_cr_line'].dt.month)
).clip(lower=0)

leakage_cols.extend(['earliest_cr_line', 'sec_app_earliest_cr_line'])


#drop them
df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

# ------------------------------------------------------------
# 2. Load and process macro data (no look‑ahead)
# ------------------------------------------------------------

# Load your local files
macro_monthly = pd.read_csv('monthly_fed.csv', parse_dates=['observation_date'])
macro_quarterly = pd.read_csv('quarterly_fed.csv', parse_dates=['observation_date'])

# Ensure your data is sorted chronologically before doing any rolling/shifting
macro_monthly = macro_monthly.sort_values('observation_date').reset_index(drop=True)


# ------------------------------------------------------------
# 1. CALCULATE DELTAS (Absolute Change)
# ------------------------------------------------------------
# 3, 6, and 12-month absolute changes for Fed Funds
macro_monthly['delta_fedfunds_3m'] = macro_monthly['FEDFUNDS'] - macro_monthly['FEDFUNDS'].shift(3)
macro_monthly['delta_fedfunds_6m'] = macro_monthly['FEDFUNDS'] - macro_monthly['FEDFUNDS'].shift(6)
macro_monthly['delta_fedfunds_12m'] = macro_monthly['FEDFUNDS'] - macro_monthly['FEDFUNDS'].shift(12)

# 3, 6, and 12-month absolute changes for Unemployment
macro_monthly['delta_unrate_3m'] = macro_monthly['UNRATE'] - macro_monthly['UNRATE'].shift(3)
macro_monthly['delta_unrate_6m'] = macro_monthly['UNRATE'] - macro_monthly['UNRATE'].shift(6)
macro_monthly['delta_unrate_12m'] = macro_monthly['UNRATE'] - macro_monthly['UNRATE'].shift(12)


"""
# ------------------------------------------------------------
# 2. CALCULATE SAHM RULE INDICATOR
# ------------------------------------------------------------
# Step A: 3-month moving average of unemployment
macro_monthly['UNRATE_3m_avg'] = macro_monthly['UNRATE'].rolling(window=3).mean()

# Step B: Minimum of that 3-month average over the previous 12 months
macro_monthly['UNRATE_12m_min'] = macro_monthly['UNRATE_3m_avg'].rolling(window=12).min()

# Step C: Sahm Rule binary trigger (1 if difference >= 0.50, else 0)
macro_monthly['sahm_recession_indicator'] = np.where(
    (macro_monthly['UNRATE_3m_avg'] - macro_monthly['UNRATE_12m_min']) >= 0.50, 
    1, 
    0
)

# Clean up intermediate columns used for the calculation
macro_monthly.drop(columns=['UNRATE_3m_avg', 'UNRATE_12m_min'], inplace=True)
"""

# ------------------------------------------------------------
# Explicit publication‑lag offsets (no row shifting)
# ------------------------------------------------------------
# Monthly series (e.g. Jan 1) → available for loans from Mar 1 onward
macro_monthly['observation_date'] = macro_monthly['observation_date'] + pd.DateOffset(months=2)

# Quarterly GDP (e.g. Q1 = Jan 1) → released late April, safe from May 1 onward
macro_quarterly['observation_date'] = macro_quarterly['observation_date'] + pd.DateOffset(months=4)

# Merge onto the same timeline and forward‑fill
macro = pd.merge(macro_monthly, macro_quarterly, on='observation_date', how='outer')
macro = macro.sort_values('observation_date').ffill().dropna()

"""
# ------------------------------------------------------------
# drop macro variables which are noisey
# ------------------------------------------------------------
# DROP the noisy/lagging macro variables entirely from the dataset
macro.drop(columns=['GDPC1_PC1', 'CPIAUCSL_PC1'], inplace=True, errors='ignore')
"""

# Prepare for merge with LendingClub
macro = macro.rename(columns={'observation_date': 'issue_d'})

# ------------------------------------------------------------
# Merge with LendingClub (in your main pipeline)
# ------------------------------------------------------------
# Floor loan issue date to the 1st of the month
df['issue_d'] = df['issue_d'].dt.to_period('M').dt.to_timestamp()

# Left join so no loans are lost
df = pd.merge(df, macro, on='issue_d', how='left')

# ------------------------------------------------------------
# 3. Out‑of‑time split (chronological)
# ------------------------------------------------------------

# the specific cutoff date was chosen to ensure that the training set contains 
# enough volume of loans to be able to truly learn from the data, while the test 
# set is still large enough to provide a robust evaluation.

df = df.sort_values('issue_d')
# Train on everything before June 2017
train = df[df['issue_d'] < '2017-06-01']
# Test on everything from June 2017 onwards
test  = df[df['issue_d'] >= '2017-06-01']

X_train_raw = train.drop(columns=['issue_d', 'target'])
y_train = train['target']
X_test_raw  = test.drop(columns=['issue_d', 'target'])
y_test = test['target']

# ------------------------------------------------------------
# 4. Define feature sets for the four configurations
# ------------------------------------------------------------
# List the macroeconomic column names exactly as they appear in your merged data
macro_cols = [
    'UNRATE', 'FEDFUNDS', 
    'CPIAUCSL_PC1', 'GDPC1_PC1',
    'delta_fedfunds_3m', 
    'delta_fedfunds_6m', 'delta_fedfunds_12m',
    'delta_unrate_3m', 
    'delta_unrate_6m', 'delta_unrate_12m',
    # 'sahm_recession_indicator'
]

# The idiosyncratic (borrower‑only) columns are all others
idio_cols = [c for c in X_train_raw.columns if c not in macro_cols]

# We'll loop over two feature sets: "Idio" and "Idio+Macro"
feature_sets = {
    'Idiosyncratic Only': idio_cols,
    'With Macro Variables': idio_cols + macro_cols
}

# Store final results here
results = []

saved_xgb_probs = {}

# Loop over the two feature spaces
for space_name, feat_cols in feature_sets.items():
    # Select only the desired columns
    X_tr = X_train_raw[feat_cols].copy()
    X_te = X_test_raw[feat_cols].copy()

    # ------------------------------------------------------------
    # 4A. LR pipeline: Imputation -> WOE -> SMOTE
    # ------------------------------------------------------------
    X_tr_imp = X_tr.copy()
    X_te_imp = X_te.copy()

    # Imputation
    num_cols = X_tr_imp.select_dtypes(include=np.number).columns
    cat_cols = X_tr_imp.select_dtypes(exclude=np.number).columns

    train_medians = X_tr_imp[num_cols].median()
    X_tr_imp[num_cols] = X_tr_imp[num_cols].fillna(train_medians)
    X_te_imp[num_cols] = X_te_imp[num_cols].fillna(train_medians)

    for col in cat_cols:
        train_mode = X_tr_imp[col].mode()[0]
        X_tr_imp[col] = X_tr_imp[col].fillna(train_mode)
        X_te_imp[col] = X_te_imp[col].fillna(train_mode)

    # WOE Binning
    print(f"\n--- Training LR on {space_name} ---")
    
    binning = BinningProcess(
        variable_names=list(X_tr_imp.columns),
        categorical_variables=list(cat_cols)  
    )
    binning.fit(X_tr_imp, y_train)
    X_tr_woe = binning.transform(X_tr_imp, metric='woe')
    X_te_woe = binning.transform(X_te_imp, metric='woe')

    # SMOTE
    smote_lr = SMOTE(random_state=42)
    X_tr_woe_smote, y_tr_lr = smote_lr.fit_resample(X_tr_woe, y_train)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_woe_smote, y_tr_lr)
    lr_probs = lr.predict_proba(X_te_woe)[:, 1]

    # ------------------------------------------------------------
    # 4B. XGBoost pipeline: Encode -> SMOTE -> Train
    # ------------------------------------------------------------
    X_tr_xgb = X_tr_imp.copy()
    X_te_xgb = X_te_imp.copy()

    # Encode categoricals
    cat_cols_xgb = X_tr_xgb.select_dtypes(exclude=np.number).columns
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_tr_xgb[cat_cols_xgb] = encoder.fit_transform(X_tr_xgb[cat_cols_xgb].astype(str))
    X_te_xgb[cat_cols_xgb] = encoder.transform(X_te_xgb[cat_cols_xgb].astype(str))

    # SMOTE
    smote_xgb = SMOTE(random_state=42)
    X_tr_xgb_smote, y_tr_xgb = smote_xgb.fit_resample(X_tr_xgb, y_train)

    # Train XGBoost
    print(f"Training XGBoost on {space_name}...")
    xgb = XGBClassifier(eval_metric='logloss', max_depth=5, random_state=42)
    xgb.fit(X_tr_xgb_smote, y_tr_xgb)
    xgb_probs = xgb.predict_proba(X_te_xgb)[:, 1]

    saved_xgb_probs[space_name] = xgb_probs

    if space_name == 'With Macro Variables':
        best_xgb_model = xgb
        best_X_te_xgb = X_te_xgb.copy()
        best_X_te_imp = X_te_imp.copy()

    # ------------------------------------------------------------
    # 5. Evaluate both models for this feature set
    # ------------------------------------------------------------
    def evaluate(y_true, probs, model_name):
        pr_auc = average_precision_score(y_true, probs)
        roc_auc = roc_auc_score(y_true, probs)
        g = probs[y_true == 0]
        b = probs[y_true == 1]
        ks, _ = ks_2samp(g, b)
        print(f"{model_name} — PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}, KS: {ks:.4f}")
        return {'Model': model_name, 'Feature Set': space_name,
                'PR-AUC': pr_auc, 'ROC-AUC': roc_auc, 'KS': ks}

    results.append(evaluate(y_test, lr_probs, "Logistic Regression"))
    results.append(evaluate(y_test, xgb_probs, "XGBoost"))

# ------------------------------------------------------------
# 6. Summary table
# ------------------------------------------------------------
print("\n====== Final Results Table ======")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# ------------------------------------------------------------
# 7. SHAP on the XGBoost model trained with Macro Variables
# ------------------------------------------------------------
print("\nGenerating SHAP explanations for XGBoost with Macro Variables...")

explainer = shap.TreeExplainer(best_xgb_model) # Use the saved model
shap_sample_encoded = best_X_te_xgb.sample(2000, random_state=42)
shap_vals = explainer.shap_values(shap_sample_encoded)

shap.summary_plot(shap_vals, best_X_te_imp.loc[shap_sample_encoded.index])
plt.show()

# ==============================================================================
# 8. POST-HOC ANALYSIS: Bootstrapping & Segmentation (No retraining required)
# ==============================================================================
print("\n" + "="*50)
print("RUNNING POST-HOC STATISTICAL PROOFS")
print("="*50)

# Retrieve the saved probabilities from the loop
probs_idio = saved_xgb_probs['Idiosyncratic Only']
probs_macro = saved_xgb_probs['With Macro Variables']

# Align y_test indices just to be safe
y_test_array = y_test.values 

# ---------Sanity check lengths-----------
print("Idio probs length:", len(probs_idio))
print("Macro probs length:", len(probs_macro))
print("y_test length:", len(y_test_array))

# ------------------------------------------------------------
# Test A: Bootstrapping for Statistical Significance
# ------------------------------------------------------------
print("\n--- A. Bootstrapping Test (1,000 Iterations) ---")
n_iterations = 1000
macro_wins = 0

# Set a seed so your thesis results are perfectly reproducible
np.random.seed(42)

for i in range(n_iterations):
    # Randomly sample indices with replacement
    indices = np.random.choice(len(y_test_array), size=len(y_test_array), replace=True)
    
    y_test_sample = y_test_array[indices]
    base_sample = probs_idio[indices]
    macro_sample = probs_macro[indices]
    
    base_pr = average_precision_score(y_test_sample, base_sample)
    macro_pr = average_precision_score(y_test_sample, macro_sample)
    
    if macro_pr > base_pr:
        macro_wins += 1

win_rate = (macro_wins / n_iterations) * 100
print(f"XGBoost Macro beat XGBoost Idio in {win_rate:.1f}% of bootstrapped samples.")
if win_rate >= 95.0:
    print("CONCLUSION: Macro is significantly better (p < 0.05). It is NOT noise.")
elif win_rate <= 5.0:
    print("CONCLUSION: Macro is significantly WORSE (p < 0.05). It is NOT noise.")
else:
    print("CONCLUSION: The difference is not statistically significant at the 95% level.")

# ------------------------------------------------------------
# Test B: Segmented Analysis by FICO Score
# ------------------------------------------------------------
print("\n--- B. Segmented Analysis (Who does Macro help?) ---")
# We use the raw test set to find FICO segments
test_data_with_results = X_test_raw.copy()
test_data_with_results['target'] = y_test_array
test_data_with_results['prob_idio'] = probs_idio
test_data_with_results['prob_macro'] = probs_macro

# Define Subprime (FICO < 660) and Prime (FICO >= 700)
# Adjust these column names if your fico column is named differently
fico_col = 'fico_range_low' 

subprime = test_data_with_results[test_data_with_results[fico_col] < 680]
prime = test_data_with_results[test_data_with_results[fico_col] >= 720]

def evaluate_segment(segment_df, segment_name):
    if len(segment_df) == 0:
        return
    y_true = segment_df['target']
    pr_idio = average_precision_score(y_true, segment_df['prob_idio'])
    pr_macro = average_precision_score(y_true, segment_df['prob_macro'])
    
    print(f"\nSegment: {segment_name} (N={len(segment_df)})")
    print(f"  Idio PR-AUC : {pr_idio:.4f}")
    print(f"  Macro PR-AUC: {pr_macro:.4f}")
    print(f"  Difference  : {(pr_macro - pr_idio):.4f}")

evaluate_segment(subprime, "Vulnerable Borrowers (FICO < 680)")
evaluate_segment(prime, "Safe Borrowers (FICO >= 720)")