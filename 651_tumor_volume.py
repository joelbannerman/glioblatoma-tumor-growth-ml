# -*- coding: utf-8 -*-
# parameters
CSV_PATH = "Downloads/segmentation_volumes_by_patient.xlsx" #file path for volumes
EXCLUDE_WEEKS = ["week-000-1", "week-000-2"] # these are the weeks where the initial tumor was present, only modeling tumor size post op
KMIN = 5                   #i added this to only include patients with 5 or more weeks in the model to increase model accuracy and incorportate the lag 2
N_ESTIMATORS = 400         # Random Forest trees
TEST_SIZE = 0.2            # test size for test_train_split - these are the patients in the test group, graph outputs as well
CV_SPLITS = 10              # GroupKFold splits for grouped CV
RANDOM_STATE = 42# default parameter found online for random forest ML
OUTDIR = "/Volumes/T9/rf_outputs_notebook"  # saving graphs onto my ssd

#libraries
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#Path(OUTDIR).mkdir(parents=True, exist_ok=True)
#Path(OUTDIR, "per_patient_plots").mkdir(parents=True, exist_ok=True)
#print("Output dir:", OUTDIR)

df = pd.read_csv(CSV_PATH)

# Week to numeric
df["Week_num"] = df["Week"].astype(str).str.extract(r"(\d+)").astype(float)

# Exclude weeks
if EXCLUDE_WEEKS:
    df = df[~df["Week"].isin(EXCLUDE_WEEKS)].copy()

# Keep patients with enough distinct weeks (>=5)
week_counts = df.groupby("Patient")["Week"].nunique()
keep = week_counts[week_counts >= KMIN].index
df = df[df["Patient"].isin(keep)].copy()


#computing total volume
vol_cols_ml = ["Enhancing Core", "Necrotic Non Enhancing", "Edema Compartment"]
for i in vol_cols_ml:
    df[i] = pd.to_numeric(df[i], errors="coerce")
df["Total_Tumor_Volume_mm3"] = df[vol_cols_ml].sum(axis=1)/1000


# Sort + lags / time gaps
df = df.sort_values(["Patient", "Week_num"]).copy()
df["Lag_1_mm3"] = df.groupby("Patient")["Total_Tumor_Volume_mm3"].shift(1)
df["Lag_2_mm3"] = df.groupby("Patient")["Total_Tumor_Volume_mm3"].shift(2)
df["dWeek_1"] = df.groupby("Patient")["Week_num"].diff(1) #different in week time  (1 data point prior)
df["dWeek_2"] = df.groupby("Patient")["Week_num"].diff(2) #difference in week time (2 data points prior)

with np.errstate(divide='ignore', invalid='ignore'):
    df["log_ratio_1"] = np.log(np.where(df["Lag_1_mm3"] > 0, df["Total_Tumor_Volume_mm3"] / df["Lag_1_mm3"], 1.0))
    df["log_ratio_2"] = np.log(np.where(df["Lag_2_mm3"] > 0, df["Lag_1_mm3"] / df["Lag_2_mm3"], 1.0))

# Drop rows needing lags
req = ["Week_num", "Total_Tumor_Volume_mm3", "Lag_1_mm3", "Lag_2_mm3"]
df = df.dropna(subset=req).copy()

# Encode groups (not a feature)
enc = LabelEncoder()
df["Patient_ID"] = enc.fit_transform(df["Patient"])

feat_cols = ["Week_num", "Lag_1_mm3", "Lag_2_mm3", "dWeek_1", "dWeek_2", "log_ratio_1", "log_ratio_2"]
X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
y = df["Total_Tumor_Volume_mm3"].astype(float)
groups = df["Patient_ID"]
print("Patients:", df['Patient'].nunique(), "| Rows:", len(df))
df.head(20)

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return dict(MSE=mse, RMSE=rmse, R2=r2)

def predict():
    df_te = df.loc[test_idx, ["Patient", "Week_num", "Total_Tumor_Volume_mm3"]].copy()
    df_te["Predicted_mm3"] = y_pred
    df_te = df_te.sort_values(["Patient", "Week_num"])

    results = {}
    for patient, g in df_te.groupby("Patient"):
        first = g['Predicted_mm3'].iloc[0]
        last = g['Predicted_mm3'].iloc[-1]

        if first > last:
            results[patient] = "Stable Disease (SD) — tumor decreased"
        elif last > first:
            results[patient] = "Progressive Disease (PD) — tumor increased"
        else:
            results[patient] = "No Change"

    return results


def actual():
    df_te = df.loc[test_idx, ["Patient", "Week_num", "Total_Tumor_Volume_mm3"]].copy()
    df_te["Predicted_mm3"] = y_pred
    df_te = df_te.sort_values(["Patient", "Week_num"])

    results = {}
    for patient, g in df_te.groupby("Patient"):
        first = g['Total_Tumor_Volume_mm3'].iloc[0]
        last = g['Total_Tumor_Volume_mm3'].iloc[-1]

        if first > last:
            results[patient] = "Stable Disease (SD) — tumor decreased"
        elif last > first:
            results[patient] = "Progressive Disease (PD) — tumor increased"
        else:
            results[patient] = "No Change"

    return results

# Grouped Cross-Validation (by patient) + naive baseline
gkf = GroupKFold(n_splits=CV_SPLITS)
fold_rows = []
rf_params = dict(
    n_estimators=1200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=3,
    max_features=1.0
)
last_model = None
last_X_test = last_y_test = None

for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y.iloc[tr], y.iloc[te]

    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_tr, y_tr)
    y_hat = rf.predict(X_te)

    m = compute_metrics(y_te.values, y_hat)

    fold_rows.append({
        "fold": fold,
        **{f"RF_{k}": v for k, v in m.items()},
    })

    last_model = rf
    last_X_test, last_y_test = X_te, y_te

cv_df = pd.DataFrame(fold_rows)
avg_row = {"fold": "avg", **{c: cv_df[c].mean() for c in cv_df.columns if c != "fold"}}
cv_df = pd.concat([cv_df, pd.DataFrame([avg_row])], ignore_index=True)
display(cv_df)
#cv_df.to_csv(Path(OUTDIR, "metrics_groupcv.csv"), index=False)

# Permutation importance on the last fold
if last_model is not None:
    imp = permutation_importance(last_model, last_X_test, last_y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = pd.DataFrame({
        "feature": last_X_test.columns,
        "importance_mean": imp.importances_mean,
        "importance_std": imp.importances_std,
    }).sort_values("importance_mean", ascending=False)
display(imp_df)

#group shuffle split to keep patient groups together; either in test or train
gss = GroupShuffleSplit(n_splits=2, test_size=TEST_SIZE, random_state=RANDOM_STATE)
tr, te = next(gss.split(X, y, groups=groups))
X_tr, X_te = X.iloc[tr], X.iloc[te]
y_tr, y_te = y.iloc[tr], y.iloc[te]

rf = RandomForestRegressor(**rf_params)
rf.fit(X_tr, y_tr)
y_pred = rf.predict(X_te)

# Scatter: predicted vs actual
plt.figure(figsize=(7,6))
plt.scatter(y_te, y_pred, color='black',marker='o',alpha=0.7, label ='Ideal Fit Line')
lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
plt.plot(lims, lims, "--", linewidth=2,color='red')
plt.xlabel("Actual Total Tumor Volume (mm³)")
plt.ylabel("Predicted Total Tumor Volume (mm³)")
plt.title("Predicted vs Actual (Test patients)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Per-patient time series on train data
test_idx = X_te.index
df_te = df.loc[test_idx, ["Patient", "Week_num", "Total_Tumor_Volume_mm3"]].copy()
df_te["Predicted_mm3"] = y_pred
df_te = df_te.sort_values(["Patient", "Week_num"])

# Create output folder if it doesn't exist
os.makedirs("per_patient_plots", exist_ok=True)

for patient, g in df_te.groupby("Patient"):
    plt.figure(figsize=(9,5))
    plt.plot(g["Week_num"], g["Total_Tumor_Volume_mm3"],
             marker="o", color='black', label="Actual (mm³)")
    plt.plot(g["Week_num"], g["Predicted_mm3"],
             marker="D", color='red', label="Predicted (mm³)")

    plt.xlabel("Week (numeric)")
    plt.ylabel("Total Tumor Volume (mm³)")
    plt.title(f"Patient: {patient}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"per_patient_plots/{patient}_tumor_growth.png", dpi=300, bbox_inches='tight')

    plt.show()

    plt.close()


#testing set metrics
m_hold = compute_metrics(y_te.values, y_pred)
print("Model metrics:")
print({f"RF_{k}": v for k, v in m_hold.items()})

results = predict()
for patient, status in results.items():
    print(f"Patient {patient}: {status}")
print('----------------------------------------------------------------')
results1 = actual()
for patient, status in results1.items():
    print(f"Patient {patient}: {status}")

train_pred = rf.predict(X_tr)
test_pred = rf.predict(X_te)

print("Train R2:", r2_score(y_tr, train_pred))
print("Test R2:", r2_score(y_te, test_pred))

from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "n_estimators": [300, 500, 800, 1200],
    "max_depth": [6, 10, 14, 20, None],
    "min_samples_split": [2, 4, 6, 10],
    "min_samples_leaf": [1, 2, 3, 5, 10],
    "max_features": ["sqrt", 0.3, 0.5, 1.0]
}

gkf = GroupKFold(n_splits=5)
rf = RandomForestRegressor(random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    rf,
    param_distributions=param_grid,
    n_iter=30,
    cv=gkf.split(X, y, groups),
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2
)

search.fit(X, y)
print(search.best_params_)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

df.head(20)
df_xg = df.drop(['Enhancing Core (mL)','Necrotic Non Enhancing (mL)', 'Enhancing Core', 'Necrotic Non Enhancing', 'Edema Compartment (mL)'],axis=1)

"""**XGBOOST  MODEL**"""

from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# group-aware split (same as your RF approach)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr, te = next(gss.split(X, y, groups=groups))

X_tr, X_te = X.iloc[tr], X.iloc[te]
y_tr, y_te = y.iloc[tr], y.iloc[te]

# XGBoost model
xgb = XGBRegressor(
    n_estimators=800,
    learning_rate=0.02,
    max_depth=2,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

xgb.fit(X_tr, y_tr)

y_pred = xgb.predict(X_te)

# metrics
r2 = r2_score(y_te, y_pred)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))

print("XGBoost R²:", r2)
print("XGBoost RMSE:", rmse)

plt.figure(figsize=(7,6))
plt.scatter(y_te, y_pred, color='black',marker='o',alpha=0.7, label ='Ideal Fit Line')
lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
plt.plot(lims, lims, "--", linewidth=2,color='red')
plt.xlabel("Actual Total Tumor Volume (mm³)")
plt.ylabel("Predicted Total Tumor Volume (mm³)")
plt.title("Predicted vs Actual (Test patients)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

test_idx = X_te.index
df_te = df.loc[test_idx, ["Patient", "Week_num", "Total_Tumor_Volume_mm3"]].copy()
df_te["Predicted_mm3"] = y_pred
df_te = df_te.sort_values(["Patient", "Week_num"])

# Create output folder if it doesn't exist
for patient, g in df_te.groupby("Patient"):
    plt.figure(figsize=(9,5))
    plt.plot(g["Week_num"], g["Total_Tumor_Volume_mm3"],
             marker="o", color='black', label="Actual (cm³)")
    plt.plot(g["Week_num"], g["Predicted_mm3"],
             marker="D", color='red', label="Predicted (cm³)")

    plt.xlabel("Week (numeric)")
    plt.ylabel("Total Tumor Volume (cm³)")
    plt.title(f"Patient: {patient}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()

    plt.close()


#testing set metrics
m_hold = compute_metrics(y_te.values, y_pred)
print("Model metrics:")
print({f"RF_{k}": v for k, v in m_hold.items()})

results = predict()
for patient, status in results.items():
    print(f"Patient {patient}: {status}")
print('----------------------------------------------------------------')
results1 = actual()
for patient, status in results1.items():
    print(f"Patient {patient}: {status}")

"""****Convolutional Model****"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([Dense(128, activation='relu', input_shape= (7,)),
                    Dense(64, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(1)
])
model.compile(
    optimizer ='nadam',
    loss = 'mse',
    metrics = ['mae']
)
model.fit(
    X_tr,
    y_tr,
    epochs = 200,
    validation_split=0.2,
    verbose = 1
)

y_pred = model.predict(X_te)

print(compute_metrics(y_te,y_pred))

plt.figure(figsize=(7,7))
plt.plot(y_te, y_pred, 'bo')
plt.plot([y_te.min(), y_te.max()],
         [y_te.min(), y_te.max()], 'r--')  # identity line
plt.title('Actual Total Tumor Volume (cm3) vs Predicted')
plt.xlabel('Actual Total Tumor Volume (cm3)')
plt.ylabel('Predicted Total Tumor Volume (cm3) ')

est_idx = X_te.index
df_te = df.loc[test_idx, ["Patient", "Week_num", "Total_Tumor_Volume_mm3"]].copy()
df_te["Predicted_mm3"] = y_pred
df_te = df_te.sort_values(["Patient", "Week_num"])

# Create output folder if it doesn't exist
for patient, g in df_te.groupby("Patient"):
    plt.figure(figsize=(9,5))
    plt.plot(g["Week_num"], g["Total_Tumor_Volume_mm3"],
             marker="o", color='black', label="Actual (cm³)")
    plt.plot(g["Week_num"], g["Predicted_mm3"],
             marker="D", color='red', label="Predicted (cm³)")

    plt.xlabel("Week (numeric)")
    plt.ylabel("Total Tumor Volume (cm³)")
    plt.title(f"Patient: {patient}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()

    plt.close()


#testing set metrics
m_hold = compute_metrics(y_te.values, y_pred)
print("Model metrics:")
print({f"RF_{k}": v for k, v in m_hold.items()})



#progressive or stable
results = predict()
for patient, status in results.items():
    print(f"Patient {patient}: {status}")
print('----------------------------------------------------------------')
results1 = actual()
for patient, status in results1.items():
    print(f"Patient {patient}: {status}")

