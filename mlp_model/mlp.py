# R4 — Tabular MLPs, simplified
# Diagnosis (all) → Severity (only appy) → Management (only appy)
# - One hierarchical TEST split
# - Train/Val from the rest
# - Impute → Scale -on TRAIN data only
# - MLP (Dense → BN → Dropout), Adam, callbacks: ES + ReduceLROnPlateau
# - τ sweep for Diagnosis threshold
# - Eval: Dx on test; Sev & Mgmt on TRUE appy rows
# - Save to ./mlp_files/


import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, roc_auc_score, average_precision_score,
)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Config -----
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "processed_data" / "dataset_tabular.csv"
TEST_SIZE = 0.20
VAL_SIZE  = 0.20
TAU_GRID  = np.arange(0.70, 0.79, 0.01)  # 0.70 ... 0.78

OUT_DIR = ROOT / "mlp_files"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Load and set up targets -----
df = pd.read_csv(DATA_PATH).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# Diagnosis → label encode; keep index of "appendicitis"
le_diag = LabelEncoder()
y_diag_all = le_diag.fit_transform(df["Diagnosis"].astype(str))
diag_classes = list(le_diag.classes_)
APPY_ID = int(np.where(np.array(diag_classes) == "appendicitis")[0][0]) if "appendicitis" in diag_classes else 1

# Management → 1 if "primary surgical", else 0
mgmt_str = df["Management"].astype(str).str.lower().str.strip()
y_mgmt_bin_all = (mgmt_str == "primary surgical").astype(np.float32).values
mgmt_names = ["conservative(0)", "primary_surgical(1)"]

# Severity (only for true appy): 1 if "complicated", 0 if "uncomplicated"; NaN otherwise
sev_str = df["Severity"].astype(str).str.lower().str.strip()
is_appy_all = (y_diag_all == APPY_ID)
y_sev_bin_all = np.full(len(df), np.nan, dtype=np.float32)
y_sev_bin_all[is_appy_all] = (sev_str[is_appy_all] == "complicated").astype(np.float32)

# Numeric features only; drop any target cols if they slipped in
X_all = df.select_dtypes(include=[np.number]).copy().astype(np.float32)
for col in ["Diagnosis","Management","Severity"]:
    if col in X_all.columns:
        X_all.drop(columns=[col], inplace=True, errors="ignore")

print(f"Shapes — X: {X_all.shape}  y_diag: {y_diag_all.shape}  y_mgmt: {y_mgmt_bin_all.shape}")


# Hierarchical test split -----
def make_hierarchical_test_split(y_diag, y_mgmt, test_size=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y_diag))
    idx_appy   = idx[y_diag == APPY_ID]
    idx_nonapp = idx[y_diag != APPY_ID]

    # Appy: stratify by Management - 20% for test
    n_test_appy = int(round(test_size * idx_appy.size))
    if n_test_appy > 0 and idx_appy.size > 0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_test_appy, random_state=seed)
        tr_sub, te_sub = next(sss.split(np.zeros((idx_appy.size,1)), y_mgmt[idx_appy]))
        idx_tst_appy = idx_appy[te_sub]
    else:
        idx_tst_appy = np.array([], dtype=int)

    # Non-appy: random sample - 20% for test
    n_test_non = int(round(test_size * idx_nonapp.size))
    idx_tst_non = rng.choice(idx_nonapp, size=n_test_non, replace=False) if n_test_non > 0 else np.array([], dtype=int)

    # Final test set - combine appy + non-appy
    idx_tst = np.sort(np.concatenate([idx_tst_appy, idx_tst_non]))
    mask_tst = np.zeros(len(y_diag), dtype=bool); mask_tst[idx_tst] = True
    idx_trv = idx[~mask_tst]  # complement is TrainVal pool
    return idx_trv, idx_tst

# Build TEST once; TRAIN/VAL come from the remainder (no leakage)
idx_trv, idx_tst = make_hierarchical_test_split(y_diag_all, y_mgmt_bin_all, TEST_SIZE, SEED)

X_trv, X_tst = X_all.iloc[idx_trv], X_all.iloc[idx_tst]
y_diag_trv, y_diag_tst = y_diag_all[idx_trv], y_diag_all[idx_tst]
y_mgmt_trv, y_mgmt_tst = y_mgmt_bin_all[idx_trv], y_mgmt_bin_all[idx_tst]
y_sev_trv,  y_sev_tst  = y_sev_bin_all[idx_trv],  y_sev_bin_all[idx_tst]

# Train/Val set (stratify by Diagnosis) - val is 20% of TrainVal set
X_tr, X_va, y_diag_tr, y_diag_va, y_mgmt_tr, y_mgmt_va, y_sev_tr, y_sev_va = train_test_split(
    X_trv, y_diag_trv, y_mgmt_trv, y_sev_trv,
    test_size=VAL_SIZE, random_state=SEED, stratify=y_diag_trv
)

print("\n[Split] Diagnosis counts — TrainVal:", np.bincount(y_diag_trv),
      " Test:", np.bincount(y_diag_tst))
if (y_diag_tst == APPY_ID).any():
    print("[Split] Test (APPY) Management counts:", np.bincount(y_mgmt_tst[y_diag_tst==APPY_ID].astype(int)))


# Preprocess: impute → scale (fit on TRAIN only) -----
imputer = SimpleImputer(strategy="median")     # handle missing values with median
X_tr_imp = imputer.fit_transform(X_tr)         # fit on TRAIN
X_va_imp = imputer.transform(X_va)             # apply to VAL
X_tst_imp = imputer.transform(X_tst)           # apply to TEST
X_trv_imp = imputer.transform(X_trv)           # apply to TrainVal pool

scaler = StandardScaler().fit(X_tr_imp)        # scale using TRAIN stats only
Xs_tr  = scaler.transform(X_tr_imp)
Xs_va  = scaler.transform(X_va_imp)
Xs_tst = scaler.transform(X_tst_imp)
Xs_trv = scaler.transform(X_trv_imp)

def finite(a):
    # replace NaN/inf with 0 so the model never sees invalid numbers
    a = np.where(np.isfinite(a), a, np.nan)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
Xs_tr, Xs_va, Xs_tst, Xs_trv = map(finite, [Xs_tr, Xs_va, Xs_tst, Xs_trv])

# Quick masks for hierarchy (handy slices)
is_appy_trv = (y_diag_trv == APPY_ID)
is_appy_tr  = (y_diag_tr  == APPY_ID)
is_appy_va  = (y_diag_va  == APPY_ID)
is_appy_tst = (y_diag_tst == APPY_ID)

# MLP build -----
def build_mlp_binary(input_dim, l2=1e-4, pdrop=0.2):
    inp = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=keras.regularizers.l2(l2))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(pdrop)(x)
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=keras.regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(pdrop)(x)
    out = layers.Dense(1, activation="sigmoid")(x)  # binary prob
    return keras.Model(inp, out)

def callbacks_simple():
    return [
        keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True,
                                      monitor="val_auc", mode="max"),   # stop when val AUC stops improving
        keras.callbacks.ReduceLROnPlateau(patience=6, factor=0.5, verbose=1),  # lower LR on plateaus
    ]


# Stage 1 — Diagnosis (all rows) -----
diag = build_mlp_binary(Xs_tr.shape[1])  # new model
diag.compile(optimizer=keras.optimizers.Adam(1e-3),
             loss="binary_crossentropy",
             metrics=["accuracy", keras.metrics.AUC(name="auc")])

# labels for TRAIN/VAL (1 iff Diagnosis == appendicitis)
y_diag_tr_bin = (y_diag_tr == APPY_ID).astype(np.float32)
y_diag_va_bin = (y_diag_va == APPY_ID).astype(np.float32)

# optional reweighting for class imbalance
cw_diag = compute_class_weight("balanced", classes=np.array([0,1]), y=y_diag_tr_bin.astype(int))
cw_diag = {0: cw_diag[0], 1: cw_diag[1]}

# train on TRAIN, monitor on VAL
diag.fit(Xs_tr, y_diag_tr_bin,
         validation_data=(Xs_va, y_diag_va_bin),
         epochs=100, batch_size=32, class_weight=cw_diag,
         callbacks=callbacks_simple(), verbose=1)

# Pick τ on val via macro-F1 (simple threshold search)
p_app_va = diag.predict(Xs_va, verbose=0).ravel()
best_tau, best_f1 = 0.5, -1.0
for tau in TAU_GRID:
    f1 = f1_score(y_diag_va_bin, (p_app_va >= tau).astype(int), average="macro", zero_division=0)
    if f1 > best_f1:
        best_tau, best_f1 = tau, f1
print(f"\n[Diagnosis] chosen τ={best_tau:.3f} (val macro-F1={best_f1:.3f})")

# Refit on full TrainVal then score TEST
diag_final = build_mlp_binary(Xs_trv.shape[1])
diag_final.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss="binary_crossentropy",
                   metrics=["accuracy", keras.metrics.AUC(name="auc")])

cw_diag_full = compute_class_weight("balanced", classes=np.array([0,1]),
                                    y=(y_diag_trv == APPY_ID).astype(int))
cw_diag_full = {0: cw_diag_full[0], 1: cw_diag_full[1]}

diag_final.fit(Xs_trv, (y_diag_trv == APPY_ID).astype(np.float32),
               validation_split=0.2, epochs=100, batch_size=32,
               class_weight=cw_diag_full, callbacks=callbacks_simple(), verbose=0)

# final diagnosis results
p_app_tst = diag_final.predict(Xs_tst, verbose=0).ravel()
y_diag_tst_bin = (y_diag_tst == APPY_ID).astype(int)
y_diag_pred_bin = (p_app_tst >= best_tau).astype(int)
dx_roc = roc_auc_score(y_diag_tst_bin, p_app_tst)
dx_pr  = average_precision_score(y_diag_tst_bin, p_app_tst)

# diagnosis evaluation
print("\n[Diagnosis Test report]:\n")
print(classification_report(y_diag_tst_bin, y_diag_pred_bin, target_names=["no appy","appy"], zero_division=0))
print(f"\nROC AUC: {dx_roc:.3f}  PR AUC: {dx_pr:.3f}\n")

# === Diagnosis confusion matrix ===
labels = ["no appy", "appy"]

cm = confusion_matrix(y_diag_tst_bin, y_diag_pred_bin, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title(f"Diagnosis Confusion Matrix (τ = {best_tau:.2f})")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
plt.tight_layout()
plt.show()


# Stage 2 — Severity (TRUE appy only) -----

# slice TRAIN/VAL to true-appy rows only
X_sev_tr = Xs_tr[is_appy_tr]
y_sev_tr_bin = y_sev_tr[is_appy_tr].astype(np.float32)
X_sev_va = Xs_va[is_appy_va]
y_sev_va_bin = y_sev_va[is_appy_va].astype(np.float32)

# new model for severity
sev = build_mlp_binary(X_sev_tr.shape[1])
sev.compile(optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc"),
                     keras.metrics.AUC(name="pr_auc", curve="PR")])

# optional class weights (if both classes present)
cw_sev = None
if len(np.unique(y_sev_tr_bin)) == 2:
    _cw = compute_class_weight("balanced", classes=np.array([0,1]), y=y_sev_tr_bin.astype(int))
    cw_sev = {0: _cw[0], 1: _cw[1]}

# train on appy TRAIN, validate on appy VAL
sev.fit(X_sev_tr, y_sev_tr_bin,
        validation_data=(X_sev_va, y_sev_va_bin),
        epochs=100, batch_size=32, class_weight=cw_sev,
        callbacks=callbacks_simple(), verbose=1)

# Refit on full TrainVal
X_sev_trv = Xs_trv[is_appy_trv]
y_sev_trv_bin = y_sev_trv[is_appy_trv].astype(np.float32)

sev_final = build_mlp_binary(X_sev_trv.shape[1])
sev_final.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"),
                           keras.metrics.AUC(name="pr_auc", curve="PR")])

cw_sev_full = None
if len(np.unique(y_sev_trv_bin)) == 2:
    _cw = compute_class_weight("balanced", classes=np.array([0,1]), y=y_sev_trv_bin.astype(int))
    cw_sev_full = {0: _cw[0], 1: _cw[1]}

sev_final.fit(X_sev_trv, y_sev_trv_bin,
              validation_split=0.2, epochs=100, batch_size=32,
              class_weight=cw_sev_full, callbacks=callbacks_simple(), verbose=0)

# final Severity results
p_sev_tst_all = sev_final.predict(Xs_tst, verbose=0).ravel()
mask_true_appy_tst = (y_diag_tst == APPY_ID)
sev_true_tst = y_sev_tst[mask_true_appy_tst]
sev_pred_tst = p_sev_tst_all[mask_true_appy_tst]

sev_thr = 0.5
sev_pred_lbl = (sev_pred_tst >= sev_thr).astype(int)

sev_auc = roc_auc_score(sev_true_tst.astype(int), sev_pred_tst)
sev_ap = average_precision_score(sev_true_tst.astype(int), sev_pred_tst)

# severity evaluation
if np.isfinite(sev_pred_tst).all() and len(np.unique(sev_true_tst[~np.isnan(sev_true_tst)])) == 2:
    print("\n[Severity Test report]:\n")
    print(classification_report(sev_true_tst.astype(int), sev_pred_lbl, zero_division=0))
    print(f"\n ROC AUC: {sev_auc:.3f}  PR AUC: {sev_ap:.3f}\n")

    # === Severity confusion matrix ===
    labels = ["non-complicated", "complicated"]

    cm = confusion_matrix(sev_true_tst.astype(int), sev_pred_lbl, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Oranges", colorbar=False, values_format="d")
    ax.set_title(f"Severity Confusion Matrix (τ = {sev_thr:.2f})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.show()

else:
    print("\n[Severity|TRUE appy] AUC skipped (non-finite or single-class).")


# Stage 3 — Management (TRUE appy only), features-only -----

# appy-only TrainVal pool for management
mgmt_X_trv = Xs_trv[is_appy_trv]
mgmt_y_trv = y_mgmt_trv[is_appy_trv].astype(np.float32)

# create a fresh train/val split inside the appy TrainVal pool
X_mgmt_tr, X_mgmt_va, y_mgmt_tr2, y_mgmt_va2 = train_test_split(
    mgmt_X_trv, mgmt_y_trv, test_size=VAL_SIZE, random_state=SEED, stratify=mgmt_y_trv
)

# new model for management
mgmt = build_mlp_binary(X_mgmt_tr.shape[1])
mgmt.compile(optimizer=keras.optimizers.Adam(1e-3),
             loss="binary_crossentropy",
             metrics=["accuracy", keras.metrics.AUC(name="auc")])

# optional class weights (if both classes present)
cw_mgmt = None
if len(np.unique(y_mgmt_tr2)) == 2:
    _cw = compute_class_weight("balanced", classes=np.array([0,1]), y=y_mgmt_tr2.astype(int))
    cw_mgmt = {0: _cw[0], 1: _cw[1]}

# train on appy TRAIN, validate on appy VAL
mgmt.fit(X_mgmt_tr, y_mgmt_tr2,
         validation_data=(X_mgmt_va, y_mgmt_va2),
         epochs=100, batch_size=32, class_weight=cw_mgmt,
         callbacks=callbacks_simple(), verbose=1)

# final Management results
tau = float(best_tau)
gate = (p_app_tst >= tau)
print(f"\n[Gating] τ={tau:.3f} → coverage={gate.mean()*100:.1f}% of test cases")

mgmt_true_mask = (y_diag_tst == APPY_ID)
mgmt_true = y_mgmt_tst[mgmt_true_mask].astype(int)
mgmt_p_all = mgmt.predict(Xs_tst, verbose=0).ravel()
mgmt_p = mgmt_p_all[mgmt_true_mask]

mgmt_roc = roc_auc_score(mgmt_true, mgmt_p)
mgmt_pr  = average_precision_score(mgmt_true, mgmt_p)

# management evaluation
if np.isfinite(mgmt_p).any():
    mgmt_pred = (mgmt_p >= 0.5).astype(int)
    print("\n[Management Test report:]\n")
    print(classification_report(mgmt_true, mgmt_pred, target_names=mgmt_names, zero_division=0))
    print(f"\nROC AUC: {mgmt_roc:.3f}  PR AUC: {mgmt_pr:.3f}\n")

    # === Management confusion matrix ===
    labels = ["conservative", "primary_surgical"]

    cm = confusion_matrix(mgmt_true, mgmt_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Greens", colorbar=False, values_format="d")
    ax.set_title("Management Confusion Matrix (TRUE appy)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.show()
else:
    print("\n[Management|TRUE appy] Non-finite predictions.")


# Save artifacts -----
import joblib
joblib.dump(imputer, OUT_DIR / "imputer_median.joblib")
joblib.dump(scaler,  OUT_DIR / "scaler_standard.joblib")
joblib.dump(le_diag, OUT_DIR / "labelenc_diagnosis.joblib")
np.save(OUT_DIR / "diag_tau.npy", np.array([tau], dtype=np.float32))

diag_final.save(OUT_DIR / "r4_diag_final.keras")
sev_final.save(OUT_DIR / "r4_sev_final.keras")
mgmt.save(OUT_DIR / "r4_mgmt.keras")
print("\nSaved preprocessor + models to: mlp_files")