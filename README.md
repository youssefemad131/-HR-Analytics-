<h1 align="center">📊 HR Analytics — Job Change of Data Scientists</h1>
<h3 align="center">End-to-End Machine Learning Preprocessing Pipeline</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-Data%20Wrangling-150458?style=flat-square&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-Preprocessing-orange?style=flat-square&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=flat-square"/>
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=flat-square&logo=googlecolab&logoColor=black"/>
</p>

<p align="center">
  <b>Course:</b> Machine Learning &nbsp;·&nbsp;
  <b>Author:</b> Youssef Emad &nbsp;·&nbsp;
  <b>University:</b> Benha University — Computer Science, AI Track
</p>

---

## 📌 Problem Statement

Given candidate background data, predict whether a data scientist is actively seeking a new job after training — enabling companies to optimize **recruitment costs** and design **retention strategies**.

| Target | Meaning |
|--------|---------|
| `0` | Not looking for a job change |
| `1` | Looking for a job change |

> **Binary Classification** with a ~75:25 class imbalance.

---

## 📦 Dataset

**HR Analytics: Job Change of Data Scientists** — [Kaggle](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists/data)

| File | Rows | Description |
|------|------|-------------|
| `aug_train.csv` | ~19,158 | Labeled training data |
| `aug_test.csv` | ~2,129 | Unlabeled test data |

**Key Features:**

| Feature | Type | Description |
|---------|------|-------------|
| `city_development_index` | Numerical | Urbanisation level of city (0–1) |
| `experience` | Numerical | Years of work experience |
| `training_hours` | Numerical | Hours of training completed |
| `education_level` | Ordinal | Primary → High School → Graduate → Masters → PhD |
| `company_size` | Ordinal | Headcount band of current employer |
| `gender` | Categorical | Gender of candidate |
| `relevent_experience` | Categorical | Whether candidate has relevant experience |
| `enrolled_university` | Categorical | Type of university enrollment |
| `major_discipline` | Categorical | Field of study |
| `company_type` | Categorical | Type of current company |

---

## 🗺️ Pipeline Overview

```
Raw Data  (aug_train.csv · aug_test.csv)
   │
   ├── 01. Load & Explore          → shape, dtypes, target distribution
   ├── 02. Missing Value Analysis  → heatmap + smart imputation
   ├── 03. Outlier Detection       → box plots + 99th percentile capping
   ├── 04. Feature Cleaning        → fix string-encoded numerics
   ├── 05. Ordinal Encoding        → education_level, company_size
   ├── 06. One-Hot Encoding        → gender, major_discipline, company_type…
   ├── 07. Feature Engineering     → 3 new derived interaction features
   ├── 08. Drop Irrelevant Cols    → enrollee_id, city, redundant raw cols
   ├── 09. Scaling                 → StandardScaler (fit on train only)
   └── 10. Train / Val Split       → 80/20 stratified split
         │
         ├── train_clean.csv  ✅
         ├── val_clean.csv    ✅
         └── test_clean.csv   ✅
```

---

## 🛠️ Preprocessing Steps in Detail

### 1 — Missing Value Imputation

| Strategy | Applied To | Reason |
|----------|-----------|--------|
| Fill with `'Unknown'` | All categorical columns | Preserves missingness as a signal |
| Fill with **median** | All numerical columns | Robust to skew and outliers |

> Both train and test are imputed with the **same function** to prevent any data leakage.

---

### 2 — Outlier Handling

| Feature | Action | Reason |
|---------|--------|--------|
| `training_hours` | Capped at **99th percentile** | Right-skewed with extreme high values |
| `city_development_index` | No capping | Bounded [0, 1] by nature |

---

### 3 — Feature Cleaning

Raw string values converted to clean integers:

| Feature | Raw → Cleaned |
|---------|---------------|
| `experience` | `'>20'` → `21` · `'<1'` → `0` |
| `last_new_job` | `'>4'` → `5` · `'never'` → `0` |

---

### 4 — Encoding

**Ordinal Encoding** — natural order preserved:

| Feature | Scale |
|---------|-------|
| `education_level` | `Unknown`=0 · `Primary School`=1 · `High School`=2 · `Graduate`=3 · `Masters`=4 · `PhD`=5 |
| `company_size` | `Unknown`=0 · `<10`=1 · `10/49`=2 · `50-99`=3 · `100-500`=4 · `500-999`=5 · `1000-4999`=6 · `5000-9999`=7 · `10000+`=8 |

**One-Hot Encoding** (`drop_first=True`) applied to:
`gender` · `relevent_experience` · `enrolled_university` · `major_discipline` · `company_type`

> Train and test columns are **aligned** after OHE to handle any category present in one but not the other.

---

### 5 — Feature Engineering

3 new interaction features capturing signals not present in raw columns:

| New Feature | Formula | Intuition |
|-------------|---------|-----------|
| `hours_per_exp` | `training_hours / (experience + 1)` | Training intensity relative to seniority |
| `cdi_x_exp` | `city_development_index × experience` | Combined urbanisation & experience signal |
| `high_cdi` | `CDI ≥ 0.9 → 1, else 0` | Binary flag for highly developed cities |

---

### 6 — Scaling

`StandardScaler` applied to 6 continuous features:

```
city_development_index · training_hours · experience
last_new_job · hours_per_exp · cdi_x_exp
```

> ⚠️ Scaler is **fit only on the training set** then applied to val and test — zero leakage.

---

### 7 — Train / Validation Split

| Split | Size | Method |
|-------|------|--------|
| Train | 80% | Stratified by `target` |
| Validation | 20% | Stratified by `target` |

---

## 💻 Code

<details>
<summary><b>📥 01. Setup & Download Dataset</b></summary>

```python
!pip install kagglehub -q

import kagglehub, shutil, os

path = kagglehub.dataset_download('arashnic/hr-analytics-job-change-of-data-scientists')
dest = '/content/hr_data'
os.makedirs(dest, exist_ok=True)
for f in os.listdir(path):
    shutil.copy(os.path.join(path, f), dest)

print('✅ Files:', os.listdir(dest))
```

</details>

<details>
<summary><b>📦 02. Imports & Load Data</b></summary>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
sns.set_theme(style='whitegrid')
SEED = 42

train = pd.read_csv('/content/hr_data/aug_train.csv')
test  = pd.read_csv('/content/hr_data/aug_test.csv')

print(f'Train shape: {train.shape}')
print(f'Test  shape: {test.shape}')
train.head()
```

</details>

<details>
<summary><b>🔍 03. Exploratory Overview</b></summary>

```python
train.info()
train.describe(include='all').T

# Target distribution
counts = train['target'].value_counts()
pct    = counts / len(train) * 100

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(['Not Changing (0)', 'Changing (1)'], counts.values,
            color=['#4C72B0', '#DD8452'], edgecolor='white')
for i, (v, p) in enumerate(zip(counts.values, pct.values)):
    axes[0].text(i, v + 40, f'{v}\n({p:.1f}%)', ha='center', fontweight='bold')
axes[0].set_title('Target Distribution (Count)', fontweight='bold')

axes[1].pie(counts.values, labels=['Not Changing', 'Changing'],
            autopct='%1.1f%%', colors=['#4C72B0','#DD8452'],
            startangle=90, wedgeprops=dict(edgecolor='white'))
axes[1].set_title('Target Distribution (Proportion)', fontweight='bold')

plt.suptitle('⚠️ Class Imbalance Detected (~75% vs ~25%)', color='firebrick')
plt.tight_layout()
plt.show()
```

</details>

<details>
<summary><b>🧹 04. Missing Value Analysis & Imputation</b></summary>

```python
missing     = train.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(train) * 100).round(2)
print(pd.DataFrame({'Missing': missing, 'Pct (%)': missing_pct})[missing > 0])

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
(missing_pct[missing_pct > 0].sort_values()
 .plot(kind='barh', ax=axes[0], color='#DD8452', edgecolor='white'))
axes[0].set_title('Missing Values (%) per Column', fontweight='bold')
sns.heatmap(train[missing[missing > 0].index].isnull(),
            cbar=False, yticklabels=False, cmap='Oranges', ax=axes[1])
axes[1].set_title('Missing Value Heatmap', fontweight='bold')
plt.tight_layout(); plt.show()

def impute_df(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')
    for col in [c for c in df.select_dtypes(include=np.number).columns
                if c not in ['enrollee_id', 'target']]:
        df[col] = df[col].fillna(df[col].median())
    return df

train = impute_df(train)
test  = impute_df(test)
print(f'✅ Missing after imputation — Train: {train.isnull().sum().sum()} | Test: {test.isnull().sum().sum()}')
```

</details>

<details>
<summary><b>📦 05. Outlier Detection & Capping</b></summary>

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ['city_development_index', 'training_hours']):
    sns.boxplot(y=train[col], ax=ax, color='#4C72B0',
                flierprops=dict(marker='o', markerfacecolor='red', markersize=4))
    ax.set_title(f'{col}', fontweight='bold')
plt.suptitle('Outlier Detection — Box Plots', fontweight='bold')
plt.tight_layout(); plt.show()

cap_99 = train['training_hours'].quantile(0.99)
train['training_hours'] = train['training_hours'].clip(upper=cap_99)
test['training_hours']  = test['training_hours'].clip(upper=cap_99)
print(f'✅ training_hours capped at {cap_99:.0f} hrs (99th percentile)')
```

</details>

<details>
<summary><b>🔧 06. Feature Cleaning</b></summary>

```python
def clean_experience(val):
    if val == '>20': return 21
    if val == '<1':  return 0
    try: return int(val)
    except: return np.nan

def clean_last_new_job(val):
    if val == '>4':    return 5
    if val == 'never': return 0
    try: return int(val)
    except: return np.nan

for df in [train, test]:
    df['experience']   = df['experience'].apply(clean_experience).fillna(train['experience'].median())
    df['last_new_job'] = df['last_new_job'].apply(clean_last_new_job).fillna(train['last_new_job'].median())

print('✅ experience unique values:   ', sorted(train['experience'].unique()))
print('✅ last_new_job unique values: ', sorted(train['last_new_job'].unique()))
```

</details>

<details>
<summary><b>🔢 07. Feature Encoding</b></summary>

```python
# Ordinal encoding
edu_order  = {'Unknown':0,'Primary School':1,'High School':2,'Graduate':3,'Masters':4,'Phd':5}
size_order = {'Unknown':0,'<10':1,'10/49':2,'50-99':3,'100-500':4,
              '500-999':5,'1000-4999':6,'5000-9999':7,'10000+':8}

for df in [train, test]:
    df['education_level_enc'] = df['education_level'].map(edu_order).fillna(0).astype(int)
    df['company_size_enc']    = df['company_size'].map(size_order).fillna(0).astype(int)

# One-hot encoding
ohe_cols = ['gender', 'relevent_experience', 'enrolled_university',
            'major_discipline', 'company_type']
train_enc = pd.get_dummies(train, columns=ohe_cols, drop_first=True)
test_enc  = pd.get_dummies(test,  columns=ohe_cols, drop_first=True)
train_enc, test_enc = train_enc.align(test_enc, join='left', axis=1, fill_value=0)

print(f'✅ Shape after OHE — Train: {train_enc.shape} | Test: {test_enc.shape}')
```

</details>

<details>
<summary><b>⚗️ 08. Feature Engineering</b></summary>

```python
for df in [train_enc, test_enc]:
    df['hours_per_exp'] = df['training_hours'] / (df['experience'] + 1)
    df['cdi_x_exp']     = df['city_development_index'] * df['experience']
    df['high_cdi']      = (df['city_development_index'] >= 0.9).astype(int)

print('✅ New features added: hours_per_exp · cdi_x_exp · high_cdi')
```

</details>

<details>
<summary><b>🗑️ 09. Drop Irrelevant Columns</b></summary>

```python
drop_cols   = ['enrollee_id', 'city', 'education_level', 'company_size']
train_clean = train_enc.drop(columns=[c for c in drop_cols if c in train_enc.columns])
test_clean  = test_enc.drop(columns=[c for c in drop_cols if c in test_enc.columns])

if 'target' in test_clean.columns:
    test_clean = test_clean.drop(columns=['target'])

print(f'✅ Train columns kept: {train_clean.shape[1]} | Test: {test_clean.shape[1]}')
```

</details>

<details>
<summary><b>📏 10. Scaling</b></summary>

```python
scale_cols = [c for c in ['city_development_index', 'training_hours', 'experience',
                           'last_new_job', 'hours_per_exp', 'cdi_x_exp']
              if c in train_clean.columns]

scaler = StandardScaler()
train_clean[scale_cols] = scaler.fit_transform(train_clean[scale_cols])
test_clean[scale_cols]  = scaler.transform(test_clean[scale_cols])

print('✅ Scaling done — mean ≈ 0, std ≈ 1')
print(train_clean[scale_cols].describe().T[['mean', 'std']].round(3))
```

</details>

<details>
<summary><b>✂️ 11. Train / Validation Split</b></summary>

```python
X = train_clean.drop(columns=['target'])
y = train_clean['target']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f'X_train: {X_train.shape}  |  X_val: {X_val.shape}')
print(f'Train class balance → {y_train.value_counts(normalize=True).round(3).to_dict()}')
print(f'Val   class balance → {y_val.value_counts(normalize=True).round(3).to_dict()}')
```

</details>

<details>
<summary><b>📊 12. Correlation Heatmap</b></summary>

```python
corr_cols = scale_cols + ['education_level_enc', 'company_size_enc', 'high_cdi', 'target']
corr_cols = [c for c in corr_cols if c in train_clean.columns]

plt.figure(figsize=(11, 8))
mask = np.triu(np.ones_like(train_clean[corr_cols].corr(), dtype=bool))
sns.heatmap(train_clean[corr_cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, mask=mask,
            linewidths=0.5, annot_kws={'size': 9})
plt.title('Correlation Matrix — Numerical & Ordinal Features', fontweight='bold')
plt.tight_layout(); plt.show()

print('\nTop features correlated with target:')
print(train_clean[corr_cols].corr()['target'].drop('target')
      .abs().sort_values(ascending=False).round(3).to_string())
```

</details>

<details>
<summary><b>💾 13. Save Cleaned Files</b></summary>

```python
os.makedirs('/content/hr_data/cleaned', exist_ok=True)

X_train.assign(target=y_train.values).to_csv('/content/hr_data/cleaned/train_clean.csv', index=False)
X_val.assign(target=y_val.values).to_csv('/content/hr_data/cleaned/val_clean.csv',       index=False)
test_clean.to_csv('/content/hr_data/cleaned/test_clean.csv',                              index=False)

print('✅ Saved: train_clean.csv | val_clean.csv | test_clean.csv')
```

</details>

---

## ✅ Preprocessing Summary

| Step | Action | Detail |
|------|--------|--------|
| Missing — categorical | Fill `'Unknown'` | Treats missingness as its own category |
| Missing — numerical | Fill **median** | Robust to skewed distributions |
| Outliers | Cap `training_hours` at 99th pct | Reduces right-tail distortion |
| `experience` | `'>20'`→21 · `'<1'`→0 | Enables numerical treatment |
| `last_new_job` | `'>4'`→5 · `'never'`→0 | Enables numerical treatment |
| `education_level` | Ordinal encoded (0–5) | Preserves natural order |
| `company_size` | Ordinal encoded (0–8) | Preserves natural order |
| Categorical cols | One-Hot (`drop_first=True`) | No ordinal assumption |
| New features | `hours_per_exp` · `cdi_x_exp` · `high_cdi` | Capture interaction signals |
| Scaling | `StandardScaler` on 6 numeric cols | Fit on train only — no leakage |
| Split | 80/20 stratified | Preserves class ratio in both sets |

---

## 📁 Repository Structure

```
HR-Analytics-Job-Change/
│
├── 📁 Documentation/                    ← Report & docs
├── 📁 Presentation/                     ← Slides
└── 📁 code/
    ├── ML_Project.ipynb                 ← Full preprocessing notebook
    ├── data/
    │   ├── aug_train.csv                ← Raw training data
    │   ├── aug_test.csv                 ← Raw test data
    │   └── cleaned/
    │       ├── train_clean.csv          ← Ready for modeling ✅
    │       ├── val_clean.csv            ✅
    │       └── test_clean.csv           ✅
    └── README.md
```

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

---

## 📚 Concepts Used

`Exploratory Data Analysis` · `Missing Value Imputation` · `Outlier Detection & Capping` · `Ordinal Encoding` · `One-Hot Encoding` · `Feature Engineering` · `StandardScaler` · `Stratified Train/Val Split` · `Correlation Analysis` · `Binary Classification`

---

## 🔮 Next Steps

| # | What | Why |
|---|------|-----|
| 1 | **SMOTE oversampling** | Fix ~75:25 class imbalance before modeling |
| 2 | **Model training** | Logistic Regression → Random Forest → XGBoost |
| 3 | **Hyperparameter tuning** | `GridSearchCV` or `Optuna` |
| 4 | **SHAP explainability** | Understand per-prediction feature impact |
| 5 | **Streamlit deployment** | Interactive prediction app |

---

<p align="center">
  📊 Built with Python · pandas · scikit-learn · seaborn &nbsp;·&nbsp; Benha University 2025
</p>
