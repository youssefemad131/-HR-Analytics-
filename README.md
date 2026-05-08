# 📊 HR Analytics — Job Change of Data Scientists

A complete **Machine Learning Preprocessing Pipeline** that prepares the HR Analytics dataset for predicting whether a data scientist is looking for a job change.

> **Course:** Machine Learning
> **Author:** Youssef Emad
> **University:** Benha University — Computer Science, AI Track

---

## 📌 Problem

Given candidate background data, predict whether a data scientist is:

| Target | Meaning |
|--------|---------|
| `0` | Not looking for a job change |
| `1` | Looking for a job change |

> This is a **Binary Classification** problem with imbalanced classes.

---

## 📦 Dataset

**HR Analytics: Job Change of Data Scientists** — [Kaggle](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists/data)

| File | Description |
|------|-------------|
| `aug_train.csv` | Labeled training data |
| `aug_test.csv` | Unlabeled test data |

**Key Features:**

| Feature | Description |
|---------|-------------|
| `city_development_index` | Development level of candidate's city (0–1) |
| `experience` | Years of experience |
| `education_level` | Highest degree obtained |
| `company_size` | Size of current employer |
| `training_hours` | Hours of training completed |
| `major_discipline` | Field of study |
| `company_type` | Type of current company |
| `target` | 0 = Stay, 1 = Looking to switch |

---

## 🔁 Pipeline Overview

```
Raw Data
   │
   ├── 1. Load & Explore          → shape, dtypes, target distribution
   ├── 2. Missing Value Analysis  → heatmap + imputation
   ├── 3. Outlier Detection       → boxplots + 99th percentile capping
   ├── 4. Feature Cleaning        → fix experience, last_new_job strings
   ├── 5. Ordinal Encoding        → education_level, company_size
   ├── 6. One-Hot Encoding        → gender, major_discipline, company_type...
   ├── 7. Feature Engineering     → 3 new derived features
   ├── 8. Drop Irrelevant Cols    → enrollee_id, city, redundant cols
   ├── 9. Scaling                 → StandardScaler on numeric features
   └── 10. Train / Val Split      → 80/20 stratified split
         │
         └── ✅ Clean CSVs saved
```

---

## 🛠️ Preprocessing Steps

### Step 1 — Missing Values

| Strategy | Applied To |
|----------|-----------|
| Fill with `'Unknown'` | All categorical columns (preserves missingness signal) |
| Fill with **median** | All numerical columns |

### Step 2 — Outliers

| Feature | Action |
|---------|--------|
| `training_hours` | Capped at **99th percentile** (right-skewed distribution) |
| `city_development_index` | No capping needed (bounded 0–1) |

### Step 3 — Feature Cleaning

| Feature | Raw Values | Cleaned To |
|---------|-----------|------------|
| `experience` | `'>20'`, `'<1'` | `21`, `0` |
| `last_new_job` | `'>4'`, `'never'` | `5`, `0` |

### Step 4 — Encoding

**Ordinal Encoding** (natural order preserved):

| Feature | Scale |
|---------|-------|
| `education_level` | Unknown=0 → PhD=5 |
| `company_size` | Unknown=0 → 10000+=8 |

**One-Hot Encoding** (`drop_first=True`):
`gender`, `relevent_experience`, `enrolled_university`, `major_discipline`, `company_type`

### Step 5 — Feature Engineering

| New Feature | Formula | Meaning |
|-------------|---------|---------|
| `hours_per_exp` | `training_hours / (experience + 1)` | Productivity proxy |
| `cdi_x_exp` | `city_development_index × experience` | City-experience interaction |
| `high_cdi` | `city_development_index >= 0.9` | Binary: highly developed city |

### Step 6 — Scaling

`StandardScaler` applied to:
`city_development_index`, `training_hours`, `experience`, `last_new_job`, `hours_per_exp`, `cdi_x_exp`

### Step 7 — Train / Val Split

| Split | Size | Method |
|-------|------|--------|
| Train | 80% | Stratified by target |
| Validation | 20% | Stratified by target |

---

## 💻 Code

### 📥 1. Setup & Download

```python
!pip install kagglehub -q

import kagglehub
import shutil, os

path = kagglehub.dataset_download('arashnic/hr-analytics-job-change-of-data-scientists')

dest = '/content/hr_data'
os.makedirs(dest, exist_ok=True)
for f in os.listdir(path):
    shutil.copy(os.path.join(path, f), dest)
```

---

### 📦 2. Imports & Load Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)
sns.set_theme(style='whitegrid')

train = pd.read_csv('/content/hr_data/aug_train.csv')
test  = pd.read_csv('/content/hr_data/aug_test.csv')

print('Train shape:', train.shape)
print('Test  shape:', test.shape)
```

---

### 🔍 3. Exploratory Overview

```python
train.info()
train.describe(include='all').T

# Target distribution
print(train['target'].value_counts())

fig, ax = plt.subplots(figsize=(5, 4))
train['target'].value_counts().plot(kind='bar', color=['steelblue', 'coral'], ax=ax)
ax.set_xticklabels(['Not looking (0)', 'Looking (1)'], rotation=0)
ax.set_title('Target Distribution')
plt.tight_layout()
plt.show()
```

---

### 🧹 4. Missing Value Analysis & Imputation

```python
# Heatmap
missing     = train.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(train) * 100).round(2)
missing_df  = pd.DataFrame({'Missing': missing, 'Pct (%)': missing_pct})
print(missing_df[missing_df['Missing'] > 0])

plt.figure(figsize=(10, 4))
sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.show()

# Imputation
def impute_df(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in ['enrollee_id', 'target']]
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

train = impute_df(train)
test  = impute_df(test)
```

---

### 📦 5. Outlier Detection & Capping

```python
num_features = ['city_development_index', 'training_hours']

fig, axes = plt.subplots(1, len(num_features), figsize=(12, 4))
for ax, col in zip(axes, num_features):
    sns.boxplot(y=train[col], ax=ax, color='lightblue')
    ax.set_title(col)
plt.suptitle('Outlier Check — Box Plots')
plt.tight_layout()
plt.show()

# Cap training_hours at 99th percentile
cap_99 = train['training_hours'].quantile(0.99)
train['training_hours'] = train['training_hours'].clip(upper=cap_99)
test['training_hours']  = test['training_hours'].clip(upper=cap_99)
print(f'training_hours capped at {cap_99:.0f} hours')
```

---

### 🔧 6. Feature Cleaning

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
    df['experience']    = df['experience'].apply(clean_experience)
    df['experience']    = df['experience'].fillna(df['experience'].median())
    df['last_new_job']  = df['last_new_job'].apply(clean_last_new_job)
    df['last_new_job']  = df['last_new_job'].fillna(df['last_new_job'].median())
```

---

### 🔢 7. Feature Encoding

```python
# Ordinal Encoding
edu_order = {'Unknown':0,'Primary School':1,'High School':2,
             'Graduate':3,'Masters':4,'Phd':5}
size_order = {'Unknown':0,'<10':1,'10/49':2,'50-99':3,'100-500':4,
              '500-999':5,'1000-4999':6,'5000-9999':7,'10000+':8}

for df in [train, test]:
    df['education_level_enc'] = df['education_level'].map(edu_order).fillna(0).astype(int)
    df['company_size_enc']    = df['company_size'].map(size_order).fillna(0).astype(int)

# One-Hot Encoding
ohe_cols = ['gender', 'relevent_experience', 'enrolled_university',
            'major_discipline', 'company_type']

train_enc = pd.get_dummies(train, columns=ohe_cols, drop_first=True)
test_enc  = pd.get_dummies(test,  columns=ohe_cols, drop_first=True)
train_enc, test_enc = train_enc.align(test_enc, join='left', axis=1, fill_value=0)
```

---

### ⚗️ 8. Feature Engineering

```python
for df in [train_enc, test_enc]:
    df['hours_per_exp'] = df['training_hours'] / (df['experience'] + 1)
    df['cdi_x_exp']     = df['city_development_index'] * df['experience']
    df['high_cdi']      = (df['city_development_index'] >= 0.9).astype(int)
```

---

### 🗑️ 9. Drop Irrelevant Columns

```python
drop_cols = ['enrollee_id', 'city', 'education_level', 'company_size']

train_clean = train_enc.drop(columns=[c for c in drop_cols if c in train_enc.columns])
test_clean  = test_enc.drop(columns=[c for c in drop_cols if c in test_enc.columns])
```

---

### 📏 10. Scaling

```python
scale_cols = ['city_development_index', 'training_hours', 'experience',
              'last_new_job', 'hours_per_exp', 'cdi_x_exp']
scale_cols = [c for c in scale_cols if c in train_clean.columns]

scaler = StandardScaler()
train_clean[scale_cols] = scaler.fit_transform(train_clean[scale_cols])
test_clean[scale_cols]  = scaler.transform(test_clean[scale_cols])
```

---

### ✂️ 11. Train / Validation Split

```python
X = train_clean.drop(columns=['target'])
y = train_clean['target']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'X_train: {X_train.shape}  |  X_val: {X_val.shape}')
print(f'Class balance → {y_train.value_counts(normalize=True).round(3).to_dict()}')
```

---

### 📊 12. Correlation Heatmap

```python
corr_cols = scale_cols + ['education_level_enc', 'company_size_enc', 'high_cdi', 'target']
corr_cols = [c for c in corr_cols if c in train_clean.columns]

plt.figure(figsize=(10, 7))
sns.heatmap(train_clean[corr_cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Matrix — Numeric Features')
plt.tight_layout()
plt.show()
```

---

### 💾 13. Save Cleaned Files

```python
os.makedirs('/content/hr_data/cleaned', exist_ok=True)

X_train.assign(target=y_train.values).to_csv('/content/hr_data/cleaned/train_clean.csv', index=False)
X_val.assign(target=y_val.values).to_csv('/content/hr_data/cleaned/val_clean.csv',       index=False)
test_clean.to_csv('/content/hr_data/cleaned/test_clean.csv',                              index=False)

print('✅ Saved: train_clean.csv | val_clean.csv | test_clean.csv')
```

---

## ✅ Preprocessing Summary

| Step | Action |
|------|--------|
| Missing values | Categorical → `'Unknown'`, Numerical → median |
| Outliers | `training_hours` capped at 99th percentile |
| `experience` | `'>20'`→21, `'<1'`→0 |
| `last_new_job` | `'>4'`→5, `'never'`→0 |
| `education_level` | Ordinal encoded (0–5) |
| `company_size` | Ordinal encoded (0–8) |
| Categorical cols | One-hot encoded (`drop_first=True`) |
| New features | `hours_per_exp`, `cdi_x_exp`, `high_cdi` |
| Scaling | `StandardScaler` on 6 numeric features |
| Split | 80/20 stratified train/val split |

---

## 📁 Repository Structure

```
HR-Analytics-Job-Change/
│
├── HR_Analytics_Preprocessing.ipynb   ← Full preprocessing notebook
├── data/
│   ├── aug_train.csv                  ← Raw training data
│   ├── aug_test.csv                   ← Raw test data
│   └── cleaned/
│       ├── train_clean.csv            ← Ready for modeling
│       ├── val_clean.csv
│       └── test_clean.csv
└── README.md
```

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

---

## 📚 Concepts Used

- `Exploratory Data Analysis (EDA)`
- `Missing Value Imputation`
- `Outlier Detection & Capping`
- `Ordinal & One-Hot Encoding`
- `Feature Engineering`
- `StandardScaler Normalization`
- `Stratified Train/Val Split`
- `Correlation Analysis`

---

## 💬 Repo Description

```
📊 Full ML preprocessing pipeline for HR Analytics — Job Change of Data Scientists.
Covers EDA, missing value imputation, outlier capping, ordinal + one-hot encoding,
feature engineering, scaling, and stratified splitting. Built with pandas, scikit-learn & seaborn.
```
