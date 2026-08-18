# Credit Card Default Prediction — ML Assignment 2

**Name:** Prateek Gupta
**BITS ID:** 2025AC05898
**Course:** Machine Learning (AIMLCZG565), M.Tech (AIML), BITS Pilani WILP

---

## a. Problem Statement

Lenders need to know, in advance, which customers are likely to default on their
credit card payments. A default that is not anticipated is a direct financial
loss, so any early signal has real value to a creditor.

This project builds a binary classifier that predicts whether a customer will
default on their next month's payment. The prediction uses only information a
creditor already holds: general customer parameters (credit limit, personal information like age, education etc.)
and payment history (bill amounts, paid amounts, and repayment status over the
preceding six months). No external or hard-to-obtain data is required, which
makes the approach practical to deploy.

---

## b. Dataset Description

**Source:** UCI Machine Learning Repository — Default of Credit Card Clients (id = 350)
**Link:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
**Instances:** 30,000
**Features:** 23
**Target:** `default`, 1 = customer defaulted the following month, 0 = did not default

### Class distribution

| Class | Count | Proportion |
|---|---|---|
| No default (0) | 23,364 | 77.9% |
| Default (1) | 6,636 | 22.1% |

The dataset is imbalanced towards **non-defaulters**. This is important for
interpreting the results and is discussed in the observations below.

### Feature groups

The 23 features fall into three groups.

**1. Personal / demographic information**

| Feature | Description |
|---|---|
| `LIMIT_BAL` | Amount of given credit (NT dollars), including individual and family credit |
| `SEX` | Gender |
| `EDUCATION` | Education level |
| `MARRIAGE` | Marital status |
| `AGE` | Age in years |

**2. Repayment status history** — `PAY_0`, `PAY_2` … `PAY_6`

These record the repayment status in each of the six preceding months. The
coding is ordinal and interpretable: `-1` indicates the customer paid duly,
while values from `1` to `9` indicate the number of months the payment was
delayed. A higher number therefore means worse repayment behaviour, with `9`
being the most severe. This makes the repayment-status columns intuitively the
most informative features in the dataset.

**3. Billing and payment amounts**

| Feature group | Description |
|---|---|
| `BILL_AMT1` … `BILL_AMT6` | Amount of the bill statement for each of the six months |
| `PAY_AMT1` … `PAY_AMT6` | Amount actually paid in each of the six months |

### Note on undocumented category codes

The documented coding for `EDUCATION` is 1–4 and for `MARRIAGE` is 1–3.
However, the data also contains additional codes in both columns whose meaning
is not explained anywhere in the dataset documentation. Since their meaning is
unknown, no interpretation was drawn for them and no merging into an "others"
category was performed. The values were retained as they appear in the
source data and passed to the models as they are.

### Preprocessing

- 80/20 stratified train-test split (`random_state=42`), giving 24,000 training
  and 6,000 test instances. Stratification preserved the class ratio on both
  sides (77.88% / 22.12% train, 77.88% / 22.12% test).
- `StandardScaler` applied inside a scikit-learn `Pipeline` for Logistic
  Regression and kNN, both of which are sensitive to feature scale.
- Tree-based models and Gaussian Naive Bayes were used without scaling, as they
  are scale-invariant.
- No missing values were present. No rows or columns were dropped.

---

## c. GitHub Repository Link

**Repository:** https://github.com/tekkguru/ml-assignment2-credit-default

**Live Streamlit App:** https://t8cbfbsfrxtzaspecialone.streamlit.app/

---

## d. Models Used

All five models were trained on the identical training split and evaluated on
the identical held-out test set of 6,000 instances.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree | 0.8163 | 0.7450 | 0.6543 | 0.3595 | 0.4640 | 0.3881 |
| kNN | 0.8065 | 0.7326 | 0.6153 | 0.3338 | 0.4328 | 0.3507 |
| Naive Bayes | 0.4160 | 0.6516 | 0.2496 | 0.8176 | 0.3824 | 0.1111 |
| Random Forest (Ensemble) | 0.8170 | 0.7728 | 0.6615 | 0.3534 | 0.4607 | 0.3884 |

### Hyperparameters

| Model | Settings |
|---|---|
| Logistic Regression | `max_iter=1000`, StandardScaler in pipeline |
| Decision Tree | `max_depth=6` |
| kNN | `n_neighbors=15`, StandardScaler in pipeline |
| Naive Bayes | GaussianNB, default parameters |
| Random Forest | `n_estimators=100`, `max_depth=12` |

All models used `random_state=42` for reproducibility.

---

## e. Observations

### General observation on the results

Naive Bayes gave a notably low accuracy. For the other four models the accuracy
clustered around 0.81.

However, **accuracy is not the right metric to judge these models**, because the
dataset is imbalanced towards non-defaulters. A model that always predicts "no
default" would still be correct about 78% of the time without learning anything
at all. Accuracy therefore has to be read against that baseline, not against
zero.

For this reason the comparison was made primarily on **MCC** and **AUC**. Across
the models the AUC values sit in a middle range, indicating a real but weak
signal rather than a strong one — with the exception of Naive Bayes, where the
signal is very low.

### Per-model observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Accuracy is in line with the other models at 0.8077, but the AUC is the lowest among the four working models at 0.7076. This suggests the linear decision boundary does not capture the structure of the data as well as the tree-based methods. Precision is the highest in the table (0.6868), but recall is very low (0.2396) — the model flags very few customers as defaulters, and misses most real defaults. |
| **Decision Tree** | Trained with a maximum depth of 6 to prevent the tree from growing until it memorises the training data. With this constraint it performs well, improving on Logistic Regression across AUC (0.7450), F1 (0.4640) and MCC (0.3881). |
| **kNN** | Trained with 15 nearest neighbours. Its MCC of 0.3507 sits in between the Decision Tree and Logistic Regression, so it performs better than Logistic Regression and considerably better than Naive Bayes, but does not reach the level of the tree-based models. `StandardScaler` was applied in the pipeline, which is necessary here because the distance calculation would otherwise be dominated by the large-scale monetary features. |
| **Naive Bayes** | The worst performing model. Its accuracy of 0.4160 is far below what a trivial "always predict no default" rule would achieve. Its recall of 0.8176 is the highest in the table, but this should not be read as good performance — the high recall comes from flagging almost every customer as a defaulter, which is why precision collapses to 0.2496, barely above the 22% base rate. The metrics that matter here, AUC (0.6516) and MCC (0.1111), are both the lowest of all five models. |
| **Random Forest (Ensemble)** | The best performing model. It gives the strongest results not only on precision and recall balance, but also on the two metrics that matter most for this problem: the highest MCC (0.3884) and the highest AUC (0.7728). The higher AUC in particular gives more confidence in the model's predictions, since it indicates the model ranks defaulters above non-defaulters more reliably than the others. |
| **Overall Winner** | **Random Forest.** It achieves the highest AUC and the highest MCC, which are the appropriate metrics for an imbalanced dataset of this kind. While its accuracy is only marginally above the Decision Tree, the clear separation on AUC (0.7728 vs 0.7450) shows genuinely better discriminative ability rather than a difference that could be attributed to noise. |

---

## f. Repository Structure

```
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- ml_assignment2_credit_default.ipynb
|-- credit_default.csv
`-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    `-- random_forest.joblib
```

---

## Streamlit App Features

The deployed application implements all required features:

- **Dataset upload option (CSV)** — test data is uploaded through the app
- **Model selection dropdown** — all five trained models are selectable
- **Display of evaluation metrics** — all six metrics computed on the uploaded data
- **Confusion matrix** — rendered as an annotated heatmap

The results of each model on the test data are visible in the app by selecting
the model from the sidebar dropdown.

---

## Environment

| Library | Version |
|---|---|
| scikit-learn | 1.6.1 |
| pandas | 2.2.2 |
| numpy | 2.0.2 |
| joblib | 1.5.3 |

---

## BITS Virtual Lab Execution

The assignment was executed on BITS Virtual Lab. The screenshot of the
execution is included in the submitted PDF. The results obtained on the Virtual
Lab were identical to those obtained during development, confirming
reproducibility of the fixed random seed across environments.
