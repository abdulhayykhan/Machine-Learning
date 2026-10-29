# Employee Attrition Analyzer

A Machine Learning project developed for the **Lab #7 Open Ended Lab** of the Machine Learning Practical course. The project analyzes employee records and predicts whether an employee is likely to leave the company using multiple classification models.

## Student Information

- **Name:** Abdul Hayy Khan
- **Roll No:** 24F-AI-051
- **Course:** Machine Learning (Practical) — AI-3202
- **Department:** Artificial Intelligence
- **University:** Dawood University of Engineering & Technology (DUET), Karachi

---

## Project Objective

The objective of this project is to build a machine learning-based **Employee Attrition Analyzer** that predicts whether an employee is likely to leave the company.

The project demonstrates concepts covered in previous Machine Learning labs, including:

- Data preprocessing
- Categorical data encoding
- Train-test splitting
- Feature scaling
- Classification models
- Model comparison
- Evaluation metrics
- Confusion matrices
- Feature importance
- Data visualization

---

## Dataset

The project uses the **IBM HR Analytics Employee Attrition & Performance** dataset.

- **Total Records:** 1,470 employees
- **Original Features:** 35 columns
- **Target Variable:** `Attrition`
- **Target Classes:**
  - `Yes` → Employee leaves the company
  - `No` → Employee stays in the company

### Class Distribution

| Attrition | Count |
|-----------|------:|
| No | 1,233 |
| Yes | 237 |

The dataset contains both numerical and categorical employee information such as age, income, job role, overtime, job satisfaction, years at company, and other employee-related attributes.

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Checked the dataset structure, shape, information, and missing values.
3. Removed duplicate records.
4. Removed unnecessary columns:
   - `EmployeeCount`
   - `EmployeeNumber`
   - `Over18`
   - `StandardHours`
5. Converted the target variable:
   - `Yes` → `1`
   - `No` → `0`
6. Applied one-hot encoding to categorical features using `pd.get_dummies()`.
7. Used `drop_first=True` to avoid redundant dummy variables.
8. Checked the final dataset for missing values.

### Final Dataset

After preprocessing:

- **Rows:** 1,470
- **Features:** 45

---

## Train-Test Split

The dataset was divided into training and testing sets:

- **Training Data:** 80% — 1,176 records
- **Testing Data:** 20% — 294 records
- **Random State:** 42
- **Stratification:** Applied using the target variable

---

## Machine Learning Models

Three classification models were trained and compared.

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

Feature scaling was applied using `StandardScaler` before training the model.

```python
LogisticRegression(max_iter=1000, random_state=42)
```

### 2. Decision Tree

A Decision Tree classifier was used to learn decision rules from the employee data.

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)
```

### 3. Random Forest

Random Forest was used as an ensemble classification model consisting of multiple decision trees.

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

---

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|---------:|----------:|-------:|---------:|
| Logistic Regression | 86.05% | 61.54% | 34.04% | **43.84%** |
| Decision Tree | — | — | — | 24.62% |
| Random Forest | — | — | — | 16.95% |

> Logistic Regression achieved the highest F1 score among the three models.

---

## Best Performing Model

Based on the **F1 Score**, Logistic Regression performed the best.

**Logistic Regression F1 Score: 43.84%**

Although the Logistic Regression model achieved an accuracy of **86.05%**, the lower recall shows that some employees who actually leave the company were not correctly identified.

This is important because employee attrition data is imbalanced, with significantly more employees staying than leaving.

---

## Confusion Matrix

Confusion matrices were generated for all three models to analyze their classification performance.

For Logistic Regression:

```text
[[237, 10],
 [31, 16]]
```

Where:

- **True Negatives:** 237
- **False Positives:** 10
- **False Negatives:** 31
- **True Positives:** 16

---

## Feature Importance

Feature importance was analyzed using the **Random Forest** model.

The top features were visualized using a bar chart to identify which employee attributes contributed most to the model's predictions.

Some important features included:

- Monthly Income
- Age
- OverTime

A visualization of the **Top 15 Features** is included in the notebook.

---

## Visualizations

The project includes several visualizations:

- Employee attrition distribution
- F1 score comparison
- Confusion matrices
- Random Forest feature importance
- Top 15 important features

These visualizations help interpret and compare the performance of the developed models.

---

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**

---

## Project Structure

```text
Employee-Attrition-Analyzer/
│
├── Employee_Attrition_Analyzer.ipynb
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
└── README.md
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/abdulhayykhan/Employee-Attrition-Analyzer.git
```

### 2. Open the Project

Open the notebook:

```text
Employee_Attrition_Analyzer.ipynb
```

using Jupyter Notebook, JupyterLab, or Google Colab.

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Run the Notebook

Run the notebook cells sequentially to:

1. Load the dataset
2. Preprocess the data
3. Train the models
4. Generate predictions
5. Evaluate the models
6. Generate visualizations
7. Compare model performance

---

## Conclusion

The Employee Attrition Analyzer successfully applies foundational Machine Learning concepts to a real-world employee dataset.

Three classification models — **Logistic Regression, Decision Tree, and Random Forest** — were implemented and compared.

Based on the F1 Score, **Logistic Regression performed the best with an F1 score of 43.84%**.

The project demonstrates the complete machine learning workflow:

**Load → Clean → Drop → Encode → Split → Scale → Train → Predict → Evaluate → Compare → Visualize**

---

## Academic Context

This project was developed as part of the **Lab #7 Open Ended Lab** for the Machine Learning Practical course at Dawood University of Engineering & Technology (DUET).