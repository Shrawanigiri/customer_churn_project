# Customer Churn Prediction Using Machine Learning
**3rd-Year B.Tech Machine Learning Mini-Project (AI & Data Science)**

---

## 📌 1. Project Objective
Customer churn represents the loss of subscribers or clients who discontinue their service with a company. For telecommunication service providers, acquiring new customers costs significantly more than retaining existing ones. 

The objective of this project is to build an end-to-end Machine Learning solution that:
1. Analyzes customer demographic, service, and billing patterns.
2. Identifies key factors associated with customer churn.
3. Trains a supervised classification model (**Random Forest Classifier**) to predict whether a customer is likely to churn or stay.
4. Provides an interactive web dashboard for real-time predictions and stakeholder decision support.

---

## 📊 2. Dataset Overview
* **Source**: `dataset.csv` (Telecommunications customer records)
* **Total Observations**: 30 rows
* **Total Features**: 21 columns (20 input features + 1 target variable)
* **Missing Values**: 0 (no imputation required)
* **Duplicate Rows**: 0
* **Target Variable**: `Churn`
  * Class 0 (`No`): 15 customers (50.0% retention)
  * Class 1 (`Yes`): 15 customers (50.0% churn)
  * *Distribution*: Perfectly balanced binary classification
* **Identifier Column**: `customerID` (unique string IDs `C001`–`C030`, dropped during preprocessing)

### Important Attributes:
* **Account Information**: `tenure` (months with company), `Contract` (Month-to-month, One year, Two year), `PaperlessBilling`, `PaymentMethod`.
* **Services Subscribed**: `PhoneService`, `MultipleLines`, `InternetService` (DSL, Fiber optic, No), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
* **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
* **Financials**: `MonthlyCharges`, `TotalCharges`.

---

## 🔄 3. Project Methodology
The project follows a standard, rigorous data science lifecycle:
$$\text{Dataset} \longrightarrow \text{Preprocessing} \longrightarrow \text{EDA} \longrightarrow \text{Train/Test Split} \longrightarrow \text{Random Forest Training} \longrightarrow \text{Evaluation} \longrightarrow \text{Feature Importance} \longrightarrow \text{Streamlit Dashboard}$$

1. **Preprocessing**:
   * Removed `customerID` to eliminate non-generalizable noise.
   * Converted target `Churn` to binary integers (`Yes` $\to$ 1, `No` $\to$ 0).
   * One-hot encoded categorical columns using `pd.get_dummies(..., drop_first=True)` to prevent the dummy variable trap (multicollinearity).
   * Preserved numeric attributes without scaling because tree-based models are invariant to monotonic feature scaling.
2. **Exploratory Data Analysis (EDA)**:
   * Visualized target balance, tenure distributions, monthly charge differences, and contract type trends.
3. **Train-Test Split**:
   * 80% Training set (24 observations) / 20% Testing set (6 observations).
   * Applied `stratify=y` to preserve the 50/50 churn proportion across both subsets.
   * Fixed `random_state=42` for exact reproducibility.
4. **Model Training**:
   * Trained a `RandomForestClassifier` with 100 estimators (`n_estimators=100`, `random_state=42`).
5. **Cross-Validation & Evaluation**:
   * Performed Stratified 5-Fold Cross-Validation strictly on the training set.
   * Evaluated final performance on the held-out test set using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
6. **Feature Importance**:
   * Computed Mean Decrease in Impurity (Gini importance) to determine which features most effectively split classes.
7. **Interactive Deployment**:
   * Built a Streamlit web application (`app.py`) allowing manual input and preset evaluations.

---

## 🛠️ 4. Technologies Used
* **Python 3.13**: Core programming language
* **Pandas**: Data manipulation, filtering, and one-hot encoding
* **NumPy**: Numerical operations and vector processing
* **Scikit-learn**: Machine learning model (`RandomForestClassifier`), metrics, and cross-validation
* **Matplotlib & Seaborn**: Statistical visualizations and confusion matrix heatmaps
* **Streamlit**: Web dashboard interface
* **Joblib**: Model and column serialization

---

## 📂 5. Project Structure
```text
customer_churn_project/
│
├── dataset.csv          # Single source of truth dataset
├── model.py             # Preprocessing, model training, evaluation & prediction logic
├── app.py               # Interactive Streamlit dashboard
├── requirements.txt     # Minimal dependencies list
├── churn_model.pkl      # Trained Random Forest model file
├── model_columns.pkl    # Encoded feature schema
└── README.md            # Complete documentation and viva prep guide
```

---

## 🚀 6. How to Run

### Step 1: Install Dependencies
```bash
python -m pip install -r requirements.txt
```
*(On Windows systems where the Python Launcher is used, replace `python` with `py`)*

### Step 2: (Optional) Run Model Script from Terminal
```bash
python model.py
```
This trains the model, computes cross-validation on the training set, evaluates the held-out test set, and updates `churn_model.pkl`.

### Step 3: Launch the Streamlit Dashboard
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌲 7. Machine Learning Model: Random Forest
Random Forest is an ensemble supervised learning algorithm based on **Bagging (Bootstrap Aggregation)**:
1. **Ensemble of Trees**: It constructs 100 individual decision trees (`n_estimators=100`).
2. **Bootstrap Sampling**: Each decision tree is trained on a random sample of the training data drawn with replacement.
3. **Random Feature Subspace**: At each split in a tree, only a random subset of features is considered, ensuring low correlation between individual trees.
4. **Majority Voting**: For classification, predictions from all 100 trees are aggregated via majority voting, and class probabilities represent the percentage of trees voting for each class.

---

## 📈 8. Actual Model Results (Calculated from Data)

### A. Test Set Performance (6 Held-Out Samples)
* **Accuracy**: **100.0%** (1.0000)
* **Precision**: **100.0%** (1.0000)
* **Recall**: **100.0%** (1.0000)
* **F1-Score**: **100.0%** (1.0000)
* **5-Fold Cross-Validation Accuracy (on Training Set)**: **100.0%** ($\pm 0.0\%$)

### B. Confusion Matrix
$$
\begin{bmatrix}
\text{TN = 3} & \text{FP = 0} \\
\text{FN = 0} & \text{TP = 3}
\end{bmatrix}
$$

### C. Top Features for Classification
1. **`tenure`**: `19.67%` — Retained customers average 51.0 months tenure vs 11.1 months for churners.
2. **`MonthlyCharges`**: `13.71%` — Churners average $85.82/month vs $63.42/month for retained accounts.
3. **`PaperlessBilling_Yes`**: `12.11%` — Strongly correlated with modern online billing profiles.
4. **`TotalCharges`**: `11.88%` — Reflects cumulative revenue and account duration.
5. **`OnlineSecurity_Yes`**: `8.64%` — Add-on service usage is higher among retained customers.
6. **`InternetService_Fiber optic`**: `8.00%` — Fiber users exhibited higher churn rates than DSL users.

---

## ⚠️ 9. Academic Dataset Limitation
* **Sample Size Disclosure**: The dataset consists of 30 customer records, resulting in 24 training observations and 6 test observations.
* **Separability**: On this specific dataset, features such as `Contract` (all Month-to-month customers churned; all 1-year and 2-year customers stayed) and `tenure` perfectly divide the classes.
* **Generalization Note**: While 100% accuracy is mathematically valid on this specific data split, it must **not** be presented as evidence of production-level perfection. Real-world telecom datasets (e.g., thousands of customers) have noise, overlapping boundaries, and edge cases, where accuracy typically ranges between 78% and 85%.

---

## 🎓 10. Comprehensive Viva Guide: Questions & Answers

### 1. Why did you choose Random Forest over a single Decision Tree or Logistic Regression?
**Answer**: A single Decision Tree is prone to high variance and overfitting. Random Forest combines an ensemble of decision trees using bagging (bootstrap aggregation) and random feature subspace sampling, which significantly reduces variance and produces more robust predictions. Furthermore, unlike Logistic Regression, Random Forest does not assume linear decision boundaries and handles interactions between categorical and continuous variables naturally.

### 2. Why did you remove `customerID`?
**Answer**: `customerID` is an arbitrary unique string assigned to each customer. It has zero generalization capability. If retained, the model might create artificial split rules based on unique IDs, leading to severe overfitting and data leakage.

### 3. Why is this a classification problem?
**Answer**: The goal is to predict a categorical outcome: whether a customer will leave (`Churn = Yes`) or stay (`Churn = No`). Because the target consists of discrete categories rather than a continuous numerical value (which would be regression), this is a binary classification problem.

### 4. What is the target variable?
**Answer**: The target variable is `Churn`. In our dataset, it has two classes: `Yes` (customer churned, encoded as 1) and `No` (customer retained, encoded as 0).

### 5. Why did you use a train-test split?
**Answer**: To evaluate how well the trained model generalizes to unseen data. Evaluating the model on the same data it was trained on produces overly optimistic results (data memorization). The held-out test set simulates incoming future customers.

### 6. Why use `stratify=y`?
**Answer**: Stratification guarantees that the proportion of classes in the target variable is preserved in both the training set and testing set. In our dataset, 50% of customers churn and 50% stay. With `stratify=y`, both the training set (24 rows) and test set (6 rows) maintain exactly a 50/50 balance (3 Churn, 3 Stay in test).

### 7. What does `random_state=42` do?
**Answer**: It seeds the pseudo-random number generator used in data splitting and tree construction. Setting a fixed random seed ensures reproducibility, meaning running the code multiple times will yield identical splits, tree selections, and evaluation metrics.

### 8. What is Precision?
**Answer**: Precision measures the accuracy of positive predictions:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
Out of all customers that the model flagged as "Likely to Churn", precision represents what fraction actually churned. High precision means very few false alarms.

### 9. What is Recall?
**Answer**: Recall (Sensitivity) measures the coverage of actual positive cases:
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
Out of all customers who actually churned, recall represents what fraction the model successfully caught. In churn prediction, **Recall is usually the most important metric** because failing to identify a churning customer (False Negative) means the company loses that customer without any retention effort.

### 10. What is F1-Score?
**Answer**: The F1-score is the harmonic mean of Precision and Recall:
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
It provides a single balanced score that penalizes extreme imbalances between Precision and Recall.

### 11. What is a Confusion Matrix?
**Answer**: A $2 \times 2$ contingency table comparing actual target classes against predicted classes:
* **True Positive (TP)**: Actual Churn, Predicted Churn.
* **True Negative (TN)**: Actual Stay, Predicted Stay.
* **False Positive (FP)**: Actual Stay, Predicted Churn (Type I Error).
* **False Negative (FN)**: Actual Churn, Predicted Stay (Type II Error).

### 12. What is Feature Importance in Random Forest?
**Answer**: Feature importance (Gini importance or Mean Decrease in Impurity) measures how much a feature's split nodes reduce impurity across all 100 decision trees. Features that frequently split high-impurity nodes into pure subsets receive higher importance scores.

### 13. Does feature importance imply causation?
**Answer**: **No.** Feature importance only indicates statistical association and predictive utility within the mathematical model. For example, having a `Month-to-month` contract was heavily used by the trees to separate churners, but this does not prove that the contract itself caused the customer to leave. Correlation does not imply causation.

### 14. Why doesn't Random Forest require feature scaling?
**Answer**: Decision trees make orthogonal axis-parallel splits based on feature ordering (e.g., $X_j \le \text{threshold}$). Because the split criterion depends only on the relative order of values rather than distance or gradient magnitude (unlike KNN, SVM, or Neural Networks), scaling or normalizing features has zero effect on tree splits.

### 15. What is the primary limitation of this dataset?
**Answer**: The dataset has only 30 records (6 in the test set). Because the sample size is small and classes are cleanly separated by tenure and contract type, the model achieves 100% test accuracy. However, this sample size cannot capture the full variance and edge cases found in large telecommunication populations.

### 16. How could this project be improved with a larger real-world dataset?
**Answer**:
1. Incorporate time-series usage trends (e.g., sudden drops in data usage or call frequency over the last 30 days).
2. Integrate customer support ticket logs and sentiment analysis of complaints.
3. Apply cost-sensitive learning to optimize retention budget allocation based on Customer Lifetime Value (CLV).
