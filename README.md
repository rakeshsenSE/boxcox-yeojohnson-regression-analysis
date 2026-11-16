# 🏗️ Concrete Compressive Strength Prediction: Power Transformation Analysis

An end-to-end machine learning feature engineering and regression analysis project evaluating the impact of **Box-Cox** and **Yeo-Johnson** power transformations on predicting concrete compressive strength.

---

## 📌 Project Overview
Real-world tabular datasets often contain skewed feature distributions that degrade the performance of linear models. In this project, we analyze the **Concrete Compressive Strength Dataset** (`1,030` samples, `9` attributes) and compare the baseline `LinearRegression` model against models trained on Gaussian-like transformed features using `PowerTransformer`.

---

## 📊 Dataset Features
* **Input Features (`8`):** Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, and Age (days).
* **Target Variable (`1`):** Compressive Strength (`Strength`).
* **Data Quality:** Verified `0` missing/null values across all `1,030` rows.

---

## 🛠️ Workflow & Methodology
1. **Exploratory Data Analysis (EDA) & Distribution Checks:**
   * Visualized feature skewness and normality before transformation using `distplot` (KDE/Histograms) and **Q-Q Plots** (`scipy.stats.probplot`).
2. **Baseline Model:**
   * Trained a standard `LinearRegression` model on raw features (`80/20` train-test split, `random_state=42`) and evaluated using $R^2$ Score and 5-Fold Cross-Validation.
3. **Box-Cox Transformation:**
   * Applied `PowerTransformer(method='box-cox')` with a small positive shift (`+ 0.000001`) to handle zero values, extracted optimal $\lambda$ (lambda) parameters, and compared pre/post-transformation distributions.
4. **Yeo-Johnson Transformation:**
   * Applied `PowerTransformer(method='yeo-johnson')` directly on the dataset (natively supporting zero/negative values), evaluated model performance, and performed a side-by-side $\lambda$ parameter comparison.

---

## 📈 Key Results & Model Comparison

| Model / Transformation Pipeline | Test $R^2$ Score | Cross-Val Mean $R^2$ |
| :--- | :---: | :---: |
| **Baseline Linear Regression** (Raw Data) | `0.6276` | `0.4609` |
| **Box-Cox + Linear Regression** | `0.8048` | `0.6663` |
| **Yeo-Johnson + Linear Regression** | **`0.8161`** | **`0.6836`** |

> **Key Takeaway:** Applying power transformations significantly normalized skewed features (such as *Age*, *Superplasticizer*, and *Slag*), boosting the test $R^2$ score from **~0.63 to ~0.82** (+18.8% improvement) and substantially improving cross-validation stability.

---

## 🧰 Tech Stack
* **Language:** Python 3.13
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning & Stats:** Scikit-Learn (`LinearRegression`, `PowerTransformer`, `train_test_split`, `cross_val_score`, `r2_score`), SciPy (`stats`)

---

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/rakeshsenSE/concrete-strength-power-transformation.git](https://github.com/rakeshsenSE/concrete-strength-power-transformation.git)
   cd concrete-strength-power-transformation
