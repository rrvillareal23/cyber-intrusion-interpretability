# cyber-intrusion-interpretability
# 🔐 Competition 2: Interpreting Cyber Intrusion Detection Models Using SHAP

## 🧠 Overview

This project is part of a cybersecurity data science competition focused on building interpretable machine learning models for detecting cyber intrusions using SHAP (SHapley Additive exPlanations). We analyze the UNSW-NB15 dataset, engineer features, train advanced models, and interpret model decisions to improve transparency and fairness.

---

## 🛠️ Frameworks, Libraries, and Skills Used

| Category                 | Tools & Libraries                                                                 |
|--------------------------|-----------------------------------------------------------------------------------|
| **Data Handling**        | `pandas`, `numpy`                                                                |
| **Visualization**        | `matplotlib`, `seaborn`, `SHAP`                                                 |
| **Machine Learning**     | `scikit-learn`, `xgboost`, `imbalanced-learn`                                   |
| **Model Stacking**       | `StackingClassifier` from `scikit-learn`                                        |
| **Hyperparameter Tuning**| `RandomizedSearchCV`                                                             |
| **Explainability**       | `SHAP` for global and local interpretability                                     |
| **Data Engineering**     | Custom feature engineering (`sload_dload_ratio`, `total_load`, `dur_per_byte`)  |

---

## 📁 Project Structure

### Part 1: Data Exploration and Model Training

- **Data Loading**: Read in UNSW-NB15 training and test datasets.
- **Feature Engineering**: Created interaction features to capture load and duration dynamics.
- **Preprocessing**: Label encoding and one-hot encoding of categorical features.
- **Balancing**: Applied `SMOTE` and `RandomUnderSampler` to address class imbalance.
- **Model Training**: Trained both `XGBoost` and `RandomForest` using `RandomizedSearchCV`.
- **Model Ensembling**: Combined top-performing models using a `StackingClassifier`.

### Part 2: Model Interpretability with SHAP

- **Global Interpretability**: Used `TreeExplainer` with SHAP to generate a global summary plot.
- **Class-Specific Analysis**: Created SHAP plots for specific attack types (e.g., Backdoor).
- **Dependence Plot**: Visualized how key features affect predictions for specific attack classes.

### Part 3: Building Trust and Ensuring Fairness

- **Transparency Discussion**: Highlighted the role of SHAP in explaining model predictions in cybersecurity.
- **Recommendations**:
  - Address class imbalance via augmentation and oversampling.
  - Use SHAP outputs to mitigate model biases.
  - Continuously update models with new data for fairness and accuracy.

---

## 📊 Model Evaluation

- **Metrics Used**: `classification_report`, `roc_auc_score`, `confusion_matrix`
- **Performance**: Ensemble model showed improved F1 scores across multiple attack classes.
- **Explainability**: SHAP revealed key drivers behind predictions, aiding in debugging and trust-building.

---

## 🔍 Key Takeaways

- Feature interaction terms significantly boosted model performance.
- Ensemble learning improves generalization and stability.
- SHAP is a powerful tool for both global and local interpretability.
- Transparent models are crucial in cybersecurity to ensure trust and fairness.

---

## 🚀 Future Work

- Automate data pipeline for real-time intrusion detection.
- Integrate with cybersecurity dashboards (e.g., SIEM tools).
- Extend SHAP analysis to other black-box models like deep neural networks.

---

## 📬 Contact

For questions or collaboration, reach out to any team member or open an issue in this repository.

