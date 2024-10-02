# Credit Card Fraud Detection Demo

Source: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]()

"The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions."

### **Interpretation of Metrics**

* **Confusion Matrix** : Shows the correct and incorrect predictions broken down by each class.
  * True Positives (TP): Frauds correctly identified.
  * True Negatives (TN): Normal transactions correctly identified.
  * False Positives (FP): Normal transactions incorrectly labeled as fraud.
  * False Negatives (FN): Frauds missed by the model.
* **Precision** : The proportion of positive identifications that were actually correct.
  * High precision minimizes false positives.
* **Recall (Sensitivity)** : The proportion of actual positives that were identified correctly.
  * High recall minimizes false negatives.
* **ROC AUC Score** : Measures the model's ability to distinguish between classes.
  * A score closer to 1 indicates better performance.
* **Precision-Recall AUC** : Useful for imbalanced datasets.
  * Focuses on the minority class performance.
