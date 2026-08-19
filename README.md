# Smart Meter Asset Classifier

This repository contains a machine learning pipeline designed to identify residential distributed energy resources (DERs)—specifically Solar Photovoltaic (PV), Heat Pumps (HP), and Electric Vehicles (EV)—using standard 15-minute resolution smart meter data.

---

## Background

Grid operators and utilities frequently lack granular visibility into behind-the-meter asset adoption. Relying on physical sub-metering hardware to track these assets is expensive and difficult to scale. 

This project bypasses the need for sub-meters by extracting domain-specific behavioral signatures directly from raw time-series consumption and injection data. The pipeline uses these features to classify households into one of five distinct operational profiles:

*   **Baseline:** Standard household load without major DERs.
*   **EV_Only:** Electric Vehicle charging detected.
*   **PV_Only:** Solar generation and grid injection detected.
*   **EV_PV:** Both EV charging and Solar generation present.
*   **HP_PV:** Heat Pump operation and Solar generation present.

---

## Performance Summary

To ensure reliable evaluation, all models were trained and tested using stratified splits, preserving the real-world distribution of minority adoption groups across the dataset.

| Model | Macro-F1 | Accuracy | Key Takeaways |
| :--- | :--- | :--- | :--- |
| **LightGBM** | **0.855** | **86.9%** | Best overall performance. Excels at capturing non-linear patterns on hybrid asset classes (e.g., `EV_PV` F1: 0.94). |
| **Logistic Regression (Balanced)** | 0.807 | 80.0% | Highly interpretable baseline model. Shows robust identification for `Baseline` (0.95 F1) and `EV_PV` (0.86 F1). |

### Detailed Classification Metrics

Below is the classification report for the primary LightGBM model evaluated on the validation set:

```text
              precision    recall  f1-score   support

    Baseline       0.91      0.97      0.94        30
     EV_Only       0.88      0.70      0.78        10
       EV_PV       0.91      0.97      0.94        30
       HP_PV       0.80      0.80      0.80        30
     PV_Only       0.86      0.80      0.83        30

    accuracy                           0.87       130
   macro avg       0.87      0.85      0.86       130
