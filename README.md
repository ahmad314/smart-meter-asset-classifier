# smart-meter-asset-classifier

A machine learning pipeline to identify residential energy assets: **Solar Photovoltaic (PV)**, **Heat Pumps (HP)**, and **Electric Vehicles (EV)**—directly from 15-minute resolution smart meter consumption and injection data.

---

## Overview

Grid operators and utilities often lack granular visibility into distributed energy resource (DER) adoption behind the meter. This project extracts domain-informed behavioral signatures from raw time-series data to classify households into five distinct asset categories without requiring sub-metering hardware:
- **Baseline** (Standard household load)
- **EV_Only** (Electric Vehicle charging)
- **PV_Only** (Solar generation & injection)
- **EV_PV** (Electric Vehicle + Solar generation)
- **HP_PV** (Heat Pump heating/cooling + Solar generation)

---

## Performance Summary

Models were evaluated using Stratified Validation and Test splits to preserve class distributions across minority adoption groups.

| Model | Macro-F1 | Accuracy | Key Strengths |
| :--- | :--- | :--- | :--- |
| **LightGBM** | **0.855** | **86.9%** | Best overall performance; strong non-linear pattern capture on hybrid asset classes (`EV_PV` F1: 0.94). |
| **Logistic Regression (Balanced)** | **0.807** | **80.0%** | Highly interpretable baseline; robust identification of `Baseline` (0.95 F1) and `EV_PV` (0.86 F1). |

### Detailed Classification Metrics

#### LightGBM (Validation Set)
```text
              precision    recall  f1-score   support
    Baseline       0.91      0.97      0.94        30
     EV_Only       0.88      0.70      0.78        10
       EV_PV       0.91      0.97      0.94        30
       HP_PV       0.80      0.80      0.80        30
     PV_Only       0.86      0.80      0.83        30

    accuracy                           0.87       130
   macro avg       0.87      0.85      0.86       130
