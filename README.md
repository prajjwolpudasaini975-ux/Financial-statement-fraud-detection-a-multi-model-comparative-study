# Financial statement fraud detection: a multi-model comparative study

Can machine learning catch financial statement fraud better than the classical accounting models? This project puts that question to the test using SEC enforcement data on Russell 3000 firms from 1995 to 2018.

## What this project does

The notebook trains and evaluates six fraud detection models on a panel of Russell 3000 firm-years labeled by SEC Accounting and Auditing Enforcement Releases (AAERs). The work breaks into three parts:

1. Three classical baselines (Beneish M-Score, Dechow F-Score, and a 71-variable logistic regression from Gepp et al.), each using its original prescribed variable set.

2. Three ML models (Random Forest, XGBoost, LightGBM) trained on the same 28 raw Compustat variables that Bao et al. (2020) used for RUSBoost. Each model is tested under class weighting, SMOTE, and random undersampling to see which imbalance strategy works best.

3. A SHAP-based variable selection step: take the top 20 variables by SHAP importance from the best Stage 1 model, retrain all three ML models on that reduced set, and compare performance against the full 28-variable versions.

On top of that, the project compares SHAP feature importance rankings across all three ML models to check whether they agree on which variables matter most, and runs a cost-weighted net benefit analysis using real-world investigation cost ratios.

## Project structure

```
FA_Python/
├── data/
│   ├── raw/                    # Original input datasets
│   │   ├── Cleaned_data_1995_2018.csv
│   │   └── sample_df.csv
│   └── processed/              # Model output datasets
│       └── final_dataset_with_predictions.csv
├── notebooks/
│   └── fraud_analytics_codes.ipynb   # Main analysis notebook
├── output/
│   ├── figures/                # Generated charts and plots
│   │   ├── fig_01_eda_imbalance.png
│   │   ├── fig_02_shap_importance.png
│   │   ├── fig_03_roc_pr_curves.png
│   │   ├── fig_04_shap_rank_heatmap.png
│   │   └── fig_05_emc_sensitivity.png
│   └── results/                # Evaluation tables
│       ├── results_model_comparison.csv
│       └── overfit_check.csv
├── requirements.txt
└── venv/
```

## Getting started

You need Python 3.10 or newer.

1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the notebook:
   ```bash
   jupyter notebook notebooks/fraud_analytics_codes.ipynb
   ```

## Dataset

The data comes from the Russell 3000 AAER panel compiled by Gepp et al. (2021), covering fiscal years 1995 through 2018. It has financial statement variables pulled from Compustat and binary fraud labels based on whether the SEC filed an enforcement action against a given firm-year.

Training uses FY1995 to FY2002. Testing uses FY2003 onward. This temporal split avoids look-ahead bias: a model trained in 2003 would only have had access to pre-2003 data.

## Models

| Model | Type | Variable set |
|---|---|---|
| M-Score | Probit regression | 8 Beneish ratio indices |
| F-Score | Logistic regression | 9 Dechow components |
| LR-71 | Logistic regression | 71 variables (Gepp et al.) |
| Random Forest | Ensemble (bagging) | 28 Compustat variables |
| XGBoost | Ensemble (boosting) | 28 Compustat variables |
| LightGBM | Ensemble (boosting) | 28 Compustat variables |

All models are evaluated on ROC-AUC, PR-AUC, F1, Expected Misclassification Cost, and Cost-Weighted Net Benefit.

## References

- Bao, Y., Ke, B., Li, B., Yu, Y. J., & Zhang, J. (2020). Detecting accounting fraud in publicly traded U.S. firms using a machine learning approach. *Journal of Accounting Research*, 58(1), 199-235.
- Beneish, M. D. (1999). The detection of earnings manipulation. *Financial Analysts Journal*, 55(5), 24-36.
- Dechow, P. M., Ge, W., Larson, C. R., & Sloan, R. G. (2011). Predicting material accounting misstatements. *Contemporary Accounting Research*, 28(1), 17-82.
- Gepp, A., Kumar, K., & Bhattacharya, S. (2021). Lifting the numbers game: Identifying key input variables and a best-performing model to detect financial statement fraud. *Accounting & Finance*, 61, 4601-4638.
- Perols, J. L., Bowen, R. M., Zimmermann, C., & Samba, B. (2017). Finding needles in a haystack: Using data analytics to improve fraud prediction. *The Accounting Review*, 92(2), 221-245.
