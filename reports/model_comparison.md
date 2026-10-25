# Model Comparison (test set)

Majority-class baseline on test: 0.610

| Model | Accuracy | Baseline | Recall (unsafe) | F1 | ROC AUC | Cost (dangerous=10, harmless=1) | CV accuracy (train) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | 0.622 | 0.61 | 0.853 | 0.351 | 0.573 | 779 | 0.639 +/- 0.016 |
| k-Nearest Neighbours | 0.633 | 0.61 | 0.863 | 0.367 | 0.617 | 736 | 0.632 +/- 0.018 |
| Naive Bayes | 0.605 | 0.61 | 0.875 | 0.266 | 0.6 | 709 | 0.623 +/- 0.008 |


Best model by AUC: k-Nearest Neighbours

Chosen threshold (from train out-of-fold cost sweep): 0.8500000000000001

Field-kit features: ph, Turbidity, Conductivity, Solids

Field-kit AUC: 0.539 vs full AUC 0.617 (loss 0.078)