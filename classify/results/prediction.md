
| Classifier      | Accuracy | F Score | Precision | Recall |
|-----------------|----------|-----------|--------|---------|
| LDADE + DecTree | 0.31 +- 0.01 | 0.32 +- 0.02 | 0.35 +- 0.02 | 0.35 +- 0.02 | 
| LDADE + LinReg  | 0.45 +- 0.02 | 0.42 +- 0.02 | 0.40 +- 0.02 | 0.45 +- 0.02 |
| LDADE + SVM     | 0.46 +- 0.01 | 0.42 +- 0.01 | 0.41 +- 0.03 | 0.46 +- 0.01 |
| LDADE + RF      | 0.38 +- 0.02 | 0.37 +- 0.01 | 0.38 +- 0.01 | 0.38 +- 0.02 |
| LDADE + NBayes  | 0.46 +- 0.00 | 0.29 +- 0.00 | 0.21 +- 0.00 | 0.46 +- 0.00 |
| TFIDF + DecTree | 0.48 +- 0.01 | 0.47 +- 0.01 | 0.47 +- 0.01 | 0.48 +- 0.01 |
| TFIDF + LinReg  | **0.55 +- 0.01** | 0.49 +- 0.01 | **0.54 +- 0.01** | **0.55 +- 0.01** |
| TFIDF + SVM     | **0.55 +- 0.01** | 0.48 +- 0.01 | 0.53 +- 0.02 | **0.55 +- 0.01** |
| TFIDF + RF      | **0.55 +- 0.01** | **0.52 +- 0.01** | 0.51 +- 0.03 | **0.55 +- 0.01** |
| TFIDF + NBayes  | 0.53 +- 0.01 | **0.52 +- 0.01** | 0.52 +- 0.02 | 0.53 +- 0.01 |


### decision_tree - pre_process_tfidf
Accuracy:0.48+-0.01; F Score:0.47+-0.01; Precision:0.47+-0.01; Recall:0.48+-0.01
### decision_tree - pre_process_pruned_tfidf
Accuracy:0.54+-0.02; F Score:0.50+-0.02; Precision:0.49+-0.02; Recall:0.54+-0.02
### decision_tree - pre_process_ldade
Accuracy:0.31+-0.01; F Score:0.32+-0.02; Precision:0.35+-0.03; Recall:0.31+-0.01
### logistic_regression - pre_process_tfidf
Accuracy:0.55+-0.01; F Score:0.49+-0.01; Precision:0.54+-0.03; Recall:0.55+-0.01
### logistic_regression - pre_process_pruned_tfidf
Accuracy:0.54+-0.02; F Score:0.47+-0.02; Precision:0.49+-0.03; Recall:0.54+-0.02
### logistic_regression - pre_process_ldade
Accuracy:0.45+-0.02; F Score:0.42+-0.02; Precision:0.40+-0.02; Recall:0.45+-0.02
### svm - pre_process_tfidf
Accuracy:0.55+-0.01; F Score:0.48+-0.01; Precision:0.53+-0.02; Recall:0.55+-0.01
### svm - pre_process_pruned_tfidf
Accuracy:0.54+-0.02; F Score:0.46+-0.02; Precision:0.48+-0.03; Recall:0.54+-0.02
### svm - pre_process_ldade
Accuracy:0.46+-0.01; F Score:0.42+-0.01; Precision:0.41+-0.03; Recall:0.46+-0.01
### random_forest - pre_process_tfidf
Accuracy:0.55+-0.01; F Score:0.52+-0.01; Precision:0.51+-0.03; Recall:0.55+-0.01
### random_forest - pre_process_pruned_tfidf
Accuracy:0.54+-0.02; F Score:0.50+-0.02; Precision:0.51+-0.02; Recall:0.54+-0.02
### random_forest - pre_process_ldade
Accuracy:0.38+-0.02; F Score:0.37+-0.01; Precision:0.38+-0.01; Recall:0.38+-0.02
### naive_bayes - pre_process_tfidf
Accuracy:0.53+-0.01; F Score:0.52+-0.01; Precision:0.52+-0.02; Recall:0.53+-0.01
### naive_bayes - pre_process_pruned_tfidf
Accuracy:0.55+-0.02; F Score:0.52+-0.03; Precision:0.50+-0.02; Recall:0.55+-0.02
### decision_tree - pre_process_ldade
Accuracy:0.31+-0.01; F Score:0.32+-0.02; Precision:0.35+-0.03; Recall:0.31+-0.01
