# Credit Risk Analysis and Prediction

### General architecture
TODO

### Initial exploration
Check the `notebooks` directory. There is jupyter notebook that perform initial data exploration before jumping into model training.
The `src/initial_exploration.py` modularize the content in the jupyter notebook and save the predictions of each model for further analysis. The `src/stat_tests.py` performs MacNemar's test to compare models pair-wise (Logistic regression vs. SVC, Logistic regression vs. XGBoost), and then we perform bootstrap sampling to calculate the confidence interval and effect size to select a model for deployment. 

