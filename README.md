# Project1
 
## Predicting Mortality of the Older US Adults in a Nationally Representative Survey

## Authors: Yiang Li and Zejian Lv

This is a project for MACS 30100 Perspectives in Computational Modeling at the University of Chicago.

The goal of this project is to train a binary classifier to predict the mortality of older US Adults in National Social and Health Survey (NSHAP) Wave 1 and Wave 3, which is a nationally representative survey of the older US adults (aged 57-85) in 2010 and 2020.

The link to the dataset: https://www.icpsr.umich.edu/web/NACDA/series/706

Present academic literature in predicting mortality has primarily focused on disease and frailty, although social, behavioral, and psychological statuses may herald broad physiological decline. We aim to predict mortality using a comprehensive set of predictors, including allostatic load measures, health behaviors, socio-demographic status, and health status. The findings of this study may help to identify older adults who are at risk of mortality and to develop interventions to improve their health.

We use the following models:
    1. Logistic regression
    2. Random forest/Decision tree

We use 19 predictors:
    1. Allostatic load measures: systolic blood pressure (continuous), body mass index (continuous), glycated hemoglobin (continuous), C-reactive protein (continuous), and hormone dehydroepiandrosterone (continuous)
            These measures were selected based on the following papers: https://academic.oup.com/psychsocgerontology/article/76/3/574/5703612 and https://academic.oup.com/psychsocgerontology/article/64B/suppl_1/i67/552266
    2. Health behaviors: smoking (binary), alcohol consumption (binary), physical activity (binary), and sleep duration (ordinal) 
    3. Socio-demographic status: age (ordinal), education (ordinal), net household assets (ordinal), marital status (binary), sex (binary), and race (binary)
    4. Health status: self-rated physical (ordinal) and mental health (ordinal), chronic COPD conditions (binary), and functional limitations (binary)
    
Our target variable is mortality (binary). There are 3005 observations in NSHAP with 893 deaths, which is a 29.7% mortality rate. The data is imbalanced (893 positive, 2112 negative), and we use the following methods to deal with the imbalance:
    1. Undersampling
    2. Oversampling
    3. Synthetic minority oversampling technique (SMOTE)
    4. Adaptive synthetic sampling approach for imbalanced learning (ADASYN)
    5. Class weights
    6. Cost-sensitive learning
    7. Ensemble methods
    8. Penalized methods
    
We use the following metrics:
    1. Accuracy
    2. Precision
    3. Recall
    4. F1 score
    5. AUC

We use the following packages:
    1. pandas
    2. numpy
    3. sklearn
    4. matplotlib
    5. seaborn
    6. keras
    7. tensorflow

Previous paper predicts 5-year mortality with olfactory dysfunction (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0107541#s2)