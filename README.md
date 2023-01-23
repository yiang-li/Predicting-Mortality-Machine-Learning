# Project1
 
## Predicting Mortality of the Older US Adults in a Nationally Representative Survey

## Authors: Yiang Li and Zejian Lv

This is a project for MACS 30100 Perspectives in Computational Modeling at the University of Chicago.

The goal of this project is to train a binary classifier to predict the mortality of the older US Adults in National Social and Health Survey (NSHAP) Wave 1 and Wave 3, which is a nationally representative survey of the older US adults (aged 57-85) in 2010 and 2020.

The link to the dataset: https://www.icpsr.umich.edu/web/NACDA/series/706

Present academic literature in predicting mortality has primarily focused on disease and frailty, although social, behavioral, and psychological statues may herald broad physiological decline. We aim to predict mortality using a comprehensive set of predictors, including allostatic load measures, health behaviors, socio-demographic status, and health status. The findings of this study may help to identify the older adults who are at risk of mortality and to develop interventions to improve their health.

We use the following models:
    1. Logistic regression
    2. Random forest/Decision tree
    3. Support vector machine

We use 19 predictors:
    1. Allostatic load measures: diastolic and systolic blood pressure (binary), body mass index (categorical), glycated hemoglobin (binary), C-reactive protein (binary), and hormone dehydroepiandrosterone (binary)
            These measures were selected based on the following papers: https://academic.oup.com/psychsocgerontology/article/76/3/574/5703612 and https://academic.oup.com/psychsocgerontology/article/64B/suppl_1/i67/552266
    2. Health behaviors: smoking (categorical), alcohol consumption (binary), and physical activity (categorical)
    3. Socio-demographic status: education (categorical), net household assets (categorical), and marital status (binary)
    4. Health status: self-rated physical (categorical) and mental health (categorical), chronic COPD conditions (binary), and functional limitations (binary)
    
Our target variable is mortality (binary). There are 3005 observations in NSHAP with 893 deaths, which is a 29.7% mortality rate. The data is imbalanced (893 positive, 2112 negative), and we use the following methods to deal with the imbalance:
    1. Undersampling
    2. Oversampling
    3. SMOTE
    4. ADASYN
    
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