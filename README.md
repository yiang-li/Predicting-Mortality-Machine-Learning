### MACS-30100 "Perspectives on Computational Modeling" Project-1

Due on **Feb 5th Sunday 11:00 pm**.

For submission on GradeScope, please only submit your **Jupyter notebook**.

For submission of the whole project, please push your Jupyter notebook (make sure to add comments for explanation, following the style of previously released coding quizzes) and your slides (< 10pages) to this GitHub repo (here), name the repo as **30100_P1_YourLastFirstName**.

Please record your presentation and upload the shared **link** to your recording (< 6min) here: 

https://youtu.be/HEdK6qYsY_Q

----------------------

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

We use 24 predictors:
    1. Allostatic load measures: systolic blood pressure (continuous), body mass index (continuous), glycated hemoglobin A1C (continuous), C-reactive protein (continuous), and hormone dehydroepiandrosterone (continuous)
            These measures were selected based on the following papers: https://academic.oup.com/psychsocgerontology/article/76/3/574/5703612 and https://academic.oup.com/psychsocgerontology/article/64B/suppl_1/i67/552266
    2. Health behaviors: smoking (binary), alcohol consumption (binary), physical activity (binary), and sleep duration (ordinal) 
    3. Socio-demographic status: age (ordinal), education (ordinal), net household assets (ordinal), marital status (binary), sex (binary), and race (binary)
    4. Health status: self-rated physical (ordinal) and mental health (ordinal), lung COPD conditions (binary), and functional limitations (binary)
    5. Network measures: size of social network (continuous), number of unique social contacts (continuous), proportion of social contacts living together (continuous), average frequency of talking to social contacts (continuous), and average closeness to social contacts (continuous)
            These measures were selected based on the following paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6110003/
    
Our target variable is mortality (binary). There are 3005 observations in NSHAP with 893 deaths, which is a 29.7% mortality rate. The data is imbalanced (893 positive, 2112 negative), and we use the following methods to deal with the imbalance:
    1. Undersampling
    2. Oversampling: Synthetic minority oversampling technique (SMOTE)
    3. Class weights
    4. Ensemble methods
    
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

Previous paper predicts 5-year mortality with olfactory dysfunction (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0107541#s2)

----------------------

Here are the **detailed instructions** for creating your presentation (previously posted on Ed):

Please organize your presentation from following four aspects (task, data, model, result). Try to summarize each point in a clear and logical way (in technical presentation, smooth logic is the most important thing), try to make your presentation short (<6min) and to the point.

- **Task**:
	- summarize your task in less than 3 sentences
	- is it classification (binary or multi-class) or regression?
- **Data**:
	- What does your data looks like? (e.g., show your dataframe)
	- What's the data size? Is there any missing value? How do you deal with missing values?
	- If it's classification, what is the class distribution (e.g., how many samples in each class, if it's unbalanced, how do you deal with that). If it's regression, what is the regression value distribution (e.g., histogram of the values)?
	- What's your feature space? How many features? What are the data types of the features? (e.g., 10 features in total, 8 numerical, 2 categorical)
	- What data preprocessing steps you have done? (e.g., encoding for categorical features, scaling for numerical features, vectorize text features)
	- How do you get training and testing set? 
- **Model**:
	- Which model(s) did you use?
	- For each model, which parameter setting do you apply (if you are not using the default value)?
	- What is the model performance? (e.g., show the score or classification report)
- **Result analysis**:
	- Interpret the model performance (e.g., top coefficient features for each class, or what does the decision tree looks like)
	- Show examples where the classifier fails (e.g., the test samples that the classifier make wrong predictions)
	- Error analysis: explain where do you think the errors come from and why? Do you have any possible solution to improve the model performance?

Please remember that: 
- try to summarize each point in a clear and logical way (in technical presentation, smooth logic is the most important thing)
- try to make your presentation short and to the point, and avoid the wordy way