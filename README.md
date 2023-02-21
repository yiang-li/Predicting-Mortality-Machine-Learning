### MACS-30100 "Perspectives on Computational Modeling" Project-1

Due on **Feb 5th Sunday 11:00 pm**.

For submission on GradeScope, please only submit your **Jupyter notebook**.

For submission of the whole project, please push your Jupyter notebook (make sure to add comments for explanation, following the style of previously released coding quizzes) and your slides (< 10pages) to this GitHub repo (here), name the repo as **30100_P1_YourLastFirstName**.

Please record your presentation and upload the shared **link** to your recording (< 6min) here: 

https://youtu.be/HEdK6qYsY_Q

---------------------- 
## Predicting Mortality of the Older US Adults in a Nationally Representative Survey

## Authors: Yiang Li and Zejian Lv

This is a project for MACS 30100 Perspectives in Computational Modeling at the University of Chicago.

The goal of this project is to train a binary classifier to predict the mortality of older US Adults in modern nationally representative aging surveys.

We use 24 predictors:
    1. Allostatic load measures: systolic blood pressure (continuous), body mass index (continuous), glycated hemoglobin A1C (continuous), C-reactive protein (continuous), and hormone dehydroepiandrosterone (continuous)
            These measures were selected based on the following papers: https://academic.oup.com/psychsocgerontology/article/76/3/574/5703612 and https://academic.oup.com/psychsocgerontology/article/64B/suppl_1/i67/552266
    2. Health behaviors: smoking (binary), alcohol consumption (binary), physical activity (binary), and sleep duration (ordinal) 
    3. Socio-demographic status: age (ordinal), education (ordinal), net household assets (ordinal), marital status (binary), sex (binary), and race (binary)
    4. Health status: self-rated physical (ordinal) and mental health (ordinal), lung COPD conditions (binary), and functional limitations (binary)
    5. Network measures: size of social network (continuous), number of unique social contacts (continuous), proportion of social contacts living together (continuous), average frequency of talking to social contacts (continuous), and average closeness to social contacts (continuous)
            These measures were selected based on the following paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6110003/

1. Body mass index: (NSHAP: BMI (continuous); WLS: z_ix011rec (continuous); HRS: HC139 - weight in pounds (continuous)
2. Hypertension: NSHAP: CONDITNS_6 (binary); WLS: z_ax341re (binary); HRS: HC005 (binary)
3. Diabetes: NSHAP: CONDITNS_7 (binary); WLS: z_ax342re (binary); HRS: HC010 (binary)
4. Self-rated health: NSHAP: PHYSHEALTH + MNTLHLTH (ordinal); WLS: z_ix001rer (ordinal); HRS: HC001 (ordinal)
5. Arthritis: NSHAP: FALLEN (binary); WLS: z_gx360re (binary); HRS: HC070 (binary)
6. Smoking: NSHAP: SMOKECIG (binary); WLS: z_ix013rec (binary); HRS: HC117 (binary)
7. Drunk alcohol: NSHAP: EVERDRNK (binary); WLS: z_gu025re (binary); HRS: HC128 (binary)
8. Age at baseline: NSHAP AGE (ordinal); WLS: z_brdxdy (ordinal, year of birth); HRS: HX067_R (ordinal, year of birth)
9. Education: NSHAP: EDUC (ordinal); WLS: z_gb103red (ordinal); HRS: HB014A (ordinal)
10. Net household assets: NSHAP: HSASSETS_RECODE (ordinal); WLS: z_gp260hec (continuous); HRS: HC134 + HQ331 + HQ376 (continuous)
11. Marital status: NSHAP: MARITAL (binary); WLS: z_gc042re (binary); HRS: HMARITAL (binary)
12. Sex: NSHAP: GENDER (binary); WLS: z_sexrsp (binary); HRS: HX060_R (binary)
13. Race: NSHAP: RACE_RECODE (binary); WLS: z_ie020re (binary); HRS: HB031A (binary)
14. Religion Importance:  BELIEFS (ordinal); WLS: z_il003rer (ordinal); HRS: HB053 (ordinal)
15. Children co-residence: NSHAP: LIVEC (binary); WLS: z_gd103kd (binary); HRS: HE012 (binary)
16. Grandchildren: NSHAP: PLAYCHLD (binary); WLS: z_id014cre (binary); HRS: HE046 (ordinal)
17. Relatives: NSHAP: CLSREL (binary); WLS: z_iz106rer (ordinal); HRS: HF174 (binary)
18. Volunteer: NSHAP: VOLUNTEER (binary); WLS: z_in504rer (ordinal); HRS: HG092 (binary)
19. Friends: NSHAP: SOCIAL (binary); WLS: z_iz023rer (ordinal); HRS: HF176 (ordinal)
20. Functional limitations: NSHAP: WALKBLK (binary); WLS: z_iv032re (binary); HRS: HG001 (binary)


Project 2 (Yiang):
The current academic literature in predicting mortality has extensively focused on disease and frailty, although social, behavioral, and psychological statuses may herald broad physiological decline. I tested the effectiveness of the machine learning algorithms on the NSHAP sample in Project 1 and learnt the important features predicting mortality. This project will extend the analysis by 1) using a different set of predictors 2) applying some new algorithms in addition to tree-based algorithm 3) using a different dataset.

Household Retire Survey (HRS) is a nationally representative survey of the older US adults (aged 50-61) collected every two years from 1992 to 2016. I use 2002 characteristics as baseline and 2016 disposition status as target. 

The link to the dataset: https://hrs.isr.umich.edu/data

Challenges:
1. The dataset is very large (over 20,000 observations) and the number of predictors is also large (over 1000).



Project 2 (Zejian):
Wisconsin Longitudinal Study (WLS) is a prospective cohort study of graduates of Wisconsin high schools. I use 2004 characteristics as baseline and 2020 disposition status as target.

The link to the dataset: https://researchers.wls.wisc.edu/data/survey-data/



Project 1:

National Social and Health Survey (NSHAP) Wave 1 (2010, as baseline) and Wave 3 (2020, disposition status as target), which is a nationally representative survey of the older US adults (aged 57-85).

The link to the dataset: https://www.icpsr.umich.edu/web/NACDA/series/706

Abstract:
The current academic literature in predicting mortality has extensively focused on disease and frailty, although social, behavioral, and psychological statuses may herald broad physiological decline. Using 24 social network and demographic factors, we developed a predictive model of 10-year mortality in a nationally representative sample of older adults in the US. We first used tree-based algorithms of Decision Tree (DT) and Random Forest (RF) that account for the interdependency of the social features and decide the splitting nodes and thresholds using entropy gain conditional on the previous splitting to discern disposition status. Additionally, we used the Support Vector Machine (SVM) that regards every sample as a node in high-level vector space and splits the nodes with an optimum plane by finding the best linear combination of features to get an optimum splitting accuracy. After the training and testing process, our algorithms reach accuracies of 74.7% for DT, 76.3% for RF, and 80.1% for SVM. We also discussed the social and demographic characteristics of the cases whose disposition statuses were either wrongly predicted as death or alive by our algorithms. The findings serve important purposes for public health practitioners in understanding the risk and protective factors of mortality in the aging process.


    
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