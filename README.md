# NYPD-Civilian-Complaints-Case-Study

## Part 1

In this case study, we will be studying 12,000 civilian complaints filed against New York City police officers, out main goal is to know whether there is a relationship between the gender of the police officer and their age to the allegation incident that they were involved in. The dataset contains 33358 observations including incident details, complainant details, and well as the police officer's information, and 30 variables including categorical and numerical values. We will be access the dataset using these values. We're going to be performing a hypothesis test for the research.

### Cleaning and EDA

To more accuratly perform statistical analysis on the dataset, we will need to perform data cleaning to reduce dimension. We will also utilize bar charts, scatter plots, etc. to perform EDA(Explainatory Data Analysis)

* **Cleaning the data** 

1. For conditions that were unable to record from the dataset, we decided to replace those values with `NaNs`. 
    * The following keyword in the dataset columns values will be converted: `'Unknown'`, `'Refused'`,`'non-conforming'`, `'Not described'`

1. We also observed categorical data such as `Transman (FTM)'`,`'Transwoman (MTF)'` in the `'complainant_gender'` column of the dataset that could be categorized into `'Male'` and `'Female'`, accordingly. 

1. Transform `'mos_gender'` column values into `'Male'` and `'Female'`

1. Combine the `'month_received'` and `'year_received'` column into `'complaint_receive_date'` and convert it to a **datatime** object.

1. Combine the `'month_closed'` and `'year_closed'` column into `'complaint_closed_date'` and convert it to a **datatime** object.

1. Combine the `'first_name'` and `'last_name'` column into `'mos_name'` and convert it to a **string** object.


* **Univariate Analysis:**
We plotted a histrgram distribution on `'year_received'` and `'year_closed'` column, and we discovered that the allegation incidents are increasing in a exponential growth on averate, we do see a pike in 2006 and 2013, it might implies to certain movements occuring in the society at the moment. Also, the two distribution are sharing a similar shape, which we can also say that the government is working on the incidents at a good pace so that the cases are being handled in time. 


* **Bivariate Analysis:**
We conducted a multiple box plot on `'mos_age_incident'` and `'mos_ethnicity'` to observe the range and mean of the officer that was being reported, as well as `'complainant_age_incident'` and `'complainant_ethnicity'`. From officer box plot, we observed that most of the officers that was being acused of allegation are around 33 years old and the distribution of officer's ethinicity are about the same except for American Indian, American Indian has the least amount of cases across all the other races. On the other hand, the complainant ages range and mean shares across all the ethnicity and in around 30-35 years old, which are also at a similar age like as the officers that we being reported. 


* **Interesting Aggregates:** 
In this case, we aggreated the `'contact_reason'` column and `'complainant_ethnicity'` with `'allegation'`, to get the most-frequent happened allegation. By knowing the different behavior, we can observed the allegations were being charged to the officer based on complainant's ethnicity along with the contact reason. 


### Assessment of Missingness
We observed the majority of the missing value of the dataset comes from the complainant information such as age and gender. We decided to assess the missingness by using complainant ethnicity and police officer's gender for comparison. Empirical distribution and permutation were being used in the assessment. Both complainant ethnicity and gender shared a similar distribution of whether or not having the null values. Tvd was also being used as test statistics and by looking at the distribution, we confirm that the two variable `'mos_gender'` and `'complainant_gender'` are dependent to each other and the missingness in this case is MAR.

On the other hand, we applied the same technique to the columns `'complainant_ethnicity'` and `'month_received'`, and we obtained a test statistic lies within the empirical distribution.  We can conclude that it is possible to get such value under the null hypothesis. Thus, `'complainant_ethnicity'` is MCAR and dependent from `'month_received'`.

In addition, the missing data (column) could be ignoarable as the data is MCAR, we will remove all the null values in the dataset for a more accurate assessment of the case study.

### Hypothesis Test
In the hypothesis test, we will be looking at the `'mos_age_incident'` and the `'mos_gender'`column to trying to determine whether there is a relationship between the gender of the police officer and their age to the allegation incident that they were involved in. More specifically, we wanted to know if male tends to have a higher age in allegations. Therefore, we will be conducting the test based on the following hypothesis: 
- **Null hypothesis**: In the population, age of male and female has the same distribution.

- **Alternative hypothesis**: In the population, male tends to have a higher age in allegations.

**Process**

We shuffled the age column `'mos_age_incident'` and assessed the difference in `'mos_gender'` and appended the test statistic in the a result list. We repeated the test for 1000 time to get the simulated distribution under the null hypothesis.

**Result**
- After conducting the results, we observed the observed value lies outside of the emprical distribution under the null hypothesis
- Therefore, we reject the null hypothesis: the two groups do not come from the same distribution. That is, officer's gender and age do not come from the same distribution, the result seems to favor the alternative hypothesis, but we cannot conclude that male has a higher age at the incident time frame.
- To improve the test result, we can introduce machine learning techniques such as logistic regression to find a better predictor to further investigate the relationship between the police officer gender and age. 

## Part 2

In this finding, we will be continue to study on the NYPD data set. More specifically, we will be using machine learning modeling to make predictions on NYPD officer's ranking during incident based on various predictors. 

### Baseline Model
To start off the research, we will be performing data cleaning and selection on needed data. Cleaning method will inherit from the previoud project by filling in missing values and filtering out ``"NaN"`` values. After cleaning the data, we will be conducting a base logistic model using the ``Sklean`` package, and ``Pipeline`` that helps fitting and transforming the data into our logistic regression model. 

- Predictor ``"mos_ethnicity"`` is used as the baseline model to predict the officer's rankin. It seems intuitively that there is a relationship between officer's ethnicity and its title. Some ethnicity seems to have a higher general ranking than the others, and that is also the reason why we choose to start of the prediction using the ``"mos_ethnicity"`` variable. 

- We know that ``"mos_ethnicity"`` is an ordinal variable by discovering that it consist various ethnicities such as ``Hispanic``, ``White``, ``Black``, ``Asian``. To catergorize this predictor, we decide to use ``OrdinalEncoder`` to encode the difference in ethnicity, and by fitting it into a pipeline and logistic regression model, we obtained a ``68%`` of accuracy of our model and a ``0.6834`` R-squared value for this baseline model.

Note: R-squared is a goodness-of-fit measure for linear regression models. It is valued between 0 to 1, the closer the number is getting to 1, means that the better the model is predicted. 

### Final Model
Although our baseline model has a pretty good prediction on the officer's ranking. We would like to further investigate and want to improve the performance of the prediction. We designed to include feature engineering and predictor searching into different modeling and found the best model for our final model by using the ``"mos_ethnicity","mos_gender", "mos_age_incident","rank_now"`` variables. In addition to ``"mos_ethnicity"``, the final(best) model has three additional features that strongly helped to predict the officer's title. In the process of searching a related predictor manually by adding new features one by one, and later resulted our final model. 

- To fit the predictors into the pipeline, we first transform the column ``"mos_age_incident"`` into standardscaler, then apply one-hot-encoder to ``"mos_gender"`` and ordinal encoder to the columns ``"mos_ethnicity"`` and ``"rank_now"``. 

- ``"mos_gender"`` are categorized by ``"M"`` and ``"F"``. One-hot encoder will be the most appropreate to the transformation.

- ``"rank_now"`` consist different ranking titles, and therfore it is being categorize as ``"mos_ethnicity"`` in above for the same reason. 

After fitting the predictors into our final model, we obtained a ``71%`` on the model accuracy. By all that means is that we are ``71%`` confident to correctly predict the officer's ranking at the indident given the ``"mos_ethnicity","mos_gender", "mos_age_incident","rank_now"`` predictors. Also, this model gives a ``0.7076`` R-squared value. It is so far the best predicted model that we obtained. 


### Fairness Evaluation
Lastly, we will be assessing the model through a fairness evaluation, we will be splitting our data and uses permutation to conduct this study. We set a test size to ``0.3`` and a ``42`` random state in our splitting so that our data can be shuffled and draw more randomly for the assessment. The observing predictors will remains the same as our final model. 

- After splitting, we obtained ``X_train, X_test, y_train, y_test`` and ready to fit the data into our modeling.

- We fit the ``X_train`` data into the final model pipeline and obtain a predicted train value, same for the ``X_test`` data. 

- After fitting the two modeling, we can see from the classification report that the two model are having the same accuracy of ``71%``. However, the f1-score on the ``X_train`` set peforms slightly better. Note: The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. (Cited from Wikipedia)

- Since the two models are obtaining a similar accuracy score and f1-score, we can say that we have a decent low false positives and low false negatives, and a true postive and true negative prediction.  Therefore, we can say that this model is pretty fair.

