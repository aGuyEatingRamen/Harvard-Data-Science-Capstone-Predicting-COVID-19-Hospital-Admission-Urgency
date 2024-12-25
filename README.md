# Harvard-Data-Science-Capstone-Predicting-COVID-19-Hospital-Admission-Urgency

This project aims to predict the urgency of hospital admission for COVID-19 patients based on symptoms and demographic data. Using a dataset collected during a peak COVID-19 wave, the project involves data cleaning, exploratory data analysis (EDA), and model building. Primary predictors include age, sex, and symptoms such as cough, fever, and fatigue, while the response variable indicates whether admission was urgent (within one day of symptom onset).

Note: The code was copied off the Jupiter Notebooks that were completed through the course, but access to the orginal dataset was not found through the course, thus I am unable to access the datset. However I was able gain some insight within the dataset:

       age  sex  cough  fever  chills  sore_throat  headache  fatigue  Urgency
0     30.0  1.0    0.0    0.0     0.0          0.0       0.0      0.0        0
1     47.0  1.0    0.0    0.0     0.0          0.0       0.0      0.0        0
2     49.0  1.0    0.0    0.0     0.0          0.0       0.0      0.0        0
3     50.0  0.0    0.0    0.0     0.0          0.0       0.0      0.0        0
4     59.0  0.0    0.0    1.0     0.0          0.0       0.0      0.0        0
...    ...  ...    ...    ...     ...          ...       ...      ...      ...
996   72.0  1.0    0.0    NaN     0.0          0.0       NaN      0.0        1
997   56.0  1.0    0.0    0.0     0.0          0.0       0.0      0.0        1
998   43.0  1.0    0.0    1.0     0.0          0.0       0.0      0.0        1
999    NaN  1.0    0.0    1.0     0.0          0.0       0.0      0.0        1
1000  50.0  0.0    0.0    0.0     0.0          0.0       0.0      0.0        1
