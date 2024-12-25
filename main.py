#All libaries from all notebooks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
%matplotlib inline
# Read the datafile "covid.csv"
df = pd.read_csv("covid.csv")

# Take a quick look at the dataframe
print(df.head())



# Check if there are any missing or Null values
print("Missing values per column:")
print(df.isnull().sum())

# Find the number of rows with missing values
num_null = df.isnull().any(axis=1).sum()
print("Number of rows with null values:", num_null)
original_index = df.index

X = df.drop(columns=['Urgency'])  
y = df['Urgency']  


# kNN impute the missing data
# Use a k value of 5

imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_imputed.index = original_index

# Replace the original dataframe with the imputed data, continue to use df for the dataframe
# Your code here
df = X_imputed.copy()
df['Urgency'] = y

#Graphs for visualization of the data
age_groups = pd.cut(df['age'], bins=[0, 10,20,30, 40,50, 60,70, 80,90, 100], labels=["0-10", "11-20","21-30", "31-40", "41-50","51-60", "61-70","71-80", "81-90", "91-100"])
age_urgency = df.loc[df['Urgency'] == 1].groupby(age_groups).size()
age_urgency.plot(kind='bar', color='blue', alpha=0.7)
plt.xlabel("Age Group")
plt.ylabel("Count of Urgent Cases")
plt.title("Urgent Hospitalization by Age Group")
plt.show()

symptoms = ['cough', 'fever', 'chills', 'sore_throat', 'headache', 'fatigue']
urgent_symptoms = df.loc[df['Urgency'] == 1, symptoms].sum()
urgent_symptoms.plot(kind='bar', color='green', alpha=0.7)
plt.xlabel("Symptoms")
plt.ylabel("Count")
plt.title("Most Common Symptoms for Urgent Cases")
plt.show()

cough_urgency = df.groupby('Urgency')['cough'].mean()
cough_urgency.plot(kind='bar', color=['red', 'orange'], alpha=0.7)
plt.xlabel("Urgency (0 = No, 1 = Yes)")
plt.ylabel("Proportion with Cough")
plt.title("Cough Prevalence by Urgency")
plt.show()

# Split the data into train and test sets with 70% for training
df_train, df_test = train_test_split(df, test_size=0.3, random_state=60)

# Save the train data into a csv called "covid_train.csv"
df_train.to_csv("covid_train.csv", index=False)
# Save the test data into a csv called "covid_test.csv"
df_test.to_csv("covid_test.csv", index=False)
print("split succesfull")

#Quick peeking of the data collected and organized
# Read the datafile "covid_train.csv"
df_train = pd.read_csv("covid_train.csv")
# Take a quick look at the dataframe
print(df_train.head())

# Read the datafile "covid_test.csv"
df_test = pd.read_csv("covid_test.csv")
# Take a quick look at the dataframe
print(df_test.head())

# Get the train predictors
X_train = df_train.drop(columns=['Urgency'])
# Get the train response variable
y_train = df_train['Urgency']
# Get the test predictors
X_test = df_test.drop(columns=['Urgency'])
# Get the test response variable
y_test = df_test['Urgency']

# Define your classification model
model = KNeighborsClassifier(n_neighbors=5)
# Fit the model on the train data
model.fit(X_train, y_train)

# Predict and compute the accuracy on the test data
y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy is {model_accuracy}")

# Define a kNN classification model with k = 7
knn_model = KNeighborsClassifier(n_neighbors=7)
# Fit the kNN model on the train data
knn_model.fit(X_train, y_train)
# Predict with kNN model
y_pred_knn = knn_model.predict(X_test)
# Define a Logistic Regression model with max_iter as 10000 and C as 0.1
log_model = LogisticRegression(max_iter=10000, C=0.1)
# Fit the Logistic Regression model on the train data
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

# Compute confusion matrices for both models
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_log = confusion_matrix(y_test, y_pred_log)

def compute_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    specificity = tn / (tn + fp) if tn + fp != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    return accuracy, precision, recall, specificity, f1
accuracy_knn, precision_knn, recall_knn, specificity_knn, f1_knn = compute_metrics(cm_knn)

# Calculate the metrics for Logistic Regression
accuracy_log, precision_log, recall_log, specificity_log, f1_log = compute_metrics(cm_log)

metric_scores = {
    "Accuracy": [accuracy_knn, accuracy_log],
    "Recall": [recall_knn, recall_log],
    "Specificity": [specificity_knn, specificity_log],
    "Precision": [precision_knn, precision_log],
    "F1-score": [f1_knn, f1_log]
}

# Display your results
print(metric_scores)

# Define a kNN classification model with k = 7
knn = KNeighborsClassifier(n_neighbors=7)
# Fit the model on the train data
knn.fit(X_train, y_train)
# Predict probabilities for the positive class on the test data using the kNN model
y_pred_knn = knn.predict_proba(X_test)[:, 1]  # The probability for the positive class (class 1)

# Define a Logistic Regression model with max_iter as 10000, C as 0.1, and a random_state of 42
logreg = LogisticRegression(max_iter=10000, C=0.1, random_state=42)
# Fit the Logistic Regression model on the train data
logreg.fit(X_train, y_train)
# Predict probabilities for the positive class on the test data using the logistic regression model
y_pred_logreg = logreg.predict_proba(X_test)[:, 1] 


def get_thresholds(y_pred_proba):
    # We only need to consider unique predicted probabilities
    unique_probas = np.unique(y_pred_proba)
    
    # Sort unique probabilities in descending order
    unique_probas_sorted = np.sort(unique_probas)[::-1]
   
    # We'll also add some additional thresholds to our set
    # This ensures our ROC curves reach the corners of the plot, (0,0) and (1,1)
    
    # Insert 1.1 at the beginning of the threshold array
    # 1.1 may seem like an odd threshold, but a value greater than 1
    # is required if we want the ROC curve to reach the lower left corner
    # (0 fpr, 0 tpr) considering one of our models produces probability predictions of 1
    thresholds = np.insert(unique_probas_sorted, 0, 1.1)
    # Append 0 to the end of the thresholds
    thresholds = np.append(thresholds, 0)
    return thresholds

# Get the thresholds for the kNN model using the predicted probabilities
knn_thresholds = get_thresholds(y_pred_knn)
# Get the thresholds for the Logistic Regression model using the predicted probabilities
logreg_thresholds = get_thresholds(y_pred_logreg)

def get_fpr(y_true, y_pred_proba, threshold):
    # Convert probabilities to binary predictions based on the threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate False Positive Rate (FPR)
    fpr = fp / (fp + tn)
    return fpr

def get_tpr(y_true, y_pred_proba, threshold):
    # Convert probabilities to binary predictions based on the threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate True Positive Rate (TPR)
    tpr = tp / (tp + fn)
    return tpr


# Calculate FPR and TPR for kNN at each threshold
knn_fpr = [get_fpr(y_test, y_pred_knn, threshold) for threshold in knn_thresholds]
knn_tpr = [get_tpr(y_test, y_pred_knn, threshold) for threshold in knn_thresholds]

# Calculate FPR and TPR for Logistic Regression at each threshold
logreg_fpr = [get_fpr(y_test, y_pred_logreg, threshold) for threshold in logreg_thresholds]
logreg_tpr = [get_tpr(y_test, y_pred_logreg, threshold) for threshold in logreg_thresholds]

# Compute the ROC AUC score for the kNN model
knn_auc = roc_auc_score(y_test, y_pred_knn)

# Compute the ROC AUC score for the Logistic Regression model
logreg_auc = roc_auc_score(y_test, y_pred_logreg)

### edTest(test_plot) ###
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize = (14,8))

# Plot KNN Regression ROC Curve
ax.plot(knn_fpr, knn_tpr, 
        label=f'KNN (area = {knn_auc:.2f})',
        color='g', lw=3)

# Plot Logistic Regression ROC Curve
ax.plot(logreg_fpr, logreg_tpr, 
        label=f'Logistic Regression (area = {logreg_auc:.2f})',
        color = 'purple', lw=3)

# Threshold annotations
label_kwargs = {}
label_kwargs['bbox'] = dict(
    boxstyle='round, pad=0.3', color='lightgray', alpha=0.6
)
eps = 0.02 # offset
for i in range(0, len(logreg_fpr), 15):
    threshold = str(np.round(logreg_thresholds[i], 2))
    ax.annotate(threshold, (logreg_fpr[i], logreg_tpr[i] - eps), fontsize=12, color='purple', **label_kwargs)

for i in range(0, len(knn_fpr)-1, 15):
    threshold = str(np.round(knn_thresholds[i], 2))
    ax.annotate(threshold, (knn_fpr[i], knn_tpr[i] + eps), fontsize=12, color='green', **label_kwargs)

# Plot diagonal line representing a random classifier
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

# Scenario 1 - Brazil
ax.fill_between([0, 0.5], [0.5, 0], color = 'red', alpha = 0.4, label='Scenario 1 - Brazil')

# Scenario 2 - Germany
ax.axhspan(0.8, 0.9, facecolor='y', alpha=0.4, label = 'Scenario 2 - Germany')

# Scenario 3 - India
ax.fill_between([0, 1], [1, 0], [0.5, -0.5], alpha = 0.4, color = 'blue', label = 'Scenario 3 - India')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=20)
ax.set_ylabel('True Positive Rate', fontsize=20)
ax.set_title('Receiver Operating Characteristic', fontsize=20)
ax.legend(loc="lower right", fontsize=15)
plt.show()
