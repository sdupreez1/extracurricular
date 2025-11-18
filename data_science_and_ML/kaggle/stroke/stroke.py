import pandas as pd
from statistics import mean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn import svm
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Can see all columns when printed in console
pd.options.display.max_columns = None

# The goal of this project is to compare how well different methods predict 
# stroke risk based on the provided data (taken from Kaggle Stroke Prediction Dataset)


                    # Loading, Cleaning & Splitting data #
                    
stroke_with_ids = pd.read_csv("stroke.csv")   # Import csv with stroke data using pandas
stroke = stroke_with_ids.drop(columns=["id"]) # Remove ids from data for ease of use

stroke.isna().sum() # There are 201 entries with NaN bmi values
nan_bmi_indexes = stroke[stroke["bmi"].isna()].index # Note the indexes of NaN bmi entries
stroke.drop(nan_bmi_indexes, inplace = True) # Removes entries with NaN bmi

# Remove the single entry with "Other" gender
stroke.drop(stroke[stroke["gender"]=="Other"].index, inplace = True) 


            # Turning catagorical data into numerical/dummy data #
        
need_to_convert = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"] 
       
# Here we drop one column for each dummy conversion to avoid multicollinearity
is_male = pd.get_dummies(stroke.gender, prefix="gender", dtype=int).drop(columns="gender_Female")
is_married = pd.get_dummies(stroke.ever_married, prefix="married", dtype=int).drop(columns="married_No")
work_types = pd.get_dummies(stroke.work_type, prefix="work", dtype=int).drop(columns="work_Never_worked")
urban_res = pd.get_dummies(stroke.Residence_type, prefix="residence", dtype=int).drop(columns="residence_Rural")
smoker = pd.get_dummies(stroke.smoking_status, dtype=int).drop(columns=["never smoked", "Unknown"])
    # ^Remove "Unknown" smoking status

# Add all these dummy variables back into the stroke DataFrame
stroke.drop(columns=need_to_convert, inplace=True) # Drop all the unconverted columns

# Add converetd cols back
stroke_dummy = pd.concat([stroke, is_male, is_married, work_types, urban_res, smoker], axis=1) 

# Separate data into predictor and response variables
stroke_rates = stroke_dummy.stroke # Use this as our output variable
stroke_predictors = stroke_dummy.drop(columns="stroke") # Use these as our predictors

# Split data into testing and training
stroke_preds_train, stroke_preds_test, stroke_rates_train, stroke_rates_test = train_test_split(stroke_predictors, stroke_rates)

# Oversampling training data
oversampled_stroke_preds_train, oversampled_stroke_rates_train = SMOTE().fit_resample(stroke_preds_train, stroke_rates_train)

# Undersampling training data
usam_preds_train, usam_response_train = RandomUnderSampler().fit_resample(stroke_preds_train, stroke_rates_train)

     # Logistic Regression - Comparing Results from Over/Undersampling #
                            
# Initialise models
osam_logit = LogisticRegression() # Logit to be used on oversampled data
usam_logit = LogisticRegression() # Logit to be used on undersampled data

# Cross-Validation
score_fcts = ["precision", "recall", "f1", "roc_auc"]

osam_logit_cv = cross_validate(
    osam_logit, oversampled_stroke_preds_train, oversampled_stroke_rates_train,
    scoring = score_fcts, return_estimator = True, 
    return_train_score = True
    )

usam_logit_cv = cross_validate(
    usam_logit, usam_preds_train, usam_response_train,
    scoring = score_fcts, return_estimator = True, 
    return_train_score = True
    )

print("\n")
for fct in score_fcts:
    osam_score_mean = round(mean(osam_logit_cv[f'test_{fct}']),2)
    usam_score_mean = round(mean(usam_logit_cv[f'test_{fct}']),2)
    print(f"Oversampled model mean CV {fct} score: ", osam_score_mean)
    print(f"Undersampled model mean CV {fct} score: ", usam_score_mean)
    print("\n")
        # Oversampled model performs better on all metrics in CV

# Fit models to oversampled and undersampled data to compare
osam_logit.fit(oversampled_stroke_preds_train, oversampled_stroke_rates_train)
usam_logit.fit(usam_preds_train, usam_response_train)

# Get predictions from both models
osam_logit_stroke_predictions = osam_logit.predict(stroke_preds_test)
usam_logit_stroke_predictions = usam_logit.predict(stroke_preds_test)

# Confusion Matricies
usam_logit_cm = confusion_matrix(stroke_rates_test, usam_logit_stroke_predictions)
osam_logit_cm = confusion_matrix(stroke_rates_test, osam_logit_stroke_predictions)

# Plot confusion matricies
ConfusionMatrixDisplay(osam_logit_cm).plot()
plt.title("Logistic Regression (Trained with Oversampling)")
plt.show()

ConfusionMatrixDisplay(usam_logit_cm).plot()
plt.title("Logistic Regression (Trained with Undersampling)")
plt.show()
    # Oversampled model has worse recall but better precision

# Classification report for logits
print("Logistic Regression (Oversampled): \n" + classification_report(stroke_rates_test, osam_logit_stroke_predictions))
print("Logistic Regression (Undersampled): \n" + classification_report(stroke_rates_test, usam_logit_stroke_predictions))
    # Precision: proportion of predicted +ve values that were correct 
        # Low precision means it is predicting actually -ve values as +ve

    # Recall: proportion of actually +ve data that was correctly predicted
        # Low recall means lots of actually +ve data was predicted to be -ve

    # F1-score: how accurate is the model overall (harmonic mean of precision and recall)
       
        # F1 is good for imbalanced data since it assesses model only based on 
        # rates of true positives, rather than getting skewed by the large 
        # number of true negative results

    # On real test data, undersampled has much better recall, 
    # lower precision, similar f1-score, and lower accuracy
                                    
# Get model probability predictions for ROC curves
osam_logit_scores = osam_logit.predict_proba(stroke_preds_test)[:,1]
usam_logit_scores = usam_logit.predict_proba(stroke_preds_test)[:,1]
    # ^Returns two values for each entry, one for Prob(0) one for Prob(1), we take only Prob(1)
 
# Get fpr and tpr for models to plot in ROC curve
osam_logit_fpr, osam_logit_tpr, osam_logit_threshold = roc_curve(stroke_rates_test, osam_logit_scores)
usam_logit_fpr, usam_logit_tpr, usam_logit_threshold = roc_curve(stroke_rates_test, usam_logit_scores)

# Plot ROC curves:
    # Oversampled
plt.figure(figsize=(5,5))
plt.title("ROC - Logistic Regression (Oversampled)")
plt.plot(osam_logit_fpr, osam_logit_tpr) # Plot FPR against TPR

plt.plot([0,1], ls="--") # Draw comparison ROC for random guessing model (y=x)
plt.plot([0,0], [0,1], c=".7"), plt.plot([1,1], c=".7") # Draw lines x=0 and y=1 (boundaries)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.show()

    # Undersampled
plt.figure(figsize=(5,5))
plt.title("ROC - Logistic Regression (Undersampled)")
plt.plot(usam_logit_fpr, usam_logit_tpr)

plt.plot([0,1], ls="--") # Draw comparison ROC for random guessing model (y=x)
plt.plot([0,0], [0,1], c=".7"), plt.plot([1,1], c=".7") # Draw lines x=0 and y=1 (boundaries)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.show()

osam_logit_auc = roc_auc_score(stroke_rates_test, osam_logit_scores) # Get AUC value
usam_logit_auc = roc_auc_score(stroke_rates_test, usam_logit_scores) # Get AUC value

# Comparing ROC for logits fitted on over and undersampled data
plt.figure(figsize=(5,5))
plt.title("ROC Comparison")
plt.plot(osam_logit_fpr, osam_logit_tpr, label = f"Oversampled (AUC = {osam_logit_auc:.2f})")
plt.plot(usam_logit_fpr, usam_logit_tpr, label = f"Undersampled (AUC = {usam_logit_auc:.2f})")

plt.plot([0,1], ls="--")
plt.plot([0,0], [0,1], c=".7"), plt.plot([1,1], c=".7") 

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.show()
        
# Undersampled has better recall and AUC, but much worse precision


                            # Random Forest #
                            
# Initialisation
rforest = RandomForestClassifier()

# rforest_cv = cross_validate(
#     rforest, stroke_preds_train, stroke_rates_train,
#     return_estimator = True, return_train_score = True,
#     scoring = "balanced_accuracy"
#     )

# test_rforest_cv = cross_validate(
#     RandomForestClassifier(), oversampled_stroke_preds_train, 
#     oversampled_stroke_rates_train, return_estimator = True, 
#     return_train_score = True, scoring = "balanced_accuracy"
#     )

rforest.fit(oversampled_stroke_preds_train, oversampled_stroke_rates_train)

rforest_importances = {"predictors":[col for col in stroke_predictors.columns],
                      "weightings":list(rforest.feature_importances_)}
rforest_importances = pd.DataFrame(rforest_importances).sort_values(by="weightings")

plt.title("Feature Importance - Random Forest")
plt.barh(rforest_importances.predictors, rforest_importances.weightings)
plt.show()

# rforest_fpr, rforest_tpr, rforest_threshold = roc_curve(stroke_rates_test, rforest_scores)

# Random Forest
rforest_predicts = rforest.predict(stroke_preds_test)
rforest_cm = confusion_matrix(stroke_rates_test, rforest_predicts)

# Visualise the tree
# plot_tree(rforest.estimators_[0], feature_names=stroke_predictors.columns)
# plt.show()

# Random forest confusion matricies and classification reports
ConfusionMatrixDisplay(rforest_cm).plot()
plt.title("Random Forest")

print("Random Forest: \n" + classification_report(stroke_rates_test, rforest_predicts))

# Random Forest
# rforest_scores = rforest.predict_proba(stroke_preds_test)[:,1]

# # ROC for random forest
# plt.figure(figsize=(5,5))
# plt.title("ROC - Random Forest")
# plt.plot(rforest_fpr, rforest_tpr)

# plt.plot([0,1], ls="--") 
# plt.plot([0,0], [0,1], c=".7"), plt.plot([1,1], c=".7") 

# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")

# plt.show()

# rforest_auc = roc_auc_score(stroke_rates_test, rforest_scores)


                        # Support Vector Classifier #
# svc_linear = make_pipeline(
#     SMOTE(),
#     svm.SVC(kernel="linear", C=1.0)
#     )

# svc_linear.fit(stroke_preds_train, stroke_rates_train)

# svc_linear.score(stroke_preds_test, stroke_rates_test) # Mean accuracy = 0.85



