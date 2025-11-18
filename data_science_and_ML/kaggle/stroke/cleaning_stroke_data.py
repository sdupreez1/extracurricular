import pandas as pd

pd.options.display.max_columns = None

                    # Loading, Cleaning & Splitting data #
                    
stroke_with_ids = pd.read_csv("stroke.csv")   # Import csv with stroke data using pandas
stroke = stroke_with_ids.drop(columns=["id"]) # Remove ids from data for ease of use

stroke.isna().sum() # There are 201 entries with NaN bmi values
nan_bmi_indexes = stroke[stroke["bmi"].isna()].index # Note the indexes of NaN bmi entries
stroke.drop(nan_bmi_indexes, inplace = True) # Removes entries with NaN bmi

# Remove the single entry with "Other" gender
stroke.drop(stroke[stroke["gender"]=="Other"].index, inplace = True) 

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

# Write clean DataFrame to a csv 
# stroke_dummy.to_csv("stroke_clean.csv", index = False) 
    # Including indexes in the output csv could induce a stroke in the user
