#pip3 install ucimlrepo
#pip3 install pandas
#pip3 install scikit-learn

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
  
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
  
# metadata 
print("Metadata:  {}".format(statlog_german_credit_data.metadata))
print("\n")
  
# variable information 
print("Variables:  {}".format(statlog_german_credit_data.variables))
print("\n")

print("Features X: {}".format(X))
print("\n")
print("Targets y:  {}".format(y))
print("\n")

good_count = y['class'].value_counts()[1] 
bad_count = y['class'].value_counts()[2] 

print("Good counts:  {}".format(good_count))
print("\n")
print("Bad counts:  {}".format(bad_count))
print("\n")

X_unique_cts = X.nunique()
print("X Unique Column Counts:  {}".format(X_unique_cts))
print("\n")

column_list = ['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10', 'Attribute11', 'Attribute12', 'Attribute13', 'Attribute14', 'Attribute15', 'Attribute16', 'Attribute17', 'Attribute18', 'Attribute19', 'Attribute20']

def ohe(df, categorical_cols):
	# create an instance of OneHotEncoder
	encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # < sklearn 1.2; > sklearn 1.2 sparse_out=False)
	
	# fit and transform the specified columns
	encoded_data = encoder.fit_transform(df[categorical_cols])
	
	# create a df from the encoded data
	encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
	
	# join the encoded df with the original df
	df = df.drop(categorical_cols, axis=1).join(encoded_df)
	
	# replace all nan values with 0
	df.fillna(0, inplace=True)
	
	return df

def scaler_transform(data, scaler=StandardScaler()):
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame(data_scaled)
    data_scaled.columns = data.columns
    data_scaled.index = data.index
    return data_scaled
	
# create a new df from the transformed data
X_transformed_df = ohe(X, column_list)
print("x_transformed df:  {}".format(X_transformed_df))
print("\n")

# combine pandas df's for features and target
# reason: 2.33 times as many good as bad datum, so use undersampling 
# to select only as much sample from majority class as exists in 
# minority class
feature_target_df = X_transformed_df.merge(y, left_index=True, right_index=True)
print("feature_target_df: {}".format(feature_target_df))
print("\n")

data_bad = feature_target_df[feature_target_df["class"] == 2]
data_good = feature_target_df[feature_target_df["class"] == 1]
data_good = data_good.sample(n=len(data_bad), random_state=123)
data = pd.concat([data_bad, data_good])
print(data_bad)
print("\n")

print(data_good)
print("\n")

print(data)
print("\n")

X2 = data.iloc[:,:-1]
print(X2)
print("\n")
X2_unique_cts = X2.nunique()
print("X2 vals, binary check (unique col val cts):  {}".format(X2_unique_cts))
print("\n")
y = data.iloc[:,-1]
print(y)
print("\n")

y = y.where(y == 1, 0)
print(y)
print("\n")
print("y-vals, binary check (unique col val cts):  {}".format(y.nunique()))
print("\n")

X_train, X_test, y_train, y_test = train_test_split(X2,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=123)
    
X_train_scaled = scaler_transform(data=X_train)

logreg = LogisticRegression(random_state=123)
parameters = {'solver': ['liblinear', 'saga'],
              'C': np.logspace(0, 10, 10)}
logreg_cv = GridSearchCV(estimator=logreg,
                         param_grid=parameters,
                         cv=5)
logreg_cv.fit(X=X_train_scaled, y=y_train)

logreg = LogisticRegression(C=logreg_cv.best_params_['C'],
                            solver=logreg_cv.best_params_['solver'],
                            random_state=123)

logreg.fit(X_train_scaled, y_train)
y_pred_train = logreg.predict(X_train_scaled)

print("Results of Training:")
print("Confusion Matrix:")
print("\n")
print(confusion_matrix(y_true=y_train, y_pred=y_pred_train))
print(classification_report(y_true=y_train,
                            y_pred=y_pred_train,
                            target_names=["good", "bad"]))
print("\n")

X_test_scaled = scaler_transform(data=X_test)

logreg.fit(X_test_scaled, y_test)
y_pred_test = logreg.predict(X_test_scaled)

print("Results of Testing:")
print("Confusion Matrix:")
print("\n")
print(confusion_matrix(y_true=y_test, y_pred=y_pred_test))
print(classification_report(y_true=y_test,
                            y_pred=y_pred_test,
                            target_names=["good", "bad"]))
print("\n")