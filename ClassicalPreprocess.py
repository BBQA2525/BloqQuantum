import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def ohe(df, categorical_cols):
	# create an instance of OneHotEncoder
	encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)# sparse=False) < sklearn 1.2; > sklearn 1.2 sparse_out=False) ; v1.6.0 = sparse_output=False
	
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