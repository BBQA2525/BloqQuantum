import numpy as np
from sklearn.linear_model import LinearRegression
import re
import os
from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import networkx as nx
import matplotlib.pyplot as plt

import dwave_networkx as dnx
import networkx as nx
from dwave.cloud import Client
from dwave.system import DWaveSampler, EmbeddingComposite
from minorminer import find_embedding
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score, recall_score

#import itertools
import random

import networkx as nx

import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from itertools import combinations

from QuantumPreprocess import scale_subnode_groups, get_subnodes
from MinorEmbedding import generate_combinations, make_combinations

from sklearn.linear_model import RidgeCV
from numpy import arange

def permutate_wo_itertools_permutation(i,lst):
    i1, i2 = divmod(i, len(lst) - 1)
    if i1 <= i2:
        i2 += 1 
    return ("'" + str(lst[i1]) + "', '" + str(lst[i2]) + "'")

def generate_linear_and_quadratic_terms(num_of_qubits):
	m = 0
	linear_list = []
	quadratic_list = []
	
	permutation_pairs_list = [permutate_wo_itertools_permutation(i,list(range(n))) for i in range(len_lst_to_permutate*(len_lst_to_permutate-1))]
	
	for m in range(n):
		linear_list.append(f"('{m}', '{m}')")
		m += 1
	
	linear_dict = {eval(item): -1 for item in linear_list}
	quadratic_dict = {eval(item): 2 for item in permutation_pairs_list}
	
	return linear_dict, quadratic_dict

n = 20
len_lst_to_permutate = len(list(range(n)))
linear, quadratic = generate_linear_and_quadratic_terms(n)

#linear = {('0', '0'): -1, ('1', '1'): -1, ('2', '2'): -1, ('3', '3'): -1, ('4', '4'): -1, ('5', '5'): -1, ('6', '6'): -1, ('7', '7'): -1, ('8', '8'): -1, ('9', '9'): -1, ('10', '10'): -1, ('11', '11'): -1, ('12', '12'): -1, ('13', '13'): -1, ('14', '14'): -1, ('15', '15'): -1, ('16', '16'): -1, ('17', '17'): -1, ('18', '18'): -1, ('19', '19'): -1}
#quadratic = {('0', '1'): 2, ('1', '2'): 2, ('2', '3'): 2, ('3', '4'): 2, ('4', '5'): 2, ('5', '6'): 2, ('6', '7'): 2, ('7', '8'): 2, ('8', '9'): 2, ('9', '10'): 2, ('10', '11'): 2, ('11', '12'): 2, ('12', '13'): 2, ('13', '14'): 2, ('14', '15'): 2, ('15', '16'): 2, ('16', '17'): 2, ('17', '18'): 2, ('18', '19'): 2, ('19', '0'): 2}
	
print("linear:  {}".format(linear))
print("\n")
print("length of linear:  {}".format(len(linear)))
print("\n")
print("quadratic:  {}".format(quadratic))
print("\n")
print("length of quadratic:  {}".format(len(quadratic)))
print("\n")
print("\n")

#pip3 install ucimlrepo
#pip3 install pandas
#pip3 install scikit-learn

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
  
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
  
# metadata 
data_metadata = statlog_german_credit_data.metadata
print("Metadata:  {}".format(data_metadata))
print("\n")
  
# variable information 
data_variables = statlog_german_credit_data.variables
print("Variables:  {}".format(data_variables))
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

X_1 = data.copy()

X2 = X_1.iloc[:,:-1]
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

##########################################################################################
## 	Hybrid-Quantum Workflow Begins Here...												##
##########################################################################################

def get_subnodes(df, str_attr):
	new_col_name = str_attr.rstrip('_')
	columns_with_col = df.filter(like=str(str_attr))
	
	return new_col_name, columns_with_col.columns.tolist()

def scale_subnode_groups(df, basename):
	m = 1
	col_names = [basename + str(m) for m in range(1, n + 1)]
	for col in col_names:
		col_name, subnodes = get_subnodes(df, str(col))
		df = combine_and_scale(df, subnodes, col_name)
		
	cols = df.columns.tolist()
	cols = cols[1:] + cols[:1]
	df = df[cols]
	# df.fillna(0, inplace=True)
	
	return df

def combine_and_scale(df, binary_cols, combined_col_name):
	attr_name = combined_col_name.strip('_')
	attr_num = attr_name.strip('Attribute')
	df[combined_col_name] = np.zeros(len(df), dtype=np.float64)
	data_type = data_variables.where(data_variables['name'] == str(attr_name), other=None)['type'].dropna()
	X_unique_cts_lst = X_unique_cts.tolist()
	
	min_val = 0.01
	max_val = 0.99
	
	for i, col in enumerate(binary_cols):
		binary_col_ending = col.strip(attr_name)
		binary_col_ending = binary_col_ending.strip('_')
		if len(binary_cols) != 0:
			init_vals = df[col].tolist()
			unique_init_vals = np.unique(init_vals)
			sorted_unique_init_vals = sorted(unique_init_vals)
			band_gap = np.round(float(1 / float((X_unique_cts_lst[int(attr_num) - 1]) + 2)), 10)
			total_nodes_unique = X[attr_name].unique().tolist()
			key_range = list(range(1, len(total_nodes_unique) + 1))
			vals_dict = dict(zip(key_range, total_nodes_unique))
			key_range2 = list(range(1, len(sorted_unique_init_vals) + 1))
			col_dict = dict(zip(key_range2, sorted_unique_init_vals))
			new_vals_dict = {v : k for k, v in vals_dict.items()}
			
			if data_type.item() == 'Categorical' and (X_unique_cts_lst[int(attr_num) - 1] != 2):
				key = new_vals_dict.get(str(binary_col_ending))
				if key is not None:
					df.loc[df[col] == 1, combined_col_name] = np.round(np.float64(key) * np.float64(band_gap), 10)
				else:
					continue
			elif data_type.item() == 'Integer' and (X_unique_cts_lst[int(attr_num) - 1] != 2):
				new_dict = {str(key): value for key, value in new_vals_dict.items()}
				key = new_dict.get(str(binary_col_ending))
				if key is not None:
					df.loc[df[col] == 1, combined_col_name] = np.round(np.float64(key) * np.float64(band_gap), 10)
				else:
					continue
			elif (data_type.item() == 'Binary') or (X_unique_cts_lst[int(attr_num) - 1] == 2):
				col_a = binary_cols[0]
				col_b = binary_cols[1]
				df[combined_col_name] = np.float64(df[col_a].fillna(df[col_b]))
		else:
			df[combined_col_name] += 0.0
	df.drop(binary_cols, axis=1, inplace=True)
	
	return df

def preprocess_ohe_data(ohe_data):
	str_attr2 = 'Attribute'
	attr_df = ohe_data.filter(like=str(str_attr2))
	attr_df_num_cols = len(attr_df.columns)
	previous_df = ohe_data # .copy()
	attr_cols_lst = attr_df.columns.tolist()
	sorted_attr_cols_lst = sorted(attr_cols_lst)
	
	X_unique_attr_lst = X_unique_cts.index[:].tolist()
	for attr in X_unique_attr_lst:
		current_df = previous_df
		attr_name = attr + '_'
		attribute_col_name2, attribute_subnodes2 = get_subnodes(current_df, str(attr_name))
		scale_attribute_df2 = combine_and_scale(current_df, attribute_subnodes2, attribute_col_name2)
		
		previous_df = scale_attribute_df2
		
	cols = scale_attribute_df2.columns.tolist()
	cols = cols[1:] + cols[:1]
	scale_attribute_df2 = scale_attribute_df2[cols]
	
	return scale_attribute_df2

# data_copy = data.copy()

data2 = pd.concat([data_bad, data_good])
data_copy2 = data2.copy()
data_copy2['class'] = data_copy2['class'].where(data_copy2['class'] == 1, np.float64(0.0))
data_copy2 = data_copy2.astype(np.float64).interpolate() 
scale_attribute_df2 = preprocess_ohe_data(data_copy2)

#print(scale_attribute_df2.columns.tolist())
scale_attribute_df2_unique_cts = scale_attribute_df2.nunique()
print("\n")
print(data_copy2)
print(scale_attribute_df2)
print(scale_attribute_df2_unique_cts)
print(X_unique_cts.tolist())
scale_attribute_df2.iloc[0]
print(scale_attribute_df2.to_dict(orient='list'))

X_2 = scale_attribute_df2.copy()
X3 = X_2.iloc[:,:-1]
y3 = data_copy2.iloc[:,-1]
y3 = y3.where(y == 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X3,
                                                    y3,
                                                    test_size=0.2,
                                                    stratify=y3,
                                                    random_state=123)
                                                    
X_train_scaled = scaler_transform(data=X_train)
X_test_scaled = scaler_transform(data=X_test)
print(type(X_train_scaled))
##########################################################################################
##########################################################################################
# end of bloq_10.py, start of bloq_19.py
##########################################################################################
import networkx as nx
import matplotlib.pyplot as plt

import dwave_networkx as dnx
import networkx as nx
from dwave.cloud import Client
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite, LeapHybridCQMSampler, LazyFixedEmbeddingComposite
from minorminer import find_embedding, busclique

import dwave.inspector

import itertools
from dwave.embedding import diagnose_embedding
from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Integer, StructureComposite, ExactSolver, ExactCQMSolver
import dwave.embedding
from dwave.embedding.chain_strength import scaled

import pandas as pd

import datetime
import os
import pickle
import json
import ast
import numpy as np

# bqm = BinaryQuadraticModel()

cqm = ConstrainedQuadraticModel()

n = 21
G1 = nx.complete_graph(n)

print(G1)
# Draw the graph
pos = nx.spring_layout(G1)  # Use a layout algorithm to position nodes
nx.draw(G1, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')

#nx.draw(G1, with_labels=True)
plt.show()

nx.draw(G1, pos, with_labels=False, node_color='skyblue', node_size=10, edge_color='gray')
plt.show()

#qpu = DWaveSampler(solver={'topology__type': 'pegasus'})
qpu = DWaveSampler(solver={'topology__type': 'zephyr'})
#print(qpu.properties) 

size = 21
pegasus_graph = dnx.pegasus_graph(size)
p1 = dnx.pegasus_graph(1)
zephyr_graph = dnx.zephyr_graph(size)

embedding = find_embedding(G1, zephyr_graph, random_seed=0)
embedding2 = {0: [6708, 10330, 6707], 1: [6855, 10876, 6856], 2: [10540, 6666], 3: [6960, 10519, 10520], 4: [6604, 6603, 10372], 5: [10750, 6582], 6: [6834, 10582], 7: [6645, 6646], 8: [10666, 6876], 9: [6561, 6562, 10288], 10: [10478, 10477], 11: [10456, 6729], 12: [6750, 10624], 13: [10708, 6792], 14: [6772, 6771, 10918], 15: [6688, 10393, 6687], 16: [6624, 10646, 10645], 17: [10435, 10436, 7002], 18: [6814, 10792, 6813], 19: [6897, 10561, 10562], 20: [10499, 10498]}
print(embedding)
print(embedding2)

diagnosis = diagnose_embedding(embedding, G1, zephyr_graph)
for problem in diagnosis:
	print(problem)

combinations = generate_combinations(n)
#print(combinations)
#print(len(combinations))

#sampler = DWaveSampler(solver=dict(topology__type='pegasus'))
sampler = DWaveSampler(solver=dict(topology__type='zephyr'))
print("1...")

structured_composite_sampler = StructureComposite(embedding, G1.nodes, G1.edges)
print("2...")

num_samples = 10
fixed_sampler = LazyFixedEmbeddingComposite(sampler)
sampler_extended_j_range = fixed_sampler.child.properties["h_range"]
samlpler_h_range = fixed_sampler.child.properties["extended_j_range"]
print("sampler h range:  {}".format(samlpler_h_range))
print("sampler extended j range: {}".format(sampler_extended_j_range))

first_two_rows = X_train_scaled.head(2)

array_lst = []
for index, spins in first_two_rows.iterrows():  # replace with X_train_scaled for full length data runs, first_two_rows for testing
	h = {key: val for key, val in zip(G1.nodes, spins)}
	J = {(u, v): 0.5 for u, v in G1.edges}
	#print("h:  {}".format(h))
	#print("J:  {}".format(J))
	
	# sample
	#sampleset = fixed_sampler.sample_ising(h, J, num_reads=num_samples, auto_scale=True, chain_strength=scaled) # bloq_19.py ln 203
	
	# process sample_results
	#sampleset_record = sampleset.record
	#sampleset_record_arr = np.array(sampleset_record["sample"][0], dtype=np.float64)
	#array_lst.append(sampleset_record_arr) # after line 310 bloq_19.py
		
###################################################################################################################

def load_array_lst_txt(dir):
	for filename in os.listdir(dir):
		filepath = os.path.join(dir, filename)
		if filename == "array_lst.npz":
			print(filepath)
			file_content = np.load(filepath)
			#file_content = []
			#with open(filepath, "r") as f:
			#	file_content.append(f.read())
	
	return file_content

def scaler_transform(data, scaler=StandardScaler()):
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame(data_scaled)
    data_scaled.columns = data.columns
    data_scaled.index = data.index
    return data_scaled

directory_path = "./data/QRC_CQIM_zephyr_2025-01-02_20-24-36/"

# collect individual best rows (will be numpy ndarray(sample) or recarray(records)))from each run into feature vectors in a list....

array_lst = load_array_lst_txt(directory_path)
print(array_lst['arr_0'])
print(type(array_lst['arr_0']))
print(array_lst['arr_0'].dtype)
print(array_lst['arr_1'])
print(array_lst)
print(type(array_lst))
##########################################################################################
import numpy as np
import pandas as pd
from pyqubo import Array, Constraint, Placeholder
from dwave.cloud import Client
from dwave.system import DWaveSampler, EmbeddingComposite

from dwave.samplers import SimulatedAnnealingSampler

# endpoint = 'https://cloud.dwavesys.com/sapi'
# token = '***' # Please relace with your own token

list_feat_vecs = []
for key, array in array_lst.items():
	for item in np.ndenumerate(array):
		#print(item)
		if item[0][0] != 20:
			list_feat_vecs.append(item[1])

list_feat_vecs = [list_feat_vecs[i:i + 20] for i in range(0, len(list_feat_vecs), 20)]
#list_feat_vecs = list_feat_vecs * 10

print("list_feat_vecs:  {}".format(list_feat_vecs))
X = np.vstack(list_feat_vecs)
X = X.astype(np.float64)
print(X)
print(len(X))

list_y_vecs = []
for key, array in array_lst.items():
	y_vector_arr = array[-1]
	list_y_vecs.append(y_vector_arr)
	
y_arr = np.vstack(list_y_vecs)
y_arr = y_arr.astype(np.float64)
sign_y_arr = np.where(y_arr < 0, 0, y_arr)
binary_y_arr = np.round(sign_y_arr)

ans_set_lst = []
for arr in X:
	print(type(arr))
	sign_P = np.where(arr < 0, 0, arr)
	P = np.round(sign_P)
	print("P:  {}".format(P))
	matrix = np.diag(P)
	transposed_matrix = np.transpose(matrix)
	print("matrix:  {}".format(matrix))
	
	N = 20 # size of feature vector
	x = Array.create('x', shape=N, vartype='BINARY') # array takes binary 0 or 1 indicate Do Not Lend (credit risk) and Lend (not a credit risk)
	
	# number of outcomes constraint
	K = 1 # specify the desired number of outcomes
	h_const1 = (K - np.dot(x, x))**2 # RMSE - minimization of penalty

	P_mean = np.mean(P)
	print("P_mean:  {}".format(P_mean))
	
	# cost function 1: maximize probability
	h_cost1 = 0
	h_cost1 -= np.dot(x, P) # P: feature vector length N=20

	# cost function 2: minimize covariance
	h_cost2 = 0
	i = 0
	for i in range(N):
		for j in range(N):
			Q = ((P[i] - P_mean) * (P[j] - P_mean)) / len(P)
			h_cost2 += x[i] * x[j] * np.sum(Q)

	h_cost = h_cost1 #+ h_cost2

	# prepare QUBO
	h = h_cost + Placeholder('lambda') * Constraint(h_const1, label='K')
	model = h.compile()

	feed_dict = {'lambda': 2}
	qubo, offset = model.to_qubo(feed_dict=feed_dict)

	print("offset:  {}".format(offset))

	# sample optimization results
	#sampler_dw = DWaveSampler(solver='Advantage_system4.1',token=token)
	#sampler_qa = EmbeddingComposite(sampler_dw)
	sampler_qa = SimulatedAnnealingSampler()

	sampleset_qa = sampler_qa.sample_qubo(qubo, num_reads=10)
	records_qa = sampleset_qa.record
	print(records_qa)
	
	sampleset_record_arr = np.array(records_qa["sample"][0], dtype=np.float64)

	record_len = len(sampleset_record_arr)
	#ans_set_lst = []
	
	for i in range(record_len):
		ans_set = []
		if sampleset_record_arr[i] == 1:
			if (int(arr[i])) == 1 or (int(arr[i]) == -1):
				ans_set.append(int(arr[i]))
				ans_set_lst.append(arr[i])

binary_predictions_test = []
for item in ans_set_lst:
	sign_predictions_test_item = np.where(item < 0, 0, item)
	binary_predictions_test_item = np.round(sign_predictions_test_item)
	binary_predictions_test.append(int(binary_predictions_test_item))
		
# change this so that negative values return as zeros
print("Binary Predictions:", binary_predictions_test)

##########################################################################################
# end of bloq_18.py, start of ...
##########################################################################################
#print(binary_y_arr)
def check_all_n(list_of_arrays, m):
    for array in list_of_arrays:
        if not all(x == m for x in array):
            return False
    return True
    
#binary_y_arr = binary_predictions_test

y_arr = np.vstack(binary_predictions_test)
y_arr = y_arr.astype(np.float64)
sign_y_arr = np.where(y_arr < 0, 0, y_arr)
binary_y_arr = np.round(sign_y_arr)
    
if check_all_n(binary_y_arr, 1) == True:
	cm_train = [int(len(binary_y_arr))]

elif check_all_n(binary_y_arr, 0) == True:
	cm_train = [int(len(binary_y_arr))]
else:
	cm_train = confusion_matrix(y_true=binary_y_arr, y_pred=binary_predictions_test)
	
print("Results of Training:  \n")
# cm_train = confusion_matrix(y_true=binary_y_arr, y_pred=binary_predictions)
print("Confusion Matrix:  \n{}".format(cm_train))
print("\n")

if (check_all_n(binary_y_arr, 1) == True) or (check_all_n(binary_y_arr, 0) == True):
	print("Single value binary array - no classification report available.")
else:
	print(classification_report(y_true=binary_y_arr,
								y_pred=binary_predictions_test,
								target_names=["good", "bad"]))
print("\n")

#precision = precision_score(y_arr, predictions)
#recall = recall_score(y_arr, predictions)
#accuracy = accuracy_score(y_arr, predictions)

#print("Accuracy: {}".format(accuracy))
#print("Precision: {}".format(precision))
#print("Recall: {}".format(recall))

##########################################################################################
# model testing
##########################################################################################
print("\n")
print("\n")

# X_test_scaled = scaler_transform(data=X_test)
# 
# model.fit(X, y_arr)
# #y_pred_test = logreg.predict(X_test_scaled)
# 
# # Make predictions
# predictions_test = model.predict(X)
# 
# # Print coefficients and intercept
# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)
# 
# # Print predictions
# print("Predictions:", predictions_test)

#Print binary predictions
# abs_predictions_test = np.abs(predictions_test)
# sign_predictions_test = np.where(predictions_test < 0, 0, predictions_test)
# binary_predictions_test = np.round(sign_predictions_test)

# change this so that negative values return as zeros
print("Binary Predictions:", binary_predictions_test)

#sign_y_arr_test = np.where(y_arr_test < 0, 0, y_arr_test)
#binary_y_arr_test = np.round(sign_y_arr_test)

if check_all_n(binary_y_arr, 1) == True:
	cm_test = [int(len(binary_y_arr))]

elif check_all_n(binary_y_arr, 0) == True:
	cm_test = [int(len(binary_y_arr))]
else:
	cm_test = confusion_matrix(y_true=binary_y_arr, y_pred=binary_predictions_test)
	
print("Results of Testing:  \n")
#cm_test = confusion_matrix(y_true=binary_y_arr, y_pred=binary_predictions_test)
print("Confusion Matrix:  \n{}".format(cm_test))
print("\n")

if (check_all_n(binary_y_arr, 1) == True) or (check_all_n(binary_y_arr, 0) == True):
	print("Single value binary array - no classification report available.")
else:
	print(classification_report(y_true=binary_y_arr,
								y_pred=binary_predictions_test,
								target_names=["good", "bad"]))
print("\n")

#precision_test = precision_score(y_arr, predictions_test)
#recall_test = recall_score(y_arr, predictions_test)
#accuracy_test = accuracy_score(y_arr, predictions_test)

#print("Accuracy: {}".format(accuracy_test))
#print("Precision: {}".format(precision_test))
#print("Recall: {}".format(recall_test))




