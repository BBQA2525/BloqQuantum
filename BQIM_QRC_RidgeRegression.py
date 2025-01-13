import networkx as nx
import matplotlib.pyplot as plt
import dwave_networkx as dnx
from dwave.cloud import Client
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite, LeapHybridCQMSampler, LazyFixedEmbeddingComposite
from minorminer import find_embedding, busclique
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score, recall_score
import random
import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import itertools
from itertools import combinations
from statlog_german_credit_data_schema import column_list
from ClassicalPreprocess import ohe, scaler_transform
from QuantumPreprocess import scale_subnode_groups, get_subnodes
from MinorEmbedding import generate_combinations, make_combinations
from ucimlrepo import fetch_ucirepo
#from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import dwave.inspector
import dwave.embedding
from dwave.embedding import diagnose_embedding
from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Integer, StructureComposite, ExactSolver, ExactCQMSolver
from dwave.embedding.chain_strength import scaled
import datetime
import os
import pickle
import json
import ast
from sklearn.linear_model import LinearRegression
import re
  
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

data2 = pd.concat([data_bad, data_good])
data_copy2 = data2.copy()
data_copy2['class'] = data_copy2['class'].where(data_copy2['class'] == 1, np.float64(0.0))
data_copy2 = data_copy2.astype(np.float64).interpolate() 
scale_attribute_df2 = data_copy2

scale_attribute_df2_unique_cts = scale_attribute_df2.nunique()
print("\n")
#print(data_copy2)
#print(scale_attribute_df2)
#print(scale_attribute_df2_unique_cts)
#print(X_unique_cts.tolist())
scale_attribute_df2.iloc[0]
#print(scale_attribute_df2.to_dict(orient='list'))

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
#print(type(X_train_scaled))

# bqm = BinaryQuadraticModel()

cqm = ConstrainedQuadraticModel()

n = 3
G1 = nx.complete_graph(n)

print(G1)
# Draw the graph
pos = nx.spring_layout(G1)  # Use a layout algorithm to position nodes
nx.draw(G1, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')

#nx.draw(G1, with_labels=True)
plt.show()

#nx.draw(G1, pos, with_labels=False, node_color='skyblue', node_size=10, edge_color='gray')
#plt.show()

#qpu = DWaveSampler(solver={'topology__type': 'pegasus'})
qpu = DWaveSampler(solver={'topology__type': 'zephyr'})
#print(qpu.properties) 

size = 3
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

#sampler = DWaveSampler(solver=dict(topology__type='pegasus'))
sampler = DWaveSampler(solver=dict(topology__type='zephyr'))
#print("1...")

structured_composite_sampler = StructureComposite(embedding, G1.nodes, G1.edges)
#print("2...")

num_samples = 1024
fixed_sampler = LazyFixedEmbeddingComposite(sampler)
sampler_extended_j_range = fixed_sampler.child.properties["h_range"]
samlpler_h_range = fixed_sampler.child.properties["extended_j_range"]
print("sampler h range:  {}".format(samlpler_h_range))
print("sampler extended j range: {}".format(sampler_extended_j_range))

bqim_X_transformed_df = X_train_scaled.copy() # X_train.copy()
#print(bqim_X_transformed_df.head(2))

first_two_rows = bqim_X_transformed_df.head(2)
#print("first_two_rows:  {}".format(first_two_rows))
	
break_df = make_combinations(bqim_X_transformed_df, n - 1)
#print("break_df:  {}".format(break_df))
print("number of df combinations (non-self quadratic terms): {}".format(len(break_df)))
#print("sample of a single df in list:  \n{}".format(break_df[0]))

y_train_df = y_train.copy()
first_two_y_rows = y_train_df.head(2)
#print("first_two_y_rows:  \n{}".format(first_two_y_rows))
df_lst = []
for df in break_df:
	break_df_class_df = pd.concat([df, y_train_df], axis=1)
	df_lst.append(break_df_class_df)
	
#print("sample of new df with 'class' (y-var) col added (triangle topology):  \n{}".format(df_lst[0]))
#print("len of new df list:  {}".format(len(df_lst)))

def quantum_annealing(df, graph, J_val):
	array_lst = []
	for index, spins in df.iterrows():  
		h = {key: val for key, val in zip(graph.nodes, spins)}
		J = {(u, v): int(J_val) for u, v in graph.edges}
	
		sampleset = fixed_sampler.sample_ising(h, J, num_reads=num_samples, auto_scale=True, chain_strength=scaled)
		sampleset_record = sampleset.record
		sampleset_record_arr = np.array(sampleset_record["sample"][0], dtype=np.float64)
		array_lst.append(sampleset_record_arr)

for df in df_lst:
	quantum_annealing(df, G1, J_val=0.5)

def create_array_lst_txt_file(dir, filename, obj):
	os.makedirs(dir, exist_ok=True)
	new_filename = f"{filename}.npz"
	filepath = os.path.join(dir, new_filename)
	np.savez(filepath, *obj)

def create_embedding_json_file(dir, data):
	os.makedirs(dir, exist_ok=True)
	embedding_filepath = os.path.join(dir, "embedding.json")
	embedding_json = json.dumps(str(embedding))
	with open(embedding_filepath, 'w') as f:
		f.write(embedding_json)
		
def create_h_J_jsons(dir):
	os.makedirs(dir, exist_ok=True)
	h_filepath = os.path.join(dir, "h.json")
	h_json = json.dumps(str(h))
	with open(h_filepath, 'w') as f:
		f.write(h_json)
	J_filepath = os.path.join(dir, "J.json")
	J_json = json.dumps(str(J))
	with open(J_filepath, 'w') as f:
		f.write(J_json)

def create_meta_data_json_file(dir, filename, data):
	os.makedirs(dir, exist_ok=True)
	filepath = os.path.join(dir, filename)
	with open(filepath, 'w') as f:
		json.dump(data, f, indent=4)

def create_sampleset_json_file(dir, filename, obj):
	os.makedirs(dir, exist_ok=True)
	new_filename = f"{filename}.json"
	filepath = os.path.join(dir, new_filename)
	samplset_json = json.dumps(str(obj))
	with open(filepath, 'w') as f:
		f.write(samplset_json)
		
def create_sampleset_record_json_file(dir, filename, obj):
	os.makedirs(dir, exist_ok=True)
	new_filename = f"{filename}.json"
	filepath = os.path.join(dir, new_filename)
	samplset_json = json.dumps(str(obj))
	with open(filepath, 'w') as f:
		f.write(samplset_json)
		
def create_sampleset_warnings_txt_file(dir, filename, obj):
	os.makedirs(dir, exist_ok=True)
	new_filename = f"{filename}.txt"
	filepath = os.path.join(dir, new_filename)
	with open(filepath, 'w') as f:
		f.write(str(obj))

def createdir_if_not_exists(dir):
	try:
		os.makedirs(dir)
	except FileExistsError:
		print("Directory '{}' already exists.".format(dir))

def load_array_lst_txt(dir):
	for filename in os.listdir(dir):
		filepath = os.path.join(dir, filename)
		if filename == "array_lst.npz":
			#print(filepath)
			file_content = np.load(filepath)
	
	return file_content

def load_and_pretty_print_json(dir):
	for filename in os.listdir(dir):
		#if filename.endswith(".json"):
		if filename == "meta_data.json":
			file_path = os.path.join(dir, filename)
			#print(file_path)
			with open(file_path, "r") as f:
				data = json.load(f)
				#print(json.dumps(data, indent=4))
				
	return json.dumps(data, indent=4)

def load_h_J_jsons(dir):
	for filename in os.listdir(dir):
		if filename == "h.json":
			h_file_path = os.path.join(dir, filename)
			#print(h_file_path)
			with open(h_file_path, "r") as f:
				h_data = json.load(f)
		if filename == "J.json":
			J_file_path = os.path.join(dir, filename)
			#print(J_file_path)
			with open(J_file_path, "r") as f:
				J_data = json.load(f)
				
	return h_data, J_data

def load_embedding_json(dir):
	for filename in os.listdir(dir):
		if filename == "embedding.json":
			file_path = os.path.join(dir, filename)
			#print(file_path)
			with open(file_path, "r") as f:
				data = json.load(f)
				
	return data

def load_sampleset_json(dir):
	for filename in os.listdir(dir):
		filepath = os.path.join(dir, filename)
		if filename == "sampleset.json":
			#print(filepath)
			with open(filepath, "r") as f:
				data = json.load(f)
				#print(json.dumps(data, indent=4))
	
	return data

def load_sampleset_record_json(dir):
	for filename in os.listdir(dir):
		filepath = os.path.join(dir, filename)
		if filename == "sampleset_record.json":
			#print(filepath)
			with open(filepath, "r") as f:
				data = json.load(f)
				#print(json.dumps(data, indent=4))
	
	return data

def load_sampleset_warnings_txt(dir):
	for filename in os.listdir(dir):
		filepath = os.path.join(dir, filename)
		if filename == "sampleset_warnings.txt":
			#print(filepath)
			file_content = []
			with open(filepath, "r") as f:
				file_content.append(f.read())
	
	return file_content

qpu_topology = sampler.properties["topology"] 	
now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
sampleset_filename = "sampleset"
directory_path = "./data/QRC_BQIM_RidgeRegression_" + str(qpu_topology['type']) + "_" + date_string

array_lst_filename = "array_lst"
create_array_lst_txt_file(directory_path, array_lst_filename, array_lst)

saved_array_lst_txt = load_array_lst_txt(directory_path)
#print("3...")
print(sampleset.first)

print(sampleset)
dwave.inspector.show(sampleset) 
samplset2 = sampleset.data(sorted_by='energy', reverse=False)
#print(samplset2) # it appears to already by sorted this way by default


##########################################################################################
#    post-processing begins here
##########################################################################################
#qpu_topology = sampler.properties["topology"] 
#qpu_version = sampler.properties["version"] 
qpu_num_qubits = sampler.properties["num_qubits"] 

sampleset_problem_id = sampleset.info["problem_id"]
sampleset_timing = sampleset.info["timing"]
sampleset_embedding_context = sampleset.info["embedding_context"]
sampleset_warnings = sampleset.info["warnings"]
print(sampleset_embedding_context)
print(sampleset_warnings)
sampleset_record = sampleset.record
print(sampleset_timing["qpu_sampling_time"])
qpu_sampling_time = sampleset_timing["qpu_sampling_time"]
qpu_anneal_time_per_sample = sampleset_timing["qpu_anneal_time_per_sample"]
qpu_readout_time_per_sample = sampleset_timing["qpu_readout_time_per_sample"]
qpu_access_time = sampleset_timing["qpu_access_time"]
qpu_access_overhead_time = sampleset_timing["qpu_access_overhead_time"]
qpu_programming_time = sampleset_timing["qpu_programming_time"]
qpu_delay_time_per_sample = sampleset_timing["qpu_delay_time_per_sample"]
total_post_processing_time = sampleset_timing["total_post_processing_time"]
post_processing_overhead_time = sampleset_timing["post_processing_overhead_time"]
print("qpu_topology:  {}".format(qpu_topology))

print(directory_path)
createdir_if_not_exists(directory_path)
create_sampleset_json_file(directory_path, sampleset_filename, sampleset)
sampleset_record_filename = "sampleset_record"
create_sampleset_record_json_file(directory_path, sampleset_record_filename, sampleset_record)
sampleset_warnings_filename = "sampleset_warnings"
create_sampleset_warnings_txt_file(directory_path, sampleset_warnings_filename, sampleset_warnings)

create_h_J_jsons(directory_path)
	
saved_h_json, saved_J_json = load_h_J_jsons(directory_path)
#print("h:  {}".format(saved_h_json))
#print("J:  {}".format(saved_J_json))

meta_data = {
	"qpu_topology": qpu_topology,
	"qpu_num_qubits": qpu_num_qubits,
	"sampleset_problem_id": sampleset_problem_id,
	"sampleset_timing": sampleset_timing,
	"sampleset_embedding_context": sampleset_embedding_context,
	#"sampleset_warnings": sampleset_warnings,
	#"qpu_sampling_time": qpu_sampling_time,
	#"qpu_anneal_time_per_sample": qpu_anneal_time_per_sample,
	#"qpu_readout_time_per_sample": qpu_readout_time_per_sample,
	#"qpu_access_time": qpu_access_time,
	#"qpu_access_overhead_time": qpu_access_overhead_time,
	#"qpu_programming_time": qpu_programming_time,
	#"qpu_delay_time_per_sample": qpu_delay_time_per_sample,
	#"total_post_processing_time": total_post_processing_time,
	#"post_processing_overhead_time": post_processing_overhead_time,
}

meta_data_filename = 'meta_data.json'
create_meta_data_json_file(directory_path, meta_data_filename, meta_data)
	
create_embedding_json_file(directory_path, embedding)
	
saved_embedding_json = load_embedding_json(directory_path)
#print(saved_embedding_json)
				
saved_meta_data_json = load_and_pretty_print_json(directory_path)
#print(saved_meta_data_json)

saved_sampleset_warnings_txt = load_sampleset_warnings_txt(directory_path)
#print(saved_sampleset_warnings_txt)

saved_sampleset_json = load_sampleset_json(directory_path)
#print(saved_sampleset_json)

saved_sampleset_record_json = load_sampleset_record_json(directory_path)
#print(saved_sampleset_record_json)

#print("Percentage of samples with >10% chain breaks is {} and >0 is {}".format(
#	np.count_nonzero(sampleset.record.chain_break_fraction > 0.10)/num_samples*100,
#	np.count_nonzero(sampleset.record.chain_break_fraction > 0.00)/num_samples*100))

print(sampleset_record)
# print(sampleset_record["energy"])
# print(min(sampleset_record["energy"]))
# print(sampleset_record["sample"][0])

sample_min_energy = min(sampleset_record["energy"])

# print(type(sampleset_record))

row_indexes = np.where(sampleset_record['energy'] == sample_min_energy)[0]

# print(row_indexes)
# print(sampleset_record[0])
# print(type(sampleset_record["sample"][0]))

array_lst = load_array_lst_txt(directory_path)

list_feat_vecs = []
for key, array in array_lst.items():
	for item in np.ndenumerate(array):
		print(item)
		if item[0][1] != 2:
			#print(item)
			list_feat_vecs.append(item[1])

list_feat_vecs = [list_feat_vecs[i:i + 380] for i in range(0, len(list_feat_vecs), 380)]

# print("array_lst:  {}".format(array_lst))
# print("list_feat_vecs:  {}".format(list_feat_vecs))
X = np.vstack(list_feat_vecs)
X = X.astype(np.float64)
# print(X)
# print(len(X))

# Target array
# or 0, also think about changing feature vectors to have -1 -> 0 ??
df_lst[2] = df_lst[2].astype(int)
list_y_vecs = df_lst[2].values.tolist()
list_y_vecs2 = []
for item in list_y_vecs:
	list_y_vecs2.append(item[2])
list_y_vecs = list_y_vecs2
	
y_arr = np.vstack(list_y_vecs)
y_arr = y_arr.astype(np.float64)
# print(y_arr)
# print(len(y_arr))
# 
# print("X:  {}".format(type(X)))
# print("y_arr:  {}".format(type(y_arr)))
# print("X:  {}".format(X.dtype))
# print("y_arr:  {}".format(y_arr.dtype))
# sign_y_arr = np.where(y_arr < 0, 0, y_arr)
binary_y_arr = np.round(sign_y_arr)

# Create and fit the linear model
# repeated 10-fold cross-validation
#### num_split has to be no greater than number of samples
cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=1)
# RidgeCV hyperparameter tuning
model = RidgeCV(alphas=arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
# fit model
model.fit(X, y_arr)

# Make predictions
predictions = model.predict(X)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Print predictions
print("Predictions:", predictions)

# Print binary predictions
abs_predictions = np.abs(predictions)
sign_prediction = np.where(predictions < 0, 0, predictions)
binary_predictions = np.round(sign_prediction)

# change this so that negative values return as zeros
print("Binary Predictions:", binary_predictions)

def check_all_n(list_of_arrays, m):
    for array in list_of_arrays:
        if not all(x == m for x in array):
            return False
    return True
    
if check_all_n(binary_y_arr, 1) == True:
	cm_train = [int(len(binary_y_arr))]

elif check_all_n(binary_y_arr, 0) == True:
	cm_train = [int(len(binary_y_arr))]
else:
	cm_train = confusion_matrix(y_true=binary_y_arr, y_pred=binary_predictions)
	
print("Results of Training:  \n")
# cm_train = confusion_matrix(y_true=binary_y_arr, y_pred=binary_predictions)
print("Confusion Matrix:  \n{}".format(cm_train))
print("\n")

if (check_all_n(binary_y_arr, 1) == True) or (check_all_n(binary_y_arr, 0) == True):
	print("Single value binary array - no classification report available.")
else:
	print(classification_report(y_true=binary_y_arr,
								y_pred=binary_predictions,
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

X_test_scaled = scaler_transform(data=X_test)

bqim_X_transformed_df_test = X_test_scaled.copy() # X_train.copy()
#print(bqim_X_transformed_df_test.head(2))

first_two_rows_test = bqim_X_transformed_df_test.head(2)
	
break_df_test = make_combinations(bqim_X_transformed_df_test, n - 1)

y_test_df = y_test.copy()
first_two_y_rows_test = y_test_df.head(2)
df_lst_test = []
for df in break_df_test:
	break_df_class_df = pd.concat([df, y_test_df], axis=1)
	df_lst.append(break_df_class_df)

#quit()

for df in df_lst_test:
	quantum_annealing(df, G1, J_val=0.5)

now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
directory_path_test = "./data/QRC_BQIM_RidgeRegression_test_" + str(qpu_topology['type']) + "_" + date_string

create_array_lst_txt_file(directory_path_test, array_lst_filename, array_lst)
saved_array_lst_txt = load_array_lst_txt(directory_path_test)


##########################################################################################
#    test data post-processing begins here
##########################################################################################
#qpu_topology = sampler.properties["topology"] 
#qpu_version = sampler.properties["version"] 
qpu_num_qubits = sampler.properties["num_qubits"] 

sampleset_problem_id = sampleset.info["problem_id"]
sampleset_timing = sampleset.info["timing"]
sampleset_embedding_context = sampleset.info["embedding_context"]
sampleset_warnings = sampleset.info["warnings"]
sampleset_record = sampleset.record
qpu_sampling_time = sampleset_timing["qpu_sampling_time"]
qpu_anneal_time_per_sample = sampleset_timing["qpu_anneal_time_per_sample"]
qpu_readout_time_per_sample = sampleset_timing["qpu_readout_time_per_sample"]
qpu_access_time = sampleset_timing["qpu_access_time"]
qpu_access_overhead_time = sampleset_timing["qpu_access_overhead_time"]
qpu_programming_time = sampleset_timing["qpu_programming_time"]
qpu_delay_time_per_sample = sampleset_timing["qpu_delay_time_per_sample"]
total_post_processing_time = sampleset_timing["total_post_processing_time"]
post_processing_overhead_time = sampleset_timing["post_processing_overhead_time"]

print(directory_path)
createdir_if_not_exists(directory_path_test)
create_sampleset_json_file(directory_path_test, sampleset_filename, sampleset)
sampleset_record_filename = "sampleset_record"
create_sampleset_record_json_file(directory_path_test, sampleset_record_filename, sampleset_record)
sampleset_warnings_filename = "sampleset_warnings"
create_sampleset_warnings_txt_file(directory_path_test, sampleset_warnings_filename, sampleset_warnings)

create_h_J_jsons(directory_path_test)
	
saved_h_json, saved_J_json = load_h_J_jsons(directory_path_test)

meta_data = {
	"qpu_topology": qpu_topology,
	"qpu_num_qubits": qpu_num_qubits,
	"sampleset_problem_id": sampleset_problem_id,
	"sampleset_timing": sampleset_timing,
	"sampleset_embedding_context": sampleset_embedding_context,
	#"sampleset_warnings": sampleset_warnings,
	#"qpu_sampling_time": qpu_sampling_time,
	#"qpu_anneal_time_per_sample": qpu_anneal_time_per_sample,
	#"qpu_readout_time_per_sample": qpu_readout_time_per_sample,
	#"qpu_access_time": qpu_access_time,
	#"qpu_access_overhead_time": qpu_access_overhead_time,
	#"qpu_programming_time": qpu_programming_time,
	#"qpu_delay_time_per_sample": qpu_delay_time_per_sample,
	#"total_post_processing_time": total_post_processing_time,
	#"post_processing_overhead_time": post_processing_overhead_time,
}

meta_data_filename = 'meta_data.json'
create_meta_data_json_file(directory_path_test, meta_data_filename, meta_data)
create_embedding_json_file(directory_path_test, embedding)
	
saved_embedding_json = load_embedding_json(directory_path_test)
saved_meta_data_json = load_and_pretty_print_json(directory_path_test)
saved_sampleset_warnings_txt = load_sampleset_warnings_txt(directory_path_test)
saved_sampleset_json = load_sampleset_json(directory_path_test)
saved_sampleset_record_json = load_sampleset_record_json(directory_path_test)

sample_min_energy = min(sampleset_record["energy"])
row_indexes = np.where(sampleset_record['energy'] == sample_min_energy)[0]

array_lst_test = load_array_lst_txt(directory_path_test)

list_feat_vecs_test = []
for key, array in array_lst_test.items():
	for item in np.ndenumerate(array):
		#print(item)
		if item[0][1] != 2:
			#print(item)
			list_feat_vecs_test.append(item[1])

list_feat_vecs_test = [list_feat_vecs_test[i:i + 380] for i in range(0, len(list_feat_vecs_test), 380)]

X_test = np.vstack(list_feat_vecs_test)
X_test = X_test.astype(np.float64)

# Target array
# or 0, also think about changing feature vectors to have -1 -> 0 ??
df_lst_test[2] = df_lst_test[2].astype(int)
list_y_vecs_test = df_lst_test[2].values.tolist()
list_y_vecs2_test = []
for item in list_y_vecs_test:
	list_y_vecs2_test.append(item[2])
list_y_vecs_test = list_y_vecs2_test
	
y_arr_test = np.vstack(list_y_vecs_test)
y_arr_test = y_arr_test.astype(np.float64)

sign_y_arr_test = np.where(y_arr_test < 0, 0, y_arr_test)
binary_y_arr_test = np.round(sign_y_arr_test)

# Make predictions
predictions_test = model.predict(X_test)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Print predictions
print("Predictions:", predictions_test)

# Print binary predictions
abs_predictions_test = np.abs(predictions_test)
sign_predictions_test = np.where(predictions_test < 0, 0, predictions_test)
binary_predictions_test = np.round(sign_predictions_test)

# change this so that negative values return as zeros
print("Binary Predictions:", binary_predictions_test)

if check_all_n(binary_y_arr_test, 1) == True:
	cm_test = [int(len(binary_y_arr_test))]

elif check_all_n(binary_y_arr_test, 0) == True:
	cm_test = [int(len(binary_y_arr_test))]
else:
	cm_test = confusion_matrix(y_true=binary_y_arr_test, y_pred=binary_predictions_test)
	
print("Results of Testing:  \n")
print("Confusion Matrix:  \n{}".format(cm_test))
print("\n")

if (check_all_n(binary_y_arr_test, 1) == True) or (check_all_n(binary_y_arr_test, 0) == True):
	print("Single value binary array - no classification report available.")
else:
	print(classification_report(y_true=binary_y_arr_test,
								y_pred=binary_predictions_test,
								target_names=["good", "bad"]))
print("\n")

#precision_test = precision_score(y_arr, predictions_test)
#recall_test = recall_score(y_arr, predictions_test)
#accuracy_test = accuracy_score(y_arr, predictions_test)

#print("Accuracy: {}".format(accuracy_test))
#print("Precision: {}".format(precision_test))
#print("Recall: {}".format(recall_test))