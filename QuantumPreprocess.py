import numpy as np

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