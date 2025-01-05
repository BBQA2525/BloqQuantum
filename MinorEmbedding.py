import itertools

def generate_combinations(n):
    elements = list(range(1, n+1))
    return list(itertools.combinations(elements, 2))

def make_combinations(df, n):
    """Generates all possible combinations of columns as separate DataFrames."""
    column_names = df.columns.tolist()
    combinations_list = []

    # Iterate over all possible combination lengths
    for r in range(n, n + 1):
        for combination in itertools.combinations(column_names, r):
            combinations_list.append(df[list(combination)])

    return combinations_list