from information_theory import *
import pandas as pd


def find_root_node (data, target_column):
    numerical_attributes = data.select_dtypes(include=['float64', 'int64']).columns
    discrete_probabilities = compute_discrete_probabilities(data)
    all_entropies = calculate_all_entropies(data, discrete_probabilities)

    max_information_gain = -1
    root_node = None

    for attribute in numerical_attributes:
        information_gain = round(calculate_information_gain(data, target_column, attribute), 4)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            root_node = attribute

    return root_node, max_information_gain


def id3_discrete(data, target_column, remaining_attributes=None):
    if remaining_attributes is None:
        remaining_attributes = data.columns.drop(target_column)

    unique_values_target = data[target_column].unique()
    if len(unique_values_target) == 1:
        return unique_values_target[0]

    if len(remaining_attributes) == 0:
        return data[target_column].mode().values[0]

    best_attribute = find_best_attribute(data, target_column, remaining_attributes)
    tree = {best_attribute: {"n_observations": data[best_attribute].value_counts().to_dict(),
                             "information_gain": calculate_information_gain(data, target_column, best_attribute),
                             "values": {}}}

    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        subtree = id3_discrete(subset, target_column, remaining_attributes.drop(best_attribute))
        tree[best_attribute]["values"][value] = subtree

    return tree


def find_best_attribute(data, target_column, remaining_attributes):
    information_gains = {
        attribute: calculate_information_gain(data, target_column, attribute)
        for attribute in remaining_attributes
    }
    best_attribute = max(information_gains, key=information_gains.get)
    return best_attribute


def get_splits (data, continuous_attribute, target_column):
    data[continuous_attribute] = pd.to_numeric(data[continuous_attribute], errors='coerce')

    data = data.dropna(subset=[continuous_attribute])

    unique_values = data[continuous_attribute].sort_values().unique()

    if len(unique_values) < 2:
        return []

    splits = [round((unique_values[i] + unique_values[i + 1]) / 2, 4) for i in range(len(unique_values) - 1)]
    return splits


def id3 (data, target_column, remaining_attributes=None, max_depth=None):
    if remaining_attributes is None:
        remaining_attributes = data.columns.drop(target_column)

    unique_values_target = data[target_column].unique()

    if len(unique_values_target) == 1:
        return unique_values_target[0]

    if len(remaining_attributes) == 0:
        return data[target_column].mode().values[0]

    is_continuous = data[remaining_attributes[0]].dtype in ['float64', 'int64']

    if is_continuous:
        splits = get_splits(data, remaining_attributes[0], target_column)
        best_split = find_best_split(data, remaining_attributes[0], target_column, splits)
        tree = {remaining_attributes[0]: {"is_continuous": True,
                                          "split_value": best_split,
                                          "information_gain": calculate_information_gain(data, target_column,
                                                                                         remaining_attributes[0]),
                                          "values": {}}}
        for subset in split_data_continuous(data, remaining_attributes[0], best_split):
            value = subset[remaining_attributes[0]].iloc[0]
            subtree = id3(subset, target_column, remaining_attributes.drop(remaining_attributes[0]),
                          max_depth=max_depth)
            tree[remaining_attributes[0]]["values"][value] = subtree
    else:
        best_attribute = find_best_attribute(data, target_column, remaining_attributes)
        tree = {best_attribute: {"is_continuous": False,
                                 "information_gain": calculate_information_gain(data, target_column, best_attribute),
                                 "values": {}}}
        for value in data[best_attribute].unique():
            subset = data[data[best_attribute] == value]
            subtree = id3(subset, target_column, remaining_attributes.drop(best_attribute), max_depth=max_depth)
            tree[best_attribute]["values"][value] = subtree

    if max_depth is not None and max_depth > 0:
        pruned_tree = prune_tree(tree, max_depth)
        return pruned_tree

    return tree


def prune_tree (tree, max_depth):
    if max_depth == 1:
        return None
    else:
        pruned_tree = {}
        for attribute, subtree in tree.items():
            pruned_subtree = prune_tree(subtree["values"], max_depth - 1)
            if pruned_subtree is not None:
                pruned_tree[attribute] = {"is_continuous": subtree["is_continuous"],
                                          "split_value": subtree["split_value"],
                                          "information_gain": subtree["information_gain"],
                                          "values": pruned_subtree}
        return pruned_tree


def find_best_split (data, continuous_attribute, target_column, splits):
    max_information_gain = -1
    best_split = None

    for split in splits:
        information_gain = round(calculate_information_gain(data, target_column, continuous_attribute), 4)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_split = split

    return best_split


def split_data_continuous (data, continuous_attribute, split_value):
    left_subset = data[data[continuous_attribute] <= split_value]
    right_subset = data[data[continuous_attribute] > split_value]
    return left_subset, right_subset


##############################################################################################################


print("**************************************************\n")
print("ID3 Algorithm Implementation\n")
print("**************************************************\n")

target_column = "successful_transfer"
root_node, max_information_gain = find_root_node(df, target_column)

print(f"Root node: {root_node}")
print(f"Greatest Information gain: Ig({target_column};{root_node}) = {max_information_gain}")

print("\n")
print("--------------------------------------------------")
print("\n")

tree = id3_discrete(df, target_column)
print("Discrete ID3:\n")
print(tree)

print("\n")
print("--------------------------------------------------")
print("\n")

continuous_attribute = "fee"
splits = get_splits(df, continuous_attribute, target_column)
print(f"Splits for continuous attribute {continuous_attribute}: \n")
print(splits)

print("\n")
print("--------------------------------------------------")
print("\n")

tree = id3(df, target_column)
