from information_theory import *


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

print(tree)

