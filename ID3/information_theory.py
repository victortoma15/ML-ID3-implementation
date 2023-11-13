import pandas as pd
import numpy as np
from math import log2


def compute_discrete_probabilities (data):
    probability_dict = {}

    for column in data.columns:
        if data[column].dtype == 'O':  # if it has string
            probabilities = data[column].value_counts(normalize=True).to_dict()
            probability_dict[column] = probabilities

    return probability_dict


def calculate_attribute_entropy (probabilities):
    entropy = 0

    for prob in probabilities.values():
        entropy -= prob * log2(prob)

    return entropy


def calculate_max_possible_entropy (probabilities):
    max_entropy = -log2(1 / len(probabilities))
    return max_entropy


def calculate_normalized_entropy (probabilities):
    entropy = calculate_attribute_entropy(probabilities)
    max_entropy = calculate_max_possible_entropy(probabilities)
    normalized_entropy = entropy / max_entropy
    return normalized_entropy


def calculate_discrete_attributes_entropies (data, probabilities_dict):
    entropies_dict = {}

    for attribute, probabilities in probabilities_dict.items():
        normalized_entropy = calculate_normalized_entropy(probabilities)
        entropies_dict[attribute] = normalized_entropy

    return entropies_dict


def calculate_conditional_entropy (data, target_column_for_conditional_entropy,
                                   attribute_column_for_conditional_entropy):
    probability_x = data[attribute_column_for_conditional_entropy].value_counts(normalize=True).to_dict()

    conditional_entropy = 0

    for x_value in probability_x:
        subset = data[data[attribute_column_for_conditional_entropy] == x_value]
        probability_y_given_x = subset[target_column_for_conditional_entropy].value_counts(normalize=True)

        entropy_y_given_x = -np.sum(probability_y_given_x * np.log2(probability_y_given_x))

        conditional_entropy += probability_x[x_value] * entropy_y_given_x

    return conditional_entropy


def calculate_information_gain (data, target_column_for_information_gain, attribute_column_for_information_gain):
    probability_y = data[target_column_for_information_gain].value_counts(normalize=True).to_dict()
    entropy_y = calculate_attribute_entropy(probability_y)

    conditional_entropy = calculate_conditional_entropy(data, target_column_for_information_gain,
                                                        attribute_column_for_information_gain)

    information_gain = entropy_y - conditional_entropy

    return information_gain


##############################################################################################################


file_path = r"C:\Users\Victor\OneDrive\Desktop\ML-ID3-implementation\football_transfers_dataset.csv"

df = pd.read_csv(file_path)

numerical_attributes = df.select_dtypes(include=['float64', 'int64']).columns

for attribute in numerical_attributes:
    attribute_mean = round(df[attribute].mean(), 4)
    attribute_variance = round(df[attribute].var(), 4)

    print(f"\nAttribute: {attribute}")
    print(f"Mean = {attribute_mean}")
    print(f"Variance = {attribute_variance}")

print("\n")
print("--------------------------------------------------")
print("\n")

all_probabilities = compute_discrete_probabilities(df)

for attribute, probabilities in all_probabilities.items():
    print(f"Probabilities for {attribute}:\n{probabilities}")
    print("\n")

print("\n")
print("--------------------------------------------------")
print("\n")

all_entropies = calculate_discrete_attributes_entropies(df, all_probabilities)

for attribute, entropy in all_entropies.items():
    print(f"Entropy for {attribute}: {entropy}")

print("\n")
print("--------------------------------------------------")
print("\n")

target_column_for_conditional_entropy = "successful_transfer"

attribute_column_for_conditional_entropy = "position"

conditional_entropy_result = round(calculate_conditional_entropy(df, target_column_for_conditional_entropy,
                                                                 attribute_column_for_conditional_entropy), 4)

print(f"Conditional Entropy: H({target_column_for_conditional_entropy}|{attribute_column_for_conditional_entropy})= "
      f"{conditional_entropy_result}")

print("\n")
print("--------------------------------------------------")
print("\n")

target_column_for_information_gain = "successful_transfer"
attribute_column_for_information_gain = "position"

information_gain_result = round(calculate_information_gain(df, target_column_for_information_gain,
                                                           attribute_column_for_information_gain), 4)

print(f"Information Gain: Ig({target_column_for_information_gain};{attribute_column_for_information_gain})= "
      f"{information_gain_result}")

print("\n")
print("--------------------------------------------------")
print("\n")
