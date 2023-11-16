import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


class DecisionTreeWithInfoGain(DecisionTreeClassifier):
    def fit (self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        self._calculate_information_gain(X)

    def _calculate_information_gain (self, X):
        self.info_gain_ = self.tree_.compute_feature_importances(normalize=False)


def identify_discrete_columns (data):
    return [col for col in data.columns if data[col].dtype == 'O']


url = "https://docs.google.com/spreadsheets/d/1TNF53t5fR1whuo4_pFNPkhig_iVCkUvqD6IdDwVb3iE/export?format=csv"
df = pd.read_csv(url)

discrete_columns = identify_discrete_columns(df)

X = df[discrete_columns]
y = df['successful_transfer']

label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = DecisionTreeWithInfoGain(criterion='entropy')

model.fit(X_train, y_train)

plt.figure(figsize=(70, 40))
plot_tree(model, filled=True, feature_names=X.columns.astype(str), class_names=[str(c) for c in model.classes_],
          rounded=True, fontsize=10)

print("Information Gain for each feature:")
for feature, gain in zip(X.columns, model.info_gain_):
    print(f"{feature}: {gain}")

plt.show()
