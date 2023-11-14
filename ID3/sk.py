import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

file_path = r"C:\Users\Victor\OneDrive\Desktop\ML-ID3-implementation\football_transfers_dataset.csv"

df = pd.read_csv(file_path)

X = df.drop('successful_transfer', axis=1)
y = df['successful_transfer']

label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='entropy')

model.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=[str(c) for c in model.classes_], rounded=True, fontsize=10)
plt.show()

