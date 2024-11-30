import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Sample dataset
data = {'age': [25, 30, np.nan, 35, 40, 22],
'income': [50000, 70000, 60000, np.nan, 80000, 90000],
'gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female'],
'purchased': ['No', 'Yes', 'No', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)
# Step 1: Handling missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[['age', 'income']] = imputer.fit_transform(df[['age', 'income']])
# Step 2: Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])],
remainder='passthrough')
df_encoded = pd.DataFrame(ct.fit_transform(df))
# Step 3: Feature scaling
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded.iloc[:, :-1]))
# Step 4: Splitting the dataset into training and testing sets
X = df_scaled
y = df_encoded.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=0)# Display the processed data
print("Original DataFrame:")
print(df)
print("\nDataFrame after handling missing data:")
print(df_encoded)
print("\nDataFrame after encoding categorical data and scaling numerical data:")
print(df_scaled)
print("\nTraining and Testing sets:")
print("X_train:\n", X_train)
print("\nX_test:\n", X_test)
print("\ny_train:\n", y_train)
print("\ny_test:\n", y_test)