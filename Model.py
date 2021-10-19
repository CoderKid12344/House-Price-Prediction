import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Step 1 _______________________  Datasets or Data Prepare

FileName = "House Pricing.csv"
FilePath = pd.read_csv(FileName)

FilePath.columns

Data = FilePath.dropna(axis=0)

# Step 2 _______________________ Labels and Features

Y = Data.Price

Data_Features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = Data[Data_Features]

# Step 3 ________________________ Creating The Model

Model = DecisionTreeClassifier(random_state=1)

Model.fit(X, Y)

# Step 4 ________________________ Testing The Model

print(X.head())

# Step 5 _______________________ Predict

print(Model.predict(X.head()))
