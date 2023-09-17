import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import math
from credit import Creditlist
dataset = Creditlist
X = dataset.iloc[:,1:3]# , how many columns to take]
y = dataset.iloc[:,3] # ,output]
X_train , X_test , y_train ,y_test = train_test_split(X,y,random_state=1234,test_size=0.2)
classifier = KNeighborsClassifier(n_neighbors=5,p=3)
classifier.fit(X_train, y_train)
y_pred =classifier.predict(X_test)
#print(accuracy_score(y_test, y_pred))
#print(f1_score(y_test , y_pred, average='macro'))
type_input = input("Enter the type of the device (0.5 or other): ")
year_input = input("Enter the manufacturing year of the device: ")

# Convert user input to numerical format (use one-hot encoding for 'type')
user_input = [int(type_input == '0.5'), int(year_input)]

# Make predictions based on user input
predicted_credit = classifier.predict([user_input])[0]

print(f"Predicted Credit: {predicted_credit}")