from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

dataset = pd.read_csv("winequality-red.csv", delimiter=";")

input = dataset.iloc[:, 0:11]
output = dataset.iloc[:, 11]

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
input.values[:, 10] = labelencoder_X.fit_transform(input.values[:, 10])

# since the data comes in a scale of 0 to 10, this is needed to we get a simple true or false
output = [(round(each / 10)) for each in output]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.33, random_state=42)


model = Sequential()
model.add(Dense(20, input_dim=11, init='uniform', activation='relu'))
model.add(Dense(30, init='uniform', activation='relu'))
model.add(Dense(40, init='uniform', activation='relu'))
model.add(Dense(30, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10000, batch_size=150, verbose=2)

model.save('wine-model.h7')

#analise
model = load_model('wine-model.h7')

#wine-model.h5 -> 92% 15k epocas
#h6 -> 85.5% 5k epocas
#H7 -> 93% 10K EPOCAS
scores = model.evaluate(input, output)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))