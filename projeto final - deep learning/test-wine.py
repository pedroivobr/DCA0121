from keras.models import load_model
import numpy as np
import pandas as pd

dataset = pd.read_csv("winequality-red.csv", delimiter=";")

input = dataset.iloc[:, 0:11]
output = dataset.iloc[:, 11]


# since the data comes in a scale of 0 to 10, this is needed to we get a simple true or false
output = [(round(each / 10)) for each in output]

model = load_model('wine-model.h5')

scores = model.evaluate(input, output)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))