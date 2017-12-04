import csv
import numpy as np
import sklearn.linear_model as model

i = 0

train_data = []
train_labels = []
test_data = []
test_labels = []

for line in open("./../../dataset/train.csv"):
	if i == 0:
		i = i + 1
		continue;
	each_data_point = line.split(',')
	each_data_point.pop(0)
	if i % 4 == 0:
		test_labels.append(each_data_point.pop(0))
		test_data.append(each_data_point)
	else:
		train_labels.append(each_data_point.pop(0))
		train_data.append(each_data_point)
	i = i + 1

train_data = np.array(train_data).astype(np.float)
train_labels = np.array(train_labels).astype(int)
test_data = np.array(test_data).astype(np.float)
test_labels = np.array(test_labels).astype(int)

print(train_data.shape, test_data.shape)

classifier = model.LogisticRegression()
classifier.fit(train_data, train_labels)
score = classifier.score(test_data, test_labels)
print(score)