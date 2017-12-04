import csv
import numpy as np
import sklearn.tree as model

i = 0
j = 0
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
		label = each_data_point.pop(0)
		if label == "1":
			j += 1
		test_labels.append(label)
		test_data.append(each_data_point)
	else:
		label = each_data_point.pop(0)
		if label == "1":
			j += 1
		train_labels.append(label)
		train_data.append(each_data_point)
	i = i + 1

train_data = np.array(train_data).astype(np.float)
train_labels = np.array(train_labels).astype(int)
test_data = np.array(test_data).astype(np.float)
test_labels = np.array(test_labels).astype(int)


print(train_data.shape, test_data.shape)

classifier = model.DecisionTreeClassifier(max_depth=100)
classifier.fit(train_data, train_labels)
score = classifier.score(test_data, test_labels)
print(score)
k = 0
for each_data in test_data:
	label = classifier.predict(each_data.reshape(1, -1))
	if label[0] == 1:
		k += 1
print(j)
print(k)