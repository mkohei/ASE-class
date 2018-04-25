from sklearn.neural_network import MLPClassifier

from sklearn import datasets, metrics

digits = datasets.load_digits()

# show image
"""
import pylab
pylab.gray()
pylab.matshow(digits.images[0])
pylab.show()
"""

# set of image(input data) and answer(output)
images_and_lables = list(zip(digits.images, digits.target))

# get data number
n_samples = len(digits.images)

# 8x8 -> 64x1
data = digits.images.reshape((n_samples, -1))

# split data for evaluation by hold on method
# Learning data: train / Test data: test / input: x / Teacher output: t
x_train = data[:int(n_samples/2)]
t_train = digits.target[:int(n_samples/2)]
x_test = data[int(n_samples/2):]

# expected test output
expected = digits.target[int(n_samples/2):]

# generate object of NNLM(Neural Network Leaning Machine)
clf = MLPClassifier(hidden_layer_sizes=(50, 20), random_state=1)

# fit = leaning
clf.fit(x_train, t_train)

# display parameters of LM
print(clf)

# test output by ML
predicted = clf.predict(x_test)

# Evaluation, Confusion Matrix, accuracy between expected & predicted
print(metrics.confusion_matrix(expected, predicted))
print("Classification report for classifier \n{}\n".format(metrics.classification_report(expected, predicted)))


