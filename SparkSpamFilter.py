from pyspark import SparkContext
from pyspark.mllib import feature
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib import classification

sc = SparkContext(appName="SpamFilter")

spam_mails = sc.textFile("spam")
ham_mails = sc.textFile("ham")

features = feature.HashingTF(numFeatures = 1000)

features_spam = spam_mails.map(lambda mail: features.transform(mail.split(" ")))
features_ham = ham_mails.map(lambda mail: features.transform(mail.split(" ")))

positive_data = features_spam.map(lambda features: LabeledPoint(1, features))
negative_data = features_ham.map(lambda features: LabeledPoint(0, features))

data = positive_data.union(negative_data)
data.cache()
(training, test) = data.randomSplit([.6, .4])

models = [classification.NaiveBayes.train(training), classification.LogisticRegressionWithSGD.train(training),
          classification.LogisticRegressionWithLBFGS.train(training), classification.SVMWithSGD.train(training)]

for x in models:

    predictionLabel = test.map(lambda y: (x.predict(y.features), y.label))

    testErr = predictionLabel.filter(lambda (v, p): v == p).count() / float(training.count())

    print "Accuracy for " + str(x.__class__) + " is " + str(testErr)

