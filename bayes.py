import random
import csv
import math


def loadCSV(filename):
    lines = csv.reader(open(filename,"rt"))
    dataset = list(lines)

    #for i in range(len(dataset)):
        #dataset[i].pop(0)
        #dataset[i] = [float(x) for x in dataset[i]]

    return dataset

def splitDataset(dataset, splitratio):
    training = int(len(dataset)*splitratio)
    train = []
    copy = list(dataset)
    while len(train) < training:
        index = random.randrange(len(copy))
        train.append(copy.pop(index))

    return [train,copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if(vector[0] not in separated):
            separated[vector[0]]=[]
        separated[vector[0]].append(vector)

    return separated

def mean(numbers):
    #print(numbers)
    return sum(numbers)/float(len(numbers))

def stddev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers))
    return math.sqrt(variance)

def calculateProbability(x,mean,stddev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stddev,2))))

    return (1/(math.sqrt(2*math.pi)*stddev))*exponent


def summarize(dataset):
    #print(dataset)
    #print("AFTER")
    for i in range(len(dataset)):
        dataset[i].pop(0)
        dataset[i] = [float(x) for x in dataset[i]]
    #print(dataset)
    #print("XXXXXXXXXXXXXXXXX")
    summaries = [(mean(attribute),stddev(attribute)) for attribute in zip(*dataset)]
    #print(attribute for attribute in zip(*dataset))
    #print("XXXXXXXXXXXXXXXXXXXXXXXXX")
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    #print(separated)
    summaries = {}
    for classVal, instance in separated.items():

        summaries[classVal] = summarize(instance)

    return summaries

def calculateClassProbability(summaries,inputVector):
    probabilities = {}
    #print(inputVector)
    for classVal, classSum in summaries.items():
        #print(classVal)
        #print(classSum)
        probabilities[classVal] = 1
        #print(classSum)
        for i in range(len(classSum)):
            mean, stddev = classSum[i]
            x = float(inputVector[i+1])
            #print("mean: "+ str(mean))
            #print("stddev: "+ str(stddev))
            #print("value: "+ str(x))
            probabilities[classVal]*=calculateProbability(x,mean,stddev)
    #print(probabilities)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbability(summaries,inputVector)
    bestLabel, bestProb = None,-1
    for classVal, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classVal
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries,testSet[i])
        predictions.append(result)

    return predictions

def getAccuracy(testSet,prediction):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][0] == prediction[i]:
            correct+=1

    return (correct/float(len(testSet)))*100.0

def main():
    filename = "abalone.csv"
    splitRatio = 0.7
    dataset=loadCSV(filename)

    trainingSet, testSet = splitDataset(dataset, splitRatio)


    summaries = summarizeByClass(trainingSet)

    #print(testSet)
    predictions = getPredictions(summaries,testSet)
    #print(predictions)
    accuracy = getAccuracy(testSet,predictions)

    print (accuracy)


main()