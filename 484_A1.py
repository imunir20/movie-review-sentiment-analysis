from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


letters = re.compile('[^a-zA-Z\'\- ]+')
dashes = re.compile('((?<=[\s\.])\-+)|(\-+(?=[\s\.]))')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
vectorizer = TfidfVectorizer(use_idf = True)

labels = []
reviews = []
testReviews = []

outputFileName = '484_A1_Attempt12.txt'
currentK = 123

# Clean training data
# Removes HTML tags, punctuation, compound adjectives, stems words
# Cleaned training data are stored reviewws list, labels stored in labels list
def cleanTrainingData():
    i = 0
    fileName = '484_A1Training.txt'
    with open(fileName, encoding = 'utf-8') as dataset:
        next(dataset)
        for line in dataset:                # Iterate through each line in the text file
            splitLine = line.split('\t')      # Split the labels and reviews apart
            splitLine[0] = int(splitLine[0])   # Convert the labels into integers
            labels.append(splitLine[0])
            splitLine[1] = BeautifulSoup(splitLine[1], 'lxml').text     # Remove HTML tags from reviews
            splitLine[1] = letters.sub(' ', splitLine[1])
            splitLine[1] = dashes.sub(' ', splitLine[1])
            splitLine[1] = splitLine[1].lower()

            allWords = word_tokenize(splitLine[1])
            cleanWords = []
        
            for w in allWords:
                if w not in stop_words and w.isalpha():
                    cleanWords.append(stemmer.stem(w))
            reviews.append(' '.join(cleanWords))

            #print(reviews[i])       # Print out clean reviews once cleaned
            i = i + 1

# Clean testing data
# Removes HTML tags, punctuation, compound adjectives, stems words
# Cleaned test data stored in tesstReviews list
def cleanTestData():
    i = 0
    fileName = '484_A1TestData.txt'
    with open(fileName, encoding = 'utf-8') as testset:
        for line in testset:                # Iterate through each line in the text file
            currLine = line
            currLine = BeautifulSoup(currLine, 'lxml').text     # Remove HTML tags from reviews
            currLine = letters.sub(' ', currLine)
            currLine = dashes.sub(' ', currLine)
            currLine = currLine.lower()
            
            allWords = word_tokenize(currLine)
            cleanWords = []

            for w in allWords:
                if w not in stop_words and w.isalpha():
                    cleanWords.append(stemmer.stem(w))
            testReviews.append(' '.join(cleanWords))

            print(testReviews[i])       # Print out rest reviewws once cleaned
            i = i + 1




# Fits TFIDF Vectorizer to training data and transforms training data into vectors
# Transforms testing data to vectors according to vectorizer fitted by training data
def vectorizeData(data, set):
    
    if set == 2:        # If set = 2, then this is the training set
        vectors = vectorizer.fit_transform(data)
    else:               # Else, this is not the training set, just transform
        vectors = vectorizer.transform(data)
    print(vectors.shape)
    return vectors
    

def findKNearestNeighbors(v1, v2, k):
    # Given 2 vectors in the form as 2D ndarrays
    # v1 should contain a vector representing a review from the validation or test data
    # v2 should contain all vectors representing the reviews from training data
    # create a list to store distances/labels of current k nearest neighbors called currNeighbors
    currNeighbors = []

    # for first k entries in v2, compute their distances, store their distances/labels as a tuple in currNeighbors
    # sort the currNeighbors list in ascending order or descending order depending on distance
    
    #distances = cosine_similarity(v1, v2)
    distances = euclidean_distances(v1, v2)
    
    # argsort sorts the indices, not the values in distances
    #highestDist = np.argsort(-1*distances)[:k]
    lowestDist = np.argsort(distances)[:k]

    # Get k nearest neighbors
    for i in range(k):
        #kIndex = highestDist[0][i]
        kIndex = lowestDist[0][i]
        currNeighbors.append((distances[0][kIndex], labels[kIndex]))    

    # Vote on label using k nearest neigbors (using majority or weighted distances)
    newLabel = getVotes(currNeighbors, k)
    return newLabel



# 2 APPROACHES for voting: MAJORITY RULES, and WEIGHTED DISTANCES

def getVotes(neighbors, k):

    # Comment first line out to use weighted, or just keep first line and return statement to use majority
    #choice = majorityVote(neighbors, k)
    choice = weightedDistanceVote(neighbors, k)
    if choice == 0:
        choice = majorityVote(neighbors, k)
    return choice



# MAJORITY RULES
# out of those k vectors, count how many are positive reviews, count how many are negative
# new test data's label should be majority class

def majorityVote(neighbors, k):
    positives = 0
    negatives = 0
    for m in range(k):       # For the k neighbors
        currentPoint = neighbors[m]     # Tally up the label votes
        if currentPoint[1] == -1:      
            negatives = negatives + 1
        else:
            positives = positives + 1
    if positives > negatives:         # If more positive labels, return positive
        return 1
    elif negatives > positives:      # If more negative labels, return negative
        return -1
    return 0                     # Else, must be a tie, return 0

# WEIGHTED DISTANCES
# add distances for each class by taking vectors and take their summation of 1/distance
# class with highest sum will be new test data's label

def weightedDistanceVote(neighbors, k):
    positiveSum = 0.0
    negativeSum = 0.0
    weightedDistance = 0.0
    for v in range(k):       # For k neighbors,
        currentPoint = neighbors[v]          

        # You can either 1/d as the weight applied to the distance, or 1/d^2
        #weightedDistance = 1/currentPoint[0]     # Get their weighted distance (1/distance), add it to label sum
        weightedDistance = 1/(currentPoint[0] * currentPoint[0])   # 1/distance^2
        if currentPoint[1] == -1:
            negativeSum = negativeSum + weightedDistance
        else:
            positiveSum = positiveSum + weightedDistance
    if positiveSum > negativeSum:              # Whichever sum is bigger means that those points were closer to new test data
        return 1
    elif negativeSum > positiveSum:
        return -1
    return 0                                # If sums were equal, label distances were equally close, return 0 for a tie
        
def makePredictions(v1, v2, k, fName):
    # v1 is from unseen data
    # v2 is from training data
    # Compute KNN with argument k for that vector in v1 with all vectors in v2
    # Store result of KNN in chosenLabel
    # Write chosenLabel to file with name fName
    writeFile = open(fName, 'w')
    for a in range(v1.shape[0] - 1):
        chosenLabel = findKNearestNeighbors(v1[a], v2, k)
        writeFile.write(str(chosenLabel) + '\n')
        print(str(a))
    # When last label written, there should be no newline, so separate this last case from the for loop
    lastLabel = findKNearestNeighbors(v1[v1.shape[0] - 1], v2, k)
    writeFile.write(str(lastLabel))
    writeFile.close()    

# K-Fold Cross validation with K = 5
# Folds are sizes of 3000 except for fold 5 (1499)
# Folds are first 3000 data in training, second 3000, etc.
# After accuracies taken from all 5 runs, the accuracies are averaged for the final result
def crossValidate():

    total = 0

    f1 = reviews[:3000]
    f2 = reviews[3000:6000]
    f3 = reviews[6000:9000]
    f4 = reviews[9000:12000]
    f5 = reviews[12000:]
    
    currTrain = f1 + f2 + f3 + f4
    trainV = vectorizeData(currTrain, 2)
    testV = vectorizeData(f5, 1)

    total = performValidation(testV, trainV, currentK, 12000, 14999, total)

    currTrain = f1 + f2 + f3 + f5
    trainV = vectorizeData(currTrain, 2)
    testV = vectorizeData(f4, 1)

    total = performValidation(testV, trainV, currentK, 9000, 12000, total)

    currTrain = f1 + f2 + f4 + f5
    trainV = vectorizeData(currTrain, 2)
    testV = vectorizeData(f3, 1)

    total = performValidation(testV, trainV, currentK, 6000, 9000, total)

    currTrain = f1 + f3 + f4 + f5
    trainV = vectorizeData(currTrain, 2)
    testV = vectorizeData(f2, 1)

    total = performValidation(testV, trainV, currentK, 3000, 6000, total)

    currTrain = f2 + f3 + f4 + f5
    trainV = vectorizeData(currTrain, 2)
    testV = vectorizeData(f1, 1)

    total = performValidation(testV, trainV, currentK, 0, 3000, total)

    print('AVERAGED ACCURACY: ' + str(total/5))    

# Actually finds the accuracy for each run done in k-fold cross validation where k = 5
def performValidation(v1, v2, k, start, end, currentScore):
    score = 0
    predictedLabel = 0
    w = 0

    for c in range(start, end):
        predictedLabel = findKNearestNeighbors(v1[w], v2, k)
        if predictedLabel == labels[c]:
            score = score + 1
        w = w + 1
    if end == 1499:
        score = score/2999
    else:
        score = score/3000
    print('Accuracy was: ' + str(score) + '\n')
    totalScore = currentScore + score
    return totalScore


# ----------------------- DRIVER/MAIN CODE ---------------------------------

cleanTrainingData()
v2 = vectorizeData(reviews, 2)
# print('\n\n\n\n\n\n')
cleanTestData()
v1 = vectorizeData(testReviews, 1)
#currentK = 123
#crossValidate()
currentK = 123
makePredictions(v1, v2, currentK, outputFileName)

