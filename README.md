# Movie Review Sentiment Analysis

How to run code:
You can run the code the same way you run any python program. You can use the typical "py 484_A1.py" to run the program
or run it from your IDE, which is what I did.

In this program, KNN was implemented from scratch to gain a better understanding of the algorithm (however, things like euclidean/cosine distance were used from libraries, did not make much sense to implement that from scratch)

Highest accuracy achieved on test data: 0.81

k-NN Implementation:
The k-NN implementation is located 484_A1.py

The 'findKNearestNeighbors(v1, v2, k)' function is where the KNN algorithm is actually implemented. This function is called by the
makePredictions function to makePredictions on the test data. The findKNearestNeighbors function also calls a function that I wrote
called getVotes, which can call two other functions called weightedDistanceVote and majorityVote. The getVotes function is used to
tally up the labels of the k nearest neighbors using either the weighted distance voting method or the majority voting method.
