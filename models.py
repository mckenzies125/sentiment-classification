# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """


class UnigramFeatureExtractor(FeatureExtractor):

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter() #initialize a counter for the words, Counter() returns a dictionary
        for wd in sentence:
            wd = wd.lower() #make all words lowercase
            # TODO: get the index (add_and_get_index vs index_of depending on the flag)
            # TA's version -> i = indexer.add_and_get_index(wd) if add_to_indexer else indexer.index_of(wd)
            if add_to_indexer: #boolean value that we pass in...True when processing training data and False when testing/validating
                i = self.indexer.add_and_get_index(wd) #add & get index of wd. We will eventually use this create our bag of words
                #we assume add_and_get_index checks if wd is already in Indexer/vocab, and if it isn't, add it. Either way, get index
                #
                #Just understand that add_and_get_index has its own way of knowing what index is associated with each wd in words
            else:
                i = self.indexer.index_of(wd) #aka we're testing or validating...get index of wd.     -> will assign -1 if index cannot be found
                #above line searches Indexer/vocab for wd's index
            if i == -1: # i == -1 add_to_indexer is False, and then the word's index couldn't be found (aka the word wasn't in any of the training data)
                continue #skip unknown words and restart loop with next wd
            feats[i] += 1 #count how many times a word of this index appears in the data 
                # TODO: if the index is -1, skip; otherwise feats[index] += 1
        return feats


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter() #initialize a counter for the words, Counter() returns a dictionary
        for wds in range(len(sentence) - 1):
            first_word = sentence[wds].lower()
            second_word = sentence[wds + 1].lower()
            bigram = first_word + " " + second_word
            # TODO: get the index (add_and_get_index vs index_of depending on the flag)
            # TA's version -> i = indexer.add_and_get_index(wd) if add_to_indexer else indexer.index_of(wd)
            if add_to_indexer: #boolean value that we pass in...True when processing training data and False when testing/validating
                i = self.indexer.add_and_get_index(bigram) #add & get index of wd. We will eventually use this create our bag of words
                #we assume add_and_get_index checks if wd is already in Indexer/vocab, and if it isn't, add it. Either way, get index
                #
                #Just understand that add_and_get_index has its own way of knowing what index is associated with each wd in words
            else:
                i = self.indexer.index_of(bigram) #aka we're testing or validating...get index of wd.     -> will assign -1 if index cannot be found
                #above line searches Indexer/vocab for wd's index
            if i == -1: # i == -1 add_to_indexer is False, and then the word's index couldn't be found (aka the word wasn't in any of the training data)
                continue #skip unknown words and restart loop with next wd
            feats[i] += 1 #count how many times a word of this index appears in the data 
                # if the index is -1, skip; otherwise feats[index] += 1
        return feats


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    #def __init__(self):
    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def dot_sparse(self, feats): #goal - use feats dict instead of dense storage scheme 
        # (C) never build x — only touch the entries that are nonzero
        # sum w[i] * c over feats.items()
        total = 0 
        for i, c in feats.items(): #same as above, just we extract values directly from feats
            total += self.weights[i] * c 
        return total

    def predict(self, sentence_of_words: List[str]) -> int:
        extracted_feats = self.feat_extractor.extract_features(sentence_of_words, add_to_indexer = False)
        #feat_extractor is in train_linear_model
        
        score = self.dot_sparse(extracted_feats)
        if score > 0:
            return 1
        else:
            return 0
        

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    random.seed(123456)

    # build vocab ONCE
    for ex in train_exs:
        feat_extractor.extract_features(
            ex.words,
            add_to_indexer = True
        )

    weights = np.zeros(len(feat_extractor.indexer))
    learning_rate = 0.025

    lrs = LogisticRegressionClassifier(weights, feat_extractor)

    for epoch in range(50):

        shuffled_exs = train_exs.copy()
        random.shuffle(shuffled_exs)

        for ex in shuffled_exs:

            feats = feat_extractor.extract_features(
                ex.words,
                add_to_indexer = False
            )

            score = lrs.dot_sparse(feats)

            y_hat = 1 / (1 + np.exp(-score))

            for i, c in feats.items():
                grad = (y_hat - ex.label) * c
                weights[i] -= learning_rate * grad

    return LogisticRegressionClassifier(weights, feat_extractor)



def lr_schedule_exploration(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], feat_extractor: FeatureExtractor):
    """
    Compare different learning-rate schedules for logistic regression.
    Plots training log likelihood and development accuracy over 50 epochs.
    """

    import matplotlib.pyplot as plt

    # Build vocabulary
    for ex in train_exs:
        feat_extractor.extract_features(
            ex.words,
            add_to_indexer=True
        )

    initial_lr = 0.025

    schedules = ["constant", "factor_decay", "1/t"]

    all_log_likelihoods = {}
    all_dev_accuracies = {}

    for schedule in schedules:

        # Each schedule starts from the same zero weights
        weights = np.zeros(len(feat_extractor.indexer))
        lrs = LogisticRegressionClassifier(weights, feat_extractor)

        train_log_likelihoods = []
        dev_accuracies = []

        # Same shuffle order for each schedule
        random.seed(123456)

        for epoch in range(50):

            # Select the learning-rate schedule
            if schedule == "constant":
                learning_rate = initial_lr

            elif schedule == "factor_decay":
                learning_rate = initial_lr * (0.95 ** epoch)

            elif schedule == "1/t":
                learning_rate = initial_lr / (epoch + 1)

            # Shuffle training examples
            shuffled_exs = train_exs.copy()
            random.shuffle(shuffled_exs)

            # SGD
            for ex in shuffled_exs:

                feats = feat_extractor.extract_features(
                    ex.words,
                    add_to_indexer=False
                )

                score = lrs.dot_sparse(feats)

                y_hat = 1 / (1 + np.exp(-score))

                for i, c in feats.items():
                    grad = (y_hat - ex.label) * c
                    weights[i] -= learning_rate * grad



            log_likelihood = 0

            for ex in train_exs:

                feats = feat_extractor.extract_features(
                    ex.words,
                    add_to_indexer=False
                )

                score = lrs.dot_sparse(feats)

                log_likelihood += (
                    ex.label * score
                    - np.logaddexp(0, score)
                )

            train_log_likelihoods.append(log_likelihood)


            num_correct = 0

            for ex in dev_exs:

                pred = lrs.predict(ex.words)

                if pred == ex.label:
                    num_correct += 1

            dev_accuracy = num_correct / len(dev_exs)
            dev_accuracies.append(dev_accuracy)

        # Save all 50 values for this schedule
        all_log_likelihoods[schedule] = train_log_likelihoods
        all_dev_accuracies[schedule] = dev_accuracies


    epochs = range(1, 51)

    for schedule in schedules:
        plt.plot(
            epochs,
            all_log_likelihoods[schedule],
            label=schedule
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training Log Likelihood")
    plt.title("Training Log Likelihood by Step Size Schedule")
    plt.legend()
    plt.show()


    for schedule in schedules:
        plt.plot(
            epochs,
            all_dev_accuracies[schedule],
            label=schedule
        )

    plt.xlabel("Epoch")
    plt.ylabel("Development Accuracy")
    plt.title("Development Accuracy by Step Size Schedule")
    plt.legend()
    plt.show()


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """

    # Constructor for the instance of NeuralSentimentClassifier(). 
    def __init__(self, network, word_embeddings):
        self.network = network # store network as attribute of this particular instance
        self.word_embeddings = word_embeddings # store word_embeddings as attribute of this particular instance
        self.word_indexer = word_embeddings.word_indexer # use word_indexer that is already associated with the word_embeddings' word_indexer. 
        

    # Question: What is GloVe?
    # Answer: A pretrained dictionary that maps words to numerical vectors representing their meanings

    # Purpose of func: takes a list of sentences and maps their words, using GloVe, to indices. We store those indices in a list.
    # This is a helper func
    def wordsToIndices(self, sentence: List[str]):
        indices = [] # initialize a holder for the indices

        unk_index = self.word_indexer.index_of("UNK") # UNK = "Unknown". We retrieve this now so if a word cannot be found, then we mark it with UNK's index

        for wd in sentence: #This is all the same as in Unigram/Bigram
            wd = wd.lower()
            i = self.word_indexer.index_of(wd) 

            if i == -1:             # If GloVe doesn't contain this word, use UNK
                i = unk_index

            indices.append(i) #save the index

        return torch.tensor(indices, dtype = torch.long) # construct a PyTorch tensor using "indices".
        # dtype=torch.long means the values in the tensor are stored as 64-bit integers
        # This is required because these values will be used as indices by the embedding layer.


    # Purpose of Func: 
    def predict(self, sentence: List[str]) -> int:

        indices = self.wordsToIndices(sentence) # Convert sentence to GloVe indices (see func directly above)

        # we are predicting, not training
        with torch.no_grad(): # Normally, when PyTorch runs data through a neural network, it keeps track of all the mathematical operations it performs so it can eventually calculate gradients and such. However, we're not training (only predicting) so we don't care ab the operations. 
            log_probs = self.network(indices) # This passes tensor of word indices into DAN network

        # Return whichever class has the larger probability
        prediction = torch.argmax(log_probs).item() # Return whichever class has the larger probability
        # torch.argmax(log_probs) -> find the index of the larger of the probability classes. 
        # ex: take --> log_probs = tensor([-2.4, -0.095])
        # so torch.argmax(log_probs) returns tensor(1)
        # item() converts the tensor into an integer

        return prediction # either 0 or 1


# This is the Deep Averging Network architecture
class DAN(nn.Module): # DAN is a subclass of nn.Module. nn.Module is parent class and DAN is child class here!
    def __init__(self, word_embeddings, hid, out): # Constructor
        super(DAN, self).__init__() # Run the constructor of DAN's parent class, nn.Module, on this object

        # Create the look-up table and call it self.embeddings
        # "word_embeddings.get_initialized_embedding_layer(frozen = True)" takes the GloVe matrix and creates an nn.Embedding object from it. Look at get_initialized_embedding_layer() to see how.
        self.embedding = word_embeddings.get_initialized_embedding_layer(frozen = True) 
        
        inp = word_embeddings.get_embedding_length() # Grab size of input embeddings. How many numbers are in each GloVe vector?
        

        #Taken from ffnn_example.py:
        
        # Neural network layers
        self.V = nn.Linear(inp, hid) # creates a fully connected linear layer that takes inp numbers as input and produces hid numbers as output
        # self.g = nn.Tanh() # creates activation layer with Tanh func
        self.g = nn.ReLU() # creates activation layer with ReLU func
        self.W = nn.Linear(hid, out) # creates layer that takes hid numbers as input and produces out numbers as output
        self.log_softmax = nn.LogSoftmax(dim = 0) # map to probability of each class using softmax func
        
        # Initialize weights according to a formula due to Xavier Glorot
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight) #initialize weights using Xavier uniform initialization (sets neural network weights by drawing random values from a uniform distribution between -limit and +limit)

        # Initialize with zeros instead
        # nn.init.zeros_(self.V.weight)
        # nn.init.zeros_(self.W.weight)

        
    
    def forward(self, indices):

        max_length = 0 
        for sentence in indices:
            if len(sentence) > max_length:
                max_length = len(sentence)
                
        for sentence in indices:
            num_padding = max_length - len(sentence)

            for i in range(num_padding):
                sentence.append(0)

        # Use the look-up table to embed the words
        embedded_words = self.embedding(sentence) # Remember: self.embedding is an nn.Embedding object, so we can give it indices
        # So, we're using self.embeddings, which is essentially a look-up table, to turn indice IDs into embedded vectors

        averaged_embedding = torch.mean(embedded_words, dim = 0) # Each input sentence will become one averaged GloVe vector. This is 300 with a 300-dimensional GloVe embeddings

        # Example:
        # "this"  → [ 2,  4,  3]
        # "movie" → [ 4,  2,  5]
        # "is"    → [ 1,  3,  2]
        # "great" → [ 5,  3,  6]

        #Cols are the dimensions of the tensor. Take avg of each col to achieve --> [3, 3, 4]

        # Feed the sentence vector through the neural network
        hidden = self.V(averaged_embedding) 
        hidden = self.g(hidden)
        output = self.W(hidden)
        log_probs = self.log_softmax(output)

        return log_probs



def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    random.seed(123456)
    torch.manual_seed(123456)

    hid = 100 # Number of neurons in hidden layer

    out = 2 # Two output classes: negative (0) and positive (1)

    num_epochs = 25
    learning_rate = 0.001

    network = DAN(word_embeddings, hid, out) # Create the DAN

    optimizer = optim.Adam(network.parameters(), lr = learning_rate) # Adam will update all trainable parameters in the DAN

    # The below code seems repetitive. Yes, this is a line-by-line copy of the wordsToIndices() method in the NeuralSentimentClassifier() class. A NeuralSentimentClassifier object has yet to be instantiated, so we do not yet have an object through which to call that method. Thus, we copy and paste the method below 
    
    # We need the GloVe indexer to turn words into indices
    word_indexer = word_embeddings.word_indexer
    unk_index = word_indexer.index_of("UNK")

    for epoch in range(num_epochs):

        # Shuffle the order of training examples each epoch
        ex_indices = list(range(len(train_exs)))
        random.shuffle(ex_indices)

        total_loss = 0.0 # Initialize loss param

        for idx in ex_indices: 

            ex = train_exs[idx] # get a training sample from train_exs at position idx

            indices = []

            for wd in ex.words:
                word_idx = word_indexer.index_of(wd.lower()) # lowercase the word and get that word's index

                if word_idx == -1:
                    word_idx = unk_index # if not in bag-of-words, label as UNK or unknown

                indices.append(word_idx)

            indices = torch.tensor(indices, dtype = torch.long)


            log_probs = network(indices) 

            y = ex.label #extract the example's label

            y_onehot = torch.zeros(out) # Create an empty one-hot vector --> y_onehot = tensor([0., 0.])
            y_onehot.scatter_(0, torch.tensor(y, dtype = torch.long), 1) # Slide a 1 into the either the 0th or 1st index to indicate the correct class (if negative class, 1 in 0th position...if pos class, 1 in 1st position)
            loss = torch.neg(log_probs).dot(y_onehot) 
            # neg(log_probs) takes the neg log likelihoods and makes them pos
            # taking the dot product of this vector and the one_hot vector gives us the minimized likelihood:

            # log_probs = tensor([-2.4, -0.095])
            # y_onehot = tensor([0., 1.])
            # (2.4)(0)+(0.095)(1) = 0.095

            optimizer.zero_grad() # clear out gradient

            loss.backward() # back propogation step
            optimizer.step() # use optimizer Adam
            total_loss += loss.item() # update loss

        #print("Total loss on epoch:" + (epoch, total_loss))

    # Wrap the trained DAN
    model = NeuralSentimentClassifier(network, word_embeddings)

    return model

    