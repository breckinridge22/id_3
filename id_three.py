import sys
import math
import pandas as pd
import numpy as np
import random
from collections import Counter

'''
Assignment 1 for Breck Stodghill and Eddie Goode

'''




'''
- Node for the tree structure

'''
class Node:
    def __init__(self, val):
        self.value = val
        self.children = {}

    def add_child(self, key, child):
        self.children[key] = child


'''
Get Index
- This method accepts a column name, and a list of attributes, and returns the
  columns index.


'''
def get_index(classifier, attributes):
    index = 0
    for x in attributes:
        if x == classifier:
            return index
        index = index + 1
    return -1

''' Print Tree
- This method accepts a tree node and a parameter for the number of idents
- It recurseively prints the data structure that holds the decision tree3 and
  layers the different levels of the tree using tabs.

'''
def printTree(tree, indents):
    print(indents*"\t", tree.value)
    for child in tree.children:
        child_indent = indents+1
        print(child_indent*"\t", child)
        printTree(tree.children[child], child_indent+1)

''' Calculate error
- Accepts data, a list of attributes, a classifier, and a complete decisions tree
- Returns the error percentage of the decision tree, given some data.
'''
def calculate_error(data, attributes, classifier, tree):
    target_i = get_index(classifier, attributes)
    predictors = get_values(data, target_i)

    total = 0
    wrong = 0
    for row in data:
        total += 1
        #print(row)
        diction = predict(row, attributes, tree, predictors)
        if diction != row[target_i]:
            wrong += 1
    if total is 0:
        return 0
    else:
        error = wrong/total
        return error *100.00

''' Find Nodes
- Accepts a tree, attributes, a list of nodes, and a list of predictor values
- Returns a list of decision nodes in the tree at least one level above a leaf node
  ready to be checked for pruning.

'''
def find_nodes(tree, attributes, nodes, predictors):
    if tree.value in attributes and all(tree.children[child].value in predictors for child in tree.children):
            #print("Node was found", tree.value)
            nodes.append(tree.value)

    for child in tree.children:
        find_nodes(tree.children[child], attributes, nodes, predictors)
    return nodes

def count_nodes(tree, attributes, nodes):
    if tree.value in attributes and tree.value not in nodes:
            nodes.append(tree.value)

    for child in tree.children:
        count_nodes(tree.children[child], attributes, nodes)
    return nodes


'''
- This is the main method of the program. It is split between Part 2 and Part 3.

- Part 2
- Reads in a dataset, and randomly splits the data into a training set and testing
  set using python random number generators. The training set makes up 80% of the
  original set, and the testing set makes up 20%. Then the ID3 algorithm is given the training set
  to construct a tree. It's training error is identified and printed. Then the tree is tested on the
  test set. Its testing error is identified and printed.

- Part 3
- Reads in a data set. Splits it into trainging-validation-testing. Runs the ID3 algorithm on the
  training set for each possible iteration of features. Then prunes the tree and calculates, the Training,
  Validation, and Testing error for each iteration.





'''
def main(argv):
    # lets assume the last attribute is the classifier
    wb = pd.read_excel('tennis_set.xls')
    #classifier = "PlayTennis"
    classifier = "PlayTennis"
    attributes = wb.columns.values
    data = wb.values

    print("\n\n\n***************** Part 1 *********************\n\n\n")

    for x in range(1,len(attributes)):
        tree = id_three(data, classifier, attributes, [], x)
        nodes = count_nodes(tree, attributes, [])
        if x > len(nodes):
            break
        train_error = calculate_error(data, attributes, classifier, tree)
        print("Train Error: after ", x, "feautres = ", train_error)


    printTree(tree, 0)

    train_error = calculate_error(data, attributes, classifier, tree)
    print("Train Error:", train_error)


    ######################## Part 2 ###############################
    # Split into train and test sets
    #
    #
    #
    #
    #############################################################
    print("\n\n\n***************** Part 2 *********************\n\n\n")

    wb_training = pd.read_excel("votes_training.xls")
    wb_testing = pd.read_excel("votes_testing.xls")

    #classifier = "PlayTennis"
    classifier = "Party"
    attributes = wb_training.columns.values

    data_training = wb_training.values
    data_testing = wb_testing.values

    for x in range(1, len(attributes)-1):
        tree = id_three(data_training, classifier, attributes, [], x)
        nodes = count_nodes(tree, attributes, [])
        #print(nodes)
        if x > len(nodes):
            break
        train_error = calculate_error(data_training, attributes, classifier, tree)
        print("Train Error: after", x, "features", train_error)
        #printTree(tree, 0)

    print("\n")

    for x in range(1, len(attributes)):
        tree = id_three(data_training, classifier, attributes, [], x)
        test_error = calculate_error(data_testing, attributes, classifier, tree)
        print("Test Error: after", x, "features", test_error)

    print("\n")
    print("Final Train Error:", train_error)

    test_error = calculate_error(data_testing, attributes, classifier, tree)
    print("Final Test Error:", test_error)

    ######################### Part 3 #############################

    # train 60%
    # validation 20%
    # test 20%

    ##############################################################
    print("\n\n\n***************** Part 3 *********************\n\n\n")
    train_size = math.floor(.8*len(data_training))
    validation_size = len(data_training) - train_size

    train = []
    validation = []

    train_random_num = random.sample(range(1,len(data_training)), math.floor(.8*len(data_training)))
    validation_random_num = []

    for i in range(1, len(data_training)):
         if i not in train_random_num:
             validation_random_num.append(i)

    # Create Random Training Subset
    for item in train_random_num:
         train.append(data_training[item])
    train = np.array(train)

    # Create Random Validation Subset
    for item in validation_random_num:
        validation.append(data_training[item])
    validation = np.array(validation)

    # Construct the tree
    for x in range(1, len(attributes)):
        tree3 = id_three(train, classifier, attributes, [], x)
        train_error3 = calculate_error(train, attributes, classifier, tree3)
        print("Train error after", x, "features", train_error3)
        #printTree(tree3, 0)

        target_i = get_index(classifier, attributes)
        predictors = get_values(data_training, target_i)

        # Find the decision nodes for pruning
        nodes = find_nodes(tree3, attributes, [], predictors)

        # Calculate Training Error
        #train_error3 = calculate_error(data_training, attributes, classifier, tree3)
        #print("Final Train error", train_error3)

        # Calculate Base Validation Error
        og_error = calculate_error(validation, attributes, classifier, tree3)
        #print("\nOG Validation error:", og_error)

        # For each decision node, prune and check with original error rates
        # If the pruned error rates is no worse, then replace the tree with the pruned tree
        for n in nodes:
            pruned_tree = copy(tree3)
            index = get_index(n, attributes)
            target_i = get_index(classifier, attributes)
            new_tree = prune(train, attributes, pruned_tree, n, index, target_i)
            #print("Pruned on", n)
            #printTree(new_tree, 0)
            pruned_error = calculate_error(validation, attributes, classifier, new_tree)
            #print("pruned validation error:", pruned_error)

            if pruned_error <= og_error:
                tree3 = new_tree
                og_error = pruned_error

        print("Validation error after", x, "features:", og_error)

        test_error = calculate_error(data_testing, attributes, classifier, tree3)
        print("Test error after", x, "features", test_error)
        print("\n")



''' Copy
- Makes a deep copy of a tree.

'''
def copy (tree):
    node = Node(tree.value)
    if len(tree.children) is 0:
        return node
    else:
        for child in tree.children:
            toAdd = copy(tree.children[child])
            node.add_child(child, toAdd)
        return node

''' Prune
- Give a feature to prune on, returns a pruned tree

'''
def prune (data, attributes, tree, n, index, target_i):
    if tree.value == n:
        # remove the children
        tree.children = {}
        #for val in get_values(data, index):
            #new_set = [row[target_i] for row in data if row[index] == val]
        #print(get_labels(data, target_i))
        count = Counter(get_labels(data, target_i))
        majority = count.most_common()
        #print("Majority", majority)
            #print(majority[0][0])
        tree.value = majority[0][0]
        return tree
    else:
        for child in tree.children:
            # get index of current value
            index = get_index(tree.value, attributes)
            new_data = [row for row in data if row[index] == child]
            prune(new_data, attributes, tree.children[child], n, index, target_i)
        return tree


''' Predict
- Given some row of data, a list of attributes, a tree, and a list of classifiers
- Returns a prediction by traversing the tree

'''
def predict(row, attributes, tree, predictors):
    if tree.value in predictors:
        return tree.value
    else:
        feature = tree.value
        feature_i = get_index(feature, attributes)
        for val in tree.children.keys():
            if row[feature_i] == val:
                return predict(row, attributes, tree.children[val], predictors)


''' Find Most Freq
- Given a singular column of data, returns the most frequent value

'''
def find_most_freq(data):
    value_freq = {}
    for x in range(len(data)):
        val = data[x][0]
        if val in value_freq:
            value_freq[val] += 1
        else:
            value_freq[val] = 1
    sorted_val_freq = sorted(value_freq, key = value_freq.get, reverse = True)
    return sorted_val_freq[:1]


''' Get Values
- Given some matrix of data, returns a list of possible values in a column index

'''
def get_values(data, index):
    values = []
    for x in range(len(data)):
        val = data[x][index]
        if val not in values:
            values.append(val)
    return values

''' Get Labels
- Given a matrix of data and an idex, returns a list of each value in the column
'''
def get_labels(data, index):
    labels = []
    for x in (data):
        labels.append(x[index])
    return labels

''' Find Best
- Given some data, a list of attributes, a classifier, and a list of already
  classified features, returns the best feature that maximizes information maxGain
  on the dataset.
'''
def find_best(data, attributes, classifier, classified):
    maxGain = -1.0
    best = ""
    #print("attributes:", attributes)
    #print("classified:", classified)
    for attribute in attributes:
        if attribute != classifier and attribute not in classified:
            #print("why")
            gain = info_gain(data, attributes, classifier, attribute)
            #print(attribute, ":", gain)
            if gain > maxGain:
                best = attribute
                maxGain = gain
    #print(best, ":", maxGain)
    return best

''' Info Gain


'''
def info_gain(data, attributes, classifier, attribute):

    target_i = get_index(classifier, attributes)
    i = get_index(attribute, attributes)

    attr_labels = get_labels(data, i)
    target_labels = get_labels(data, target_i)

    attr_freq = find_frequency(attr_labels)

    base_tropy = entropy(target_labels)
    subset_tropy = 0.0

    for val in attr_freq.keys():
        prob = attr_freq[val]/len(attr_labels)
        # get the target attr val for each row whos attr is equal to the val in the dicitonary
        subset = [row[target_i] for row in data if row[i] == val]
        subset_tropy += prob * entropy(subset)
    gain = base_tropy - subset_tropy
    return gain

''' Find frequency
- Given a list of labels, returns a dictionary of values and their frequency


'''
def find_frequency(labels):
    freq = {}
    for entry in labels:
        if entry in freq:
            freq[entry] += 1
        else:
            freq[entry] = 1
    return freq


# accepts a list of labels and returns the entropy of the label classification
def entropy(labels):
    tropy = 0.0

    freq = find_frequency(labels)
    total = len(labels)

    if len(freq) is 0:
        return tropy
    else:
        for data in freq.keys():
            prob = freq[data]/total
            tropy +=  (-prob * math.log(prob, 2))
        return tropy



# the whole ID3 algorithm
def id_three(data, classifier, attributes, classified, count):
    index = get_index(classifier, attributes)
    labels = []
    for x in data:
        labels.append(x[index])

    if len(data) is 0:
       raise ValueError('There is no data, bruh')
    elif len(attributes)-1 == len(classified) or count is 0:
        freq = find_frequency(labels)
        most = 0
        toRet = ""
        for key in freq.keys():
            if freq[key] > most:
                most = freq[key]
                toRet = key
        return Node(toRet)
    # if the list of labels are all the same, return a node with the label
    elif all(x == labels[0] for x in labels):
        return Node(labels[0])
    else:
        best = find_best(data, attributes, classifier, classified)
        best_index = get_index(best, attributes)

        node = Node(best)
        count -= 1
        classified.append(best)
        vals = get_values(data, best_index)

        for val in vals:
            subset = np.array([entry for entry in data if entry[best_index] == val])

            subtree = id_three(subset, classifier, attributes, classified, count)
            count -= len(count_nodes(subtree, attributes, []))
            node.add_child(val, subtree)

        return node




if __name__ == "__main__":
    main(sys.argv)
