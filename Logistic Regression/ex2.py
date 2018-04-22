# Ori Cohen
# ID: 207375783

import numpy as np
import math
import matplotlib.pyplot as plot

# run logistic_regression_algorithm to calculate w and b
def logistic_regression_algorithm(train_set):
    # create w, b vectors
    w = np.ones(3)
    b = np.ones(3)
    # preform 20 epoch on shuffled training set in order to calculate w, b
    for epoch in xrange(0,20):
        np.random.shuffle(train_set)
        # for each xi yi in vectors
        for row in xrange(0,train_set.size/train_set[0].size):
            # update the w,b accoriding to given Xi Yi sample
            w, b = update(w, b, train_set[row][0], train_set[row][1])
    return w,b


# create train set data and shuffle it
def create_train_set():
    train_set = np.zeros((300, 2))
    # create 100 examples of each classification N(2a,1) where a = 1,2,3
    # assume a = 1
    a = 1
    mu, sigma = 2 * a, 1  # mean and standard deviation (sqrt of 1 is one therefore sigma stays one)
    add_to_train_set(a, mu, sigma, train_set)
    a = 2  # assume a = 2
    mu, sigma = 2 * a, 1
    add_to_train_set(a, mu, sigma, train_set)
    a = 3  # assume a = 2
    mu, sigma = 2 * a, 1
    add_to_train_set(a, mu, sigma, train_set)
    np.random.shuffle(train_set)
    return train_set


# utility method in order to concat the three classification different vectors into one vector
def add_to_train_set(a, mu, sigma, train_set):
    samples_without_class = np.random.normal(mu, sigma, 100)  # create list with 100 examples assume a = 1
    if a == 1:
        for i in xrange(0, 100):
            train_set[i][0] = samples_without_class[i]
            train_set[i][1] = a
    elif a == 2:
        for i in xrange(100, 200):
            train_set[i][0] = samples_without_class[i-100]
            train_set[i][1] = a
    elif a == 3:
        for i in xrange(200, 300):
            train_set[i][0] = samples_without_class[i-200]
            train_set[i][1] = a


# update w, b according to the formula (theortical part... with softmax)
def update(w,b,sample,tag):
    eta = 0.1 # the greek letter which looks like n with tale...
    for i in xrange(0, w.size):
        # calc according to formula Wt = Wt-1 - eta(....)
        row = i + 1
        if row == tag: # preform y==row update rule
            temp_w = sample*(softmax(w, b, sample, i)-1)
            temp_b = - 1 + softmax(w, b, sample, i)
        else: # preform y!=row update rule
            temp_w = softmax(w, b, sample, i) * sample
            temp_b = softmax(w, b, sample, i)

        # update w and b
        w[i] = w[i] - eta * temp_w
        b[i] = b[i] - eta * temp_b
    return w, b


# calc softmax(W*sample+b) at i'th place according the formula @ stackoverfolow
def softmax(w, b, sample, i):
    sum = 0
    for counter in xrange(0, w.size):
        sum += np.exp(np.dot(w[counter], sample) + b[counter])
    return np.exp(np.dot(w[i], sample) + b[i]) / sum


# predict the tag of the sample
def predict(sample, w, b):
    predict_vec = []
    for i in xrange(0, w.size):
        predict_vec.append(softmax(w, b, sample, i))
    return predict_vec


# calc normal probability density @ stackoverfolow
# sd = Standard deviation = sigma
# mean = Expected value = mu
# x is what we insert to f(x)
def norm_prob_denst_func(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


# generate list which holds the numbers from start to stop (not include stop) with the given steps
def list_of_sample_points(start,end,step):
    i = start
    i += step
    list = []
    while i < end:
        list.append(i)
        i+= step
    return list


# main to run consistency algorithm
def main():
    train_set = create_train_set() # receives shuffled train set
    w, b = logistic_regression_algorithm(train_set) # train the algorithm and return w,b

    # draw plot
    # create list which holds the regression points and list contains actual points
    reg_list = []
    actual_list = []
    # use samples from 0 to 10 with 0.05 steps
    sample_index = list_of_sample_points(0,10,0.05)
    for i in xrange(0, 199):
        specific_sample = sample_index[i]
        # save the specific sample actual and regression value to mathcing list
        reg_val = predict(specific_sample, w, b)[0]
        reg_list.append(reg_val)

        actual_val = norm_prob_denst_func(specific_sample,2, 1) /\
                     (norm_prob_denst_func(specific_sample, 2, 1) +
                      norm_prob_denst_func(specific_sample, 4, 1) +
                      norm_prob_denst_func(specific_sample, 6, 1))
        actual_list.append(actual_val)

    # plot the two lists to graph
    plot.plot(sample_index, actual_list, "-g", label='actual')
    plot.plot(sample_index, reg_list, "-r", label='regression')
    plot.show()

# run main
main()
