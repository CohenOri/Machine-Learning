import numpy as np
import sys  # in order to read command line args


# main to run consistency algorithm
def main():
    # Receives two args, first is by default program name,
    # second should be the training set file path (.txt file)
    if len(sys.argv) < 2:
        raise RuntimeError('no file path for training set')

    # (try to) read the training set and insert into matrices
    try:
        training_examples = np.loadtxt(sys.argv[1])
        cols = training_examples.shape[1]
    except:
        raise RuntimeError("error opening/reading the training set file")

    # stores only the examples of the training data. each row of the matrix is an example
    X = training_examples[:, :cols - 1]
    # stores only the classification of the training data. each row of the vector is classification
    Y = training_examples[:, cols - 1]

    # run the consistency_algorithm over X, Y, save the resulted hypothesis
    h = consistency_algorithm(X, Y)
    #  convert h to conjunction string
    h_str = conjunction_to_str(h)

    # write h to output.txt file
    output_file = open("output.txt", "w")
    output_file.write(h_str)
    output_file.close()


# Receives X, Y matrices, and returns "h" such that "h" is conjunction which classifies all the
# examples given in the matrices properly
def consistency_algorithm(X, Y):
    # calculate d - length of input, "conjunction" (product (and) of literal...)
    d = X.shape[1]  # amount of different literal
    # calculate the amount of different examples
    examples = X.shape[0]

    # set h as all-negative hypothesis
    h = np.ones((2 * d,))

    # iterate over the different examples in X matrix
    for instance in xrange(examples):
        # calculate the predicted classification
        predicted_classification = calc_conjunction(h, X[instance])
        # if prediction is wrong - fix it
        if Y[instance] == 1 and predicted_classification == 0:
            # iterate over the variables (literal) in the specific example
            for literal in xrange(X[instance].shape[0]):
                # if the literal value is 1 (remove xi from h)
                if X[instance][literal] == 1:
                    h[(literal * 2) + 1] = 0
                # if the literal value is 0 (remove xi from h)
                if X[instance][literal] == 0:
                    h[(literal * 2)] = 0
    return h


# Receives conjunction and specific example ("instance")
# Returns 1 if the conjunction value (aka classification) is true otherwise returns 0
def calc_conjunction(conjunction, example):
    val = 1
    for literal in xrange(conjunction.size):
        if conjunction[literal] == 1:
            if literal%2 == 0:
                # do the 'and' as multiplication [multiplying in binary is the same as AND... 1*1 = 1
                # otherwise 0 (0*1, 1*0, 0*0) ]
                # index of the literal == literal / 2
                val = val * example[literal / 2]
            if literal%2 != 0:
                val = val * abs(1- example[literal / 2])
    return val


# function get conjunction
# the function make string from the conjunction
def conjunction_to_str(conjunction):
    conjunction_string = ""

    # iterate over the literals in the conjunction
    for literal in xrange(conjunction.size):
        if conjunction[literal] == 1:
            # calculate the literal index
            index = (literal / 2) + 1
            if literal % 2 == 0:
                # regular literal
                conjunction_string += "x" + str(index) + ","
            else:
                # not(literal)
                conjunction_string += "not(x" + str(index) + ")" + ","

    # remove the last extra "," (converts to list without the last "," and convert back to string)
    conjunction_string = [letter for letter in conjunction_string[0:-1]]
    final_conjunction = ''.join(conjunction_string)
    return final_conjunction


main()
