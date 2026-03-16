import numpy as np
import os
from numba import njit

# to help print for the trace file
def log(msg, trace):
    print(msg)
    trace.write(msg + "\n")

@njit
def near_neighbor(data, features):
    if len(features) == 0:
        return 0.0
    correct = 0
    n = data.shape[0]

    for i in range(n):
        best_distance = np.inf
        best_label = -1.0

        for j in range(n):
            if j == i:
                continue  # skip self-comparison (leave-one-out)

            distance = 0.0
            for f in features:
                distance += (data[j, f] - data[i, f]) ** 2

            if distance < best_distance:
                best_distance = distance
                best_label = data[j, 0]  # column 0 is the class label

        if best_label == data[i, 0]:
            correct += 1

    return correct / n


def forward_selection(data, num_features, filename):
    # convert data to numpy
    data = np.array(data)

    # open trace file to log search progress
    base = os.path.basename(filename)
    trace = open(f"forwardtraceback_{base}", "w")

    current_set = []              # features selected so far at this level
    best_overall_accuracy = 0.0   # best accuracy seen across all levels
    best_overall_set = []         # feature set that achieved best_overall_accuracy

    log("\nBeginning forward selection:", trace)
    empty_accuracy = near_neighbor(data, np.empty(0, dtype=np.int64))
    log(f"Using features {set()}: accuracy is {empty_accuracy * 100:.1f}%", trace)

    if empty_accuracy > best_overall_accuracy:
        best_overall_accuracy = empty_accuracy
        best_overall_set = []

    for i in range(1, num_features + 1):
        best_feature = None
        best_accuracy = -1.0

        # try adding each feature not already in the current set
        for j in range(1, num_features + 1):
            if j in current_set:
                continue

            accuracy = near_neighbor(data, current_set + [j])
            log(f"\tUsing features {set(current_set + [j])}: accuracy is {accuracy * 100:.1f}%", trace)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = j

        # append the best feature found at this level
        current_set.append(best_feature)

        # update the overall best if this level's result is an improvement
        if best_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_accuracy
            best_overall_set = list(current_set)

        log(f"Feature set {set(current_set)} has accuracy {best_accuracy * 100:.1f}%", trace)

    log(f"\nBest feature set found: {set(best_overall_set)} with accuracy {best_overall_accuracy * 100:.1f}%", trace)
    trace.close()


def backward_elimination(data, num_features, filename):
    # convert data to numpy
    data = np.array(data)

    # open trace file to log search progress
    base = os.path.basename(filename)
    trace = open(f"backwardtraceback_{base}", "w")

    current_set = list(range(1, num_features + 1))  # start with all features

    # evaluate and record the full-set baseline before any removal
    best_overall_accuracy = near_neighbor(data, current_set)
    best_overall_set = list(current_set)

    log("\nBeginning backward elimination:", trace)
    log(f"Starting with features {set(current_set)}: accuracy is {best_overall_accuracy * 100:.1f}%", trace)

    # remove one feature per iteration until only one feature remains
    for i in range(num_features - 1):
        worst_feature = None  # the feature whose removal yields the best accuracy
        best_accuracy = -1.0

        # try removing each feature still in the current set
        for j in current_set:
            candidate = [f for f in current_set if f != j]

            accuracy = near_neighbor(data, candidate)
            log(f"\tUsing features {set(candidate)}: accuracy is {accuracy * 100:.1f}%", trace)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                worst_feature = j  # removing this feature gave the best result

        # commit the removal of the worst feature
        current_set.remove(worst_feature)

        # update the overall best if this level's result is an improvement
        if best_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_accuracy
            best_overall_set = list(current_set)

        log(f"Feature set {set(current_set)} has accuracy {best_accuracy * 100:.1f}%", trace)
    empty_accuracy = near_neighbor(data, np.empty(0, dtype=np.int64))
    log(f"\tUsing features {set()}: accuracy is {empty_accuracy * 100:.1f}%", trace)

    if empty_accuracy > best_overall_accuracy:
        best_overall_accuracy = empty_accuracy
        best_overall_set = []
    log(f"\nBest feature set found: {set(best_overall_set)} with accuracy {best_overall_accuracy * 100:.1f}%", trace)
    trace.close()