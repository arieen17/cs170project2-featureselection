import random

def leave_one_out_accuracy(data, current_set, feature_to_add):
    return random.uniform(0, 1)

def forward_selection(data, num_features):
    current_set = []
    for i in range(1, num_features + 1):
        best_feature = None
        best_accuracy = -1

        for j in range(1, num_features + 1):
            if j in current_set:
                continue
            print(f" Consider adding feature {j}")
            accuracy = leave_one_out_accuracy(data, current_set, j)
            print(f" Accuracy is {accuracy*100:.1f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = j
        current_set.append(best_feature)
        print(f" On level {i}, added {best_feature} to current set\n")