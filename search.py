import math

def leave_one_out_accuracy(data, current_set, add_feature):
    features = current_set + [add_feature]
    correct = 0
    n = len(data)

    for i in range(n):
        best_distance = float('inf')
        best_label = None

        for j in range(n):
            if j == i:
                continue
            distance = math.sqrt(sum((data[j][k] - data[i][k])**2 for k in features)) # euclidean distance
            if distance < best_distance:
                best_distance = distance
                best_label = data[j][0]
        if best_label == data[i][0]:
            correct += 1
    return correct / n

def forward_selection(data, num_features):
    current_set = []
    best_overall_accuracy = 0.0
    best_overall_set = []

    print("\nBeginning forward selection:")

    for i in range(1, num_features + 1):
        best_feature = None
        best_accuracy = -1

        for j in range(1, num_features + 1):
            if j in current_set:
                continue
            accuracy = leave_one_out_accuracy(data, current_set, j)
            print(f"\tUsing features {set(current_set + [j])}: accuracy is {accuracy*100:.1f}%")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = j
        current_set.append(best_feature)

        if best_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_accuracy
            best_overall_set = list(current_set)
        print(f"Feature set {set(current_set)} has accuracy {best_accuracy*100:.1f}%")

    print(f"\nBest feature set found: {set(best_overall_set)} with accuracy {best_overall_accuracy*100:.1f}%")

def backward_elimination(data, num_features):
    current_set = list(range(1, num_features + 1))
    best_overall_accuracy = leave_one_out_accuracy(data, current_set[:-1], current_set[-1])
    best_overall_set = list(current_set)

    print("\nBeginning backward elimination:")

    for i in range(num_features - 1):
        worst_feature = None
        best_accuracy = -1

        for j in current_set:
            feature = [f for f in current_set if f != j]
            accuracy = leave_one_out_accuracy(data, feature[:-1], feature[-1])
            print(f"\tUsing features {set(feature)}: accuracy is {accuracy*100:.1f}%")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                worst_feature = j
            
            current_set.remove(worst_feature)

            if best_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_accuracy
                best_overall_set = list(current_set)
            
        print(f"\nBest feature set found: {set(best_overall_set)} with accuracy {best_overall_accuracy*100:.1f}%")
