import math

best_so_far = 0.0

def near_neighbor(data, features):
    global best_so_far
    correct = 0
    n = len(data)

    for i in range(n):
        best_distance = float('inf')
        best_label = None

        for j in range(n):
            if j == i:
                continue
            distance = math.sqrt(sum((data[j][k] - data[i][k])**2 for k in features))
            if distance < best_distance:
                best_distance = distance
                best_label = data[j][0]

        if best_label == data[i][0]:
            correct += 1

        if best_so_far > 0:
            remaining: int = n - i - 1
            if (correct + remaining) / n <= best_so_far: 
                return (correct + remaining) / n

    accuracy = correct / n
    if accuracy > best_so_far:
        best_so_far = accuracy
    return accuracy

def forward_selection(data, num_features):
    global best_so_far
    best_so_far = 0.0 
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
            accuracy = near_neighbor(data, current_set + [j])
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
    global best_so_far
    best_so_far = 0.0 
    current_set = list(range(1, num_features + 1))
    best_overall_accuracy = near_neighbor(data, current_set)
    best_overall_set = list(current_set)

    print("\nBeginning backward elimination:")

    for i in range(num_features - 1):
        worst_feature = None
        best_accuracy = -1

        for j in current_set:
            feature = [f for f in current_set if f != j]
            accuracy = near_neighbor(data, feature)
            print(f"\tUsing features {set(feature)}: accuracy is {accuracy*100:.1f}%")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                worst_feature = j
            
        current_set.remove(worst_feature)

        if best_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_accuracy
            best_overall_set = list(current_set)
        print(f"Feature set {set(current_set)} has accuracy {best_accuracy*100:.1f}%")
    print(f"\nBest feature set found: {set(best_overall_set)} with accuracy {best_overall_accuracy*100:.1f}%")
