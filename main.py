import sys
import time
from search import forward_selection, backward_elimination, near_neighbor

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            row = list(map(float, line.split()))
            if row:
                data.append(row)
    return data

def main():
    print("Welcome to Arielle Haryanto's Feature Selection Algorithm!")
    filename = input("Type in the name of the file to test: ").strip()
    try:
        data = load_data(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    num_instances = len(data)
    num_features  = len(data[0]) - 1

    all_features = list(range(1, num_features + 1))
    all_acc = near_neighbor(data, all_features)
    print("\nType the number of the algorithm you want to run:")
    print("\t1. Forward Selection")
    print("\t2. Backward Elimination")
    print(f"\nThis dataset has {num_features} features (not including the class attribute), with {num_instances} instances.")
    print(f'\nRunning nearest neighbor with all {num_features} features, using "leaving-one-out" evaluation, I get an accuracy of {all_acc*100:.1f}%')
    choice = input()
    start = time.time()
    if choice == '1':
        forward_selection(data, num_features, filename)
    elif choice == '2':
        backward_elimination(data, num_features, filename)
    else:
        print("Invalid choice, enter 1 or 2!")
    end = time.time()
    print(f"\nSearch completed in {(end - start) / 60:.2f} minutes")

if __name__ == '__main__':
    main()