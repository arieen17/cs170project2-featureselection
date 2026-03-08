import sys
from search import forward_selection

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

    print(f"\nThis dataset has {num_features} features, with {num_instances} instances.")
    print("\nType the number of the algorithm you want to run:")
    print("  1. Forward Selection")
    print("  2. Backward Elimination")
    choice = input()
    if choice == '1':
        forward_selection(data, num_features)
    # elif choice == '2':
    #     backward_elimination
    # else:
    #     print("Invalid choice. Please try again.")


if __name__ == '__main__':
    main()