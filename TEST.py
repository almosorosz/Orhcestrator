import sys

def greet(name):
    print(f"Hello, {name}!")

if __name__ == "__main__":
    name = sys.argv[1]   # read the first command-line argument
    greet(name)