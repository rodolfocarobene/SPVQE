import json
import sys


if __name__ == '__main__':
    results = {}

    for i in range(1, len(sys.argv)):
        file = open(sys.argv[1])
        data = json.load(file)
        file.close()

    

