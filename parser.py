

def parse(filename):
    dict = {}
    with open(filename, 'r') as f:
        while True:
            if f.readline().strip() == 'NODE_COORD_SECTION':
                break

        for line in f:
            if line.strip() == 'EOF':
                break
            else:
                data = line.strip().split(" ")
                dict[int(data[0])] = (int(data[1]), int(data[2]))

    return dict


DATA_WESTERN_SAHARA = 'data/wi29.tsp'
DATA_DJIBOUTI = 'data/dj38.tsp'
DATA_QATAR = 'qa194.tsp'
DATA_URUGUAY = 'uy734.tsp'

print(parse(DATA_WESTERN_SAHARA))