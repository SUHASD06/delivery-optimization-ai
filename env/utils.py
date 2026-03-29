import math

def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def get_traffic(traffic_map, loc):
    x, y = loc
    return traffic_map[x][y]

def is_clustered(target, deliveries):
    for d in deliveries:
        if abs(d[0]-target[0]) <= 2 and abs(d[1]-target[1]) <= 2:
            return True
    return False

def cluster_score(target, deliveries):
    count = 0
    for d in deliveries:
        if abs(d[0]-target[0]) <= 2 and abs(d[1]-target[1]) <= 2:
            count += 1
    return count