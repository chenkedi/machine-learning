
file_dir = "raw_data/results/batch/batchTest_%s/distance_%s.csv"
res = {'part_0':{}, 'part_1':{}, 'part_2':{}}
minCost = 9999999


for i in range(50):
    for j in range(3):
        with open(file_dir % (i, j)) as file:
            for line in file.readlines():
                if(('breakDistanceCount' in line) and len(line.split(': ')) == 2):
                    if(int(line.split(': ')[1]) == 0):
                        res['part_%s' % j][i] = minCost
                        continue

                if(('totalCost' in line) and len(line.split(': ')) == 2):
                    cost = float(line.split(': ')[1].strip().strip('Optional[').rstrip(']'))
                    if(i in res['part_%s' % j]):
                       res['part_%s' % j][i] = cost

# 0, 1, 2到批次的映射
final = {}
for key, value in res.items():
    index = int(key.split('_')[1])
    print(index)
    minCost = 8999999999
    minBatch = 0
    for batch, cost in value.items():
        if(cost < minCost):
            minCost = cost
            minBatch = batch
    final[index] = batch

print(final)




