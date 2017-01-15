import os
import numpy as np

results = []
dirs = [d[0] for d in list(os.walk('vectors'))][1:]
for d in dirs:
  with open(d + '/results.csv') as f:
    results += [f.read().replace('\n',',').replace('\r','').split(',')[3:-1]]

results = [[float(c) for c in row][1:-1:3] for row in results]

results = np.asarray(results)
#print(results)

max_exp = dirs[np.argmax(results)]
max_res = np.max(np.max(results))
print(max_exp, max_res)