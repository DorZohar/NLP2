import os
import numpy as np

results = []
dirs = [d[0] for d in list(os.walk('vectors'))][1:]

new_dirs = []
for idx, d in enumerate(dirs):
    if not os.path.exists(d + '/results.csv'):
      del(dirs[idx])
      continue
    new_dirs += [d]
    with open(d + '/results.csv') as f:
      results += [f.read().replace('\n',',').replace('\r','').split(',')[3:-1]]
         
results = [[float(c) for c in row][1:-1:3] for row in results]

results = np.asarray(results)
#print(results)

max_exp = new_dirs[np.argmax(results)]
max_res = np.max(np.max(results))
print("top vector = " + str(max_exp) + " result = " + str(max_res) + "%")

print("")
#for d, row in zip(new_dirs, results):
#  print(str(row)  + '\t\t' + d)
  
ordered_results = sorted(enumerate([np.max(row) for row in results]),key=lambda x: x[1])

for row in ordered_results:
  print(str(row[1])  + '%\t\t' + new_dirs[row[0]])
    