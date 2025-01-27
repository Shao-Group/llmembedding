import sys
from embedding import bert_embedding
import numpy as np

def cos_dist(q: np.ndarray):
    x = q[1::2]
    y = q[::2]

    n = x.shape[0]
    ans = []

    for i in range(n):
        xi = x[i]
        yi = y[i].T
        
        sqr_x = np.sqrt(np.sum(xi ** 2))
        sqr_y = np.sqrt(np.sum(yi ** 2))

        cos = xi @ yi / sqr_x / sqr_y

        ans.append((1 - cos) / 2.0)

    return ans

def l2_dist(q: np.ndarray):
    x = q[1::2]
    y = q[::2]    

    n = x.shape[0]
    ans = []
    
    for i in range(n):
        xi = x[i]
        yi = y[i].T
        
        sqr_x = np.sum(xi ** 2)
        sqr_y = np.sum(yi ** 2)

        ans.append(np.sqrt(sqr_x + sqr_y - 2 * xi @ yi))

    return ans

f = open(sys.argv[1])

seqs = []
edits = []

x = f.readline()
while len(x) > 0:
    y = x.strip().split(',')
    seqs.append(y[0])
    seqs.append(y[1])

    edits.append(float(y[2]) / ((len(y[0]) + len(y[1])) / 2.0))

    x = f.readline()

emb = bert_embedding(seqs, sys.argv[2])
dist = cos_dist(emb)

from scipy import stats
res = stats.spearmanr(dist, edits)
print(res.statistic)
