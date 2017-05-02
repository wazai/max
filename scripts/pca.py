"""
PCA on 1-year SZ50 stock return
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from numpy import linalg
import matplotlib.pyplot as plt
import logging

from max.datacenter import DataCenter

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')
dc = DataCenter(start_date='2015-01-01', end_date='2017-03-31')

df = dc.price[['code', 'return']]
df = df[df.code.isin(dc.univ_dict['sz50']['code'])]
df = df.pivot(columns='code', values='return')

p = df.shape[1]

# PCA on one year

date = '2017-03-31'
df = df[(df.index > '2016-03-31') & (df.index <= '2017-03-31')]
sigmaX = df.cov()
eigval, eigvec = linalg.eig(sigmaX)

# scree plot

n_eig = np.array(range(p)) + 1
fig = plt.figure()
fig.suptitle('Scree Plot')
ax = fig.add_subplot(111)
ax.set_xlabel('Number of eigenvalues')
ax.set_ylabel('Proportion of Variance')
ax.plot(n_eig, np.cumsum(eigval)/np.sum(eigval), 'o-')
plt.show()

# first 10 PCs
# 600547 山东黄金 negatively correlated with others

n_pc = 10
pc = pd.DataFrame(eigvec[:, :n_pc], index=sigmaX.index, columns=['PC'+str(x+1) for x in range(n_pc)])
plt.imshow(pc)

# save

pca = PCA(n_components=n_pc)
df0 = np.nan_to_num(df)
pca.fit(df0)

print(pca.explained_variance_ratio_)

pca.fit_transform(df0)
