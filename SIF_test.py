from src.SIF import SIF_embedding
from src.utils import *

#%%
# input
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme

#%%
count_dict = get_all_count(token_list) #获取词频
#%%
weight_dict = getWordWeight(count_dict, weightpara)
#%%
weight_list = copy.deepcopy(token_list)

for i in range(len(weight_list)):
    for j in range(len(weight_list[i])):
        weight_list[i][j] = weight_dict[weight_list[i][j]]
#%%
# set parameters
rmpca = True
We = embedding_matrix
x = token_list
w = weight_list
# get SIF embedding
embedding_sif = sif_embedding.SIF_embedding(We, x, w, rmpca) # embedding[i,:] is the embedding for sentence i
#%%
training_data = embedding_sif
#
kmeans = KMeans(n_clusters=2, random_state=0).fit(training_data)
labels = kmeans.labels_
#%%
count = 0
for i in range(4000):
    if labels[i] == test_data[i][1]:
        count += 1
#%%
g = mixture.GaussianMixture(n_components=2)
#%%
g.fit(training_data)
#%%
labels_gmm = g.predict(training_data)
#%%
count_gmm = 0
for i in range(4000):
    if labels_gmm[i] == test_data[i][1]:
        count_gmm += 1