# %%
from gensim.models import KeyedVectors
import warnings
import numpy as np
from src.utils import DataProcess as dp
from src.utils import SifEmbedding
from src.cluster import Cluster

# %% 载入与训练的词向量
warnings.filterwarnings("ignore")
cn_model = KeyedVectors.load_word2vec_format('embeddings/sgns.zhihu.bigram', binary=False, unicode_errors="ignore")
# %% 载入数据【内容， label】
pos_path = "test_data/positive_samples.txt"
neg_path = "test_data/negative_samples.txt"
test_data = dp._load_data_from_txt(pos_path, neg_path)
# %%
token_list = dp._cut_and_encode(test_data, cn_model)
# %%
#embedding_dim = cn_model["你好"].shape[0]
embedding_dim = cn_model.vectors.shape[1]
# %%
num_words = len(cn_model.index2word)  # total = len(cn_model.index2word) 选用前50000个
embedding_matrix = np.zeros((num_words, embedding_dim))
for i in range(num_words):
    embedding_matrix[i, :] = cn_model[cn_model.index2word[i]]
#完整字典的矩阵
embedding_matrix = embedding_matrix.astype('float32')
# %%
senvec_mean = dp._ave_sen_vector(token_list, embedding_dim, embedding_matrix)
#%%
SE = SifEmbedding()
senvec_sif = SE.sif_sen_vector(embedding_matrix, token_list)
#%%
_, kmeans_labels = Cluster.kmeans_cluster(4, senvec_mean)
#%%
_, gmm_labels = Cluster.gmm_cluster(4, senvec_mean)
#%%
_, kmeans_labels_sif = Cluster.kmeans_cluster(4, senvec_sif)
#%%
_, gmm_labels_sif = Cluster.gmm_cluster(4, senvec_sif)
#%%
np.save("temp/sif_gmm_labels.npy", gmm_labels_sif)