import matplotlib.pyplot as plt
import numpy as np
import re
import jieba
import copy
from src.SIF.SIF_embedding import sif_embedding

class DataProcess:
    @staticmethod
    def _load_data_from_txt(pos_path, neg_path):
        data_list = []
        with open(pos_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                dic = eval(line)
                new_dic = [dic["text"], dic["label"]]
                data_list.append(new_dic)
        with open(neg_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                dic = eval(line)
                new_dic = [dic["text"], dic["label"]]
                data_list.append(new_dic)

        return data_list

    @staticmethod
    def _look_up_dict(sentence, embedding_dim, embedding_matrix):
        sen_matrix = np.zeros((len(sentence), embedding_dim))
        for i in range(len(sentence)):
            index = sentence[i]
            # if index >= 50000:
            #    sen_matrix[i, :] = 0
            # else:
            sen_matrix[i, :] = embedding_matrix[index, :]
        return sen_matrix

    @staticmethod
    def _get_sentence_vector_by_mean(sen_matrix):
        vector = np.mean(sen_matrix, axis=0)
        return vector

    @staticmethod
    def _ave_sen_vector(token_list, embedding_dim, embedding_matrix):
        training_data = np.zeros((len(token_list), embedding_dim))
        for i in range(len(token_list)):
            sentence = token_list[i]
            sen_matrix = DataProcess._look_up_dict(sentence, embedding_dim, embedding_matrix)
            vector = DataProcess._get_sentence_vector_by_mean(sen_matrix)
            training_data[i, :] = vector
        return training_data

    @staticmethod
    def _get_all_count(token_list):
        count_dict = {}
        for i in range(len(token_list)):
            sen = token_list[i]
            for j in range(len(sen)):
                if sen[j] not in count_dict:
                    count_dict[sen[j]] = 1
                else:
                    count_dict[sen[j]] += 1
        return count_dict

    @staticmethod
    def _getWordWeight(count_dict, a=1e-3):
        weight_dict = {}
        if a <= 0:  # when the parameter makes no sense, use unweighted
            a = 1.0

        total = 0
        for key, value in count_dict.items():
            total += value
        for key, value in count_dict.items():
            weight_dict[key] = a / (a + value / total)
        return weight_dict

    @staticmethod
    def _cut_and_encode(data_list, cn_model):
        token_list = []
        for i, ele in enumerate(data_list):
            text = ele[0]
            text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
            cut = jieba.cut(text)
            cut_list = [i for i in cut]
            for j, word in enumerate(cut_list):
                try:
                    # 将词转换为索引index
                    cut_list[j] = cn_model.vocab[word].index
                except KeyError:
                    # 如果词不在字典中，则输出0
                    cut_list[j] = 0
            token_list.append(cut_list)
        return token_list

    @staticmethod
    def _reverse_tokens(tokens, cn_model):
        text = ''
        for i in tokens:
            if i != 0:
                text = text + cn_model.index2word[i]
            else:
                text = text + ' '
        return text

    @staticmethod
    def _compute_acc(gt, pr):
        wrong_num = np.count_nonzero(gt - pr)
        return wrong_num/len(gt)


class SifEmbedding:
    def __init__(self):
        self.rmpc = True
        self.weightpara = 1e-3
    def sif_sen_vector(self, embedding_matrix, token_list):
        count_dict = DataProcess._get_all_count(token_list)  # 获取词频
        weight_dict = DataProcess._getWordWeight(count_dict, self.weightpara)
        weight_list = copy.deepcopy(token_list)
        for i in range(len(weight_list)):
            for j in range(len(weight_list[i])):
                weight_list[i][j] = weight_dict[weight_list[i][j]]
        rmpca = self.rmpc
        We = embedding_matrix
        x = token_list
        w = weight_list
        # get SIF embedding
        sifsen_vector = sif_embedding(We, x, w, rmpca)  # embedding[i,:] is the embedding for sentence i
        return sifsen_vector



# def biKmeans(dataSet, k, distMeas=distEclud):
#     m = np.shape(dataSet)[0]
#     clusterAssment = np.mat(np.zeros((m, 2)))  # 保存数据点的信息（所属类、误差）
#     centroid0 = np.mean(dataSet, axis=0).tolist() # 根据数据集均值获得第一个簇中心点
#     centList = [centroid0]  # 创建一个带有质心的 [列表]，因为后面还会添加至k个质心
#     for j in range(m):
#         clusterAssment[j, 1] = distMeas(centroid0, dataSet[j, :]) ** 2  # 求得dataSet点与质心点的SSE
#     while (len(centList) < k):
#         lowestSSE = np.inf
#         for i in range(len(centList)):
#             ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]  # 与上面kmeans一样获得属于该质心点的所有样本数据
#             # 二分类
#             centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  # 返回中心点信息、该数据集聚类信息
#             sseSplit = sum(splitClustAss[:, 1])  # 这是划分数据的SSE    加上未划分的 作为本次划分的总误差
#             sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])  # 这是未划分的数据集的SSE
#             print("划分SSE, and 未划分SSE: ", sseSplit, sseNotSplit)
#             if (sseSplit + sseNotSplit) < lowestSSE:  # 将划分与未划分的SSE求和与最小SSE相比较 确定是否划分
#                 bestCentToSplit = i  # 得出当前最适合做划分的中心点
#                 bestNewCents = centroidMat  # 划分后的两个新中心点
#                 bestClustAss = splitClustAss.copy()  # 划分点的聚类信息
#                 lowestSSE = sseSplit + sseNotSplit
#         bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(
#             centList)  # 由于是二分，所有只有0，1两个簇编号，将属于1的所属信息转为下一个中心点
#         bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit  # 将属于0的所属信息替换用来聚类的中心点
#         print('本次最适合划分的质心点: ', bestCentToSplit)
#         print('被划分数据数量: ', len(bestClustAss))
#         centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # 与上面两条替换信息相类似，这里是替换中心点信息，上面是替换数据点所属信息
#         centList.append(bestNewCents[1, :].tolist()[0])
#         clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],
#         :] = bestClustAss  # 替换部分用来聚类的数据的所属中心点与误差平方和 为新的数据
#     return np.mat(centList), clusterAssment

# 欧式距离计算
# def distEclud(vecA, vecB):
#     return np.sqrt(sum(np.power(vecA - vecB, 2)))  # 格式相同的两个向量做运算
#
#
# # 中心点生成 随机生成最小到最大值之间的值
# def randCent(dataSet, k):
#     n = np.shape(dataSet)[1]
#     centroids = np.mat(np.zeros((k, n)))  # 创建中心点，由于需要与数据向量做运算，所以每个中心点与数据得格式应该一致（特征列）
#     for j in range(n):  # 循环所有特征列，获得每个中心点该列的随机值
#         minJ = min(dataSet[:, j])
#         rangeJ = float(max(dataSet[:, j]) - minJ)
#         centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))  # 获得每列的随机值 一列一列生成
#     return centroids
#
#
# # 返回 中心点矩阵和聚类信息
# def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
#     m = np.shape(dataSet)[0]
#     clusterAssment = np.mat(np.zeros((m, 2)))  # 创建一个矩阵用于记录该样本 （所属中心点 与该点距离）
#     centroids = createCent(dataSet, k)
#     clusterChanged = True
#     while clusterChanged:
#         clusterChanged = False  # 如果没有点更新则为退出
#         for i in range(m):
#             minDist = np.inf;
#             minIndex = -1
#             for j in range(k):  # 每个样本点需要与 所有 的中心点作比较
#                 distJI = distMeas(centroids[j, :], dataSet[i, :])  # 距离计算
#                 if distJI < minDist:
#                     minDist = distJI;
#                     minIndex = j
#             if clusterAssment[i, 0] != minIndex:  # 若记录矩阵的i样本的所属中心点更新，则为True，while下次继续循环更新
#                 clusterChanged = True
#             clusterAssment[i, :] = minIndex, minDist ** 2  # 记录该点的两个信息
#         # print(centroids)
#         for cent in range(k):  # 重新计算中心点
#             # print(dataSet[nonzero(clusterAssment[:,0] == cent)[0]]) # nonzero返回True样本的下标
#             ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # 得到属于该中心点的所有样本数据
#             centroids[cent, :] = np.mean(ptsInClust, axis=0)  # 求每列的均值替换原来的中心点
#     return centroids, clusterAssment
