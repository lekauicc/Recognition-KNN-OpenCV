import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 生成训练数据
train_images = np.random.rand(10, 784)  # 假设有10个训练图像，每个图像为784维向量
train_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 对应的标签，这里假设只有两个类别

# 生成测试数据
test_image = np.random.rand(1, 784)  # 假设有1个测试图像

# 使用KNN算法进行训练和预测
k = 3  # 设置K值
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(train_images, train_labels)
distances, indices = knn.kneighbors(test_image)

# 计算匹配度得分
max_distance = np.max(distances)
match_scores = 1 - distances / max_distance
print(match_scores)
# 可视化匹配度得分
plt.bar(range(len(match_scores[0])), match_scores[0])
plt.xlabel('Training Images')
plt.ylabel('Match Score')
plt.xticks(range(len(match_scores[0])), indices[0])
plt.title('Matching Scores with Training Images')
plt.show()
