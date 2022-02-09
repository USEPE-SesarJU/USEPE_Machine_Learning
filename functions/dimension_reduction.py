from data_parser import data_import
from data_parser import data_export
from sklearn.decomposition import PCA
import numpy as np

file = 'TEST_LOGGER_logger_20220124_20-39-04.log'
result_file_name = (file[:-4] + '_PCA.csv')
result_file = data_export(result_file_name)

d = data_import(file,1)

d = d.to_numpy()
X = d[:,[0,5,6,7,8,9,10,11,12]]
y = d[:,[3,4]]

def dim_PCA(file, X,comp):
    result_file_name = (file[:-4] + '_PCA_' + str(comp) + '_dim.csv')
    result_file = data_export(result_file_name)
    pca = PCA(n_components=comp)
    pca.fit(X)
    X_reduced = pca.transform(X)
    np.savetxt(result_file, X_reduced, delimiter=",")
    return(X_reduced)

for i in range(1, X.shape[1]):
    dim_PCA(file, X, i)



# # visualization on 3D
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
# import numpy as np
# x = np.linspace(0, 10, 30)
# y = np.sin(x)
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')

# ax.scatter(X_reduced[:,0],X_reduced[:,1],X_reduced[:,2], 'o', color='black');
# plt.show()

# print(X_reduced.shape)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)