import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as den
d_width = 450
pd.set_option('display.width',d_width)
pd.set_option('display.max_columns',16)
dataset = pd.read_csv('ObesityDataSet.csv')
#Data Preprocessing
#print(dataset.isnull())
#print(dataset.isnull().sum())
imp = SimpleImputer(missing_values=np.nan,strategy = 'mean')
imp = imp.fit(dataset.iloc[:,1:4])
dataset.iloc[:,1:4] = imp.transform(dataset.iloc[:,1:4])

print(dataset.isnull().sum())

labelencoder_X=LabelEncoder()
dataset_new = dataset
#remo

#label encoder f`or gender familyhist favc smoke scc
for i in [0,4,5,9,11]:
    dataset_new.iloc[:, i] = labelencoder_X.fit_transform(dataset_new.iloc[:, i])


#label encoding for the CALC,CAEC,MTRANS
for i in [8,14,15]:
    dataset_new.iloc[:, i] = labelencoder_X.fit_transform(dataset_new.iloc[:, i])


#Label Encoding for NObesity Level
dataset_new.iloc[:, 16] = labelencoder_X.fit_transform(dataset_new.iloc[:, 16])


print('Print Data set with label encoding done of Nobesity')
print('Data set New')
print(dataset_new)


#kmeans with with height and weight
kmeans_clustering1 = dataset_new.loc[:,['Height','Weight']]
#print(kmeans_clustering1)
distortions = []
for i in range(1, 15):
    km1 = KMeans(n_clusters = i, init = 'random', n_init = 10, max_iter = 200, tol = 1e-04, random_state = 0)
    km1.fit(kmeans_clustering1)
    distortions.append(km1.inertia_)
# plot
plt.plot(range(1, 15), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

km = KMeans(n_clusters = 3, init = 'random', n_init = 10, max_iter = 200, tol = 1e-04, random_state = 0)
X = np.array(kmeans_clustering1)
y1 = km.fit_predict(kmeans_clustering1)
plt.scatter(X[y1 == 0, 0], X[y1 == 0, 1], s = 40, c = 'green', marker = 's',label ='Cluster 1')

plt.scatter(X[y1 == 1, 0], X[y1 == 1, 1], s = 40, c = 'blue', marker = 'o', label = 'Cluster 2')

plt.scatter(X[y1 == 2, 0], X[y1 == 2, 1], s = 40, c = 'yellow', marker = 'v', label = 'Cluster 3')

#plt.scatter(X[y_km == 3, 0], X[y_km == 3, 1], s = 40, c = 'grey', marker = 'p', label = 'Cluster 4')

#plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s = 200, marker ='X', c ='red', label = 'Centroids')

plt.legend(scatterpoints = 1)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()



#dbscan By between columns WEIGHT,ncp,caec

X2 = dataset_new.loc[:,['Weight','NCP','CAEC','CALC']]
# labelencoder_Gender = LabelEncoder()
# X2.iloc[:,3] = labelencoder_Gender.fit_transform(X2.iloc[:,3])
#
# scalervalues = StandardScaler()
# X_scalervalues = scalervalues.fit_transform(X2)
#print(X_scalervalues)
dbscan = DBSCAN(eps = 25, min_samples = 100)

y_dbscan = dbscan.fit_predict(X2)
#plot between Weight and NCP
plt.scatter(X2.iloc[:, 0], X2.iloc[:, 1], c = y_dbscan, cmap = "plasma")
plt.xlabel('Weight')
plt.ylabel('NCP')
plt.show()
#plot between Weight and CAEC
plt.scatter(X2.iloc[:, 0], X2.iloc[:, 2], c = y_dbscan, cmap = "plasma")
plt.xlabel('Weight')
plt.ylabel('CAEC')
plt.show()

#plot between Weight and CALC'
plt.scatter(X2.iloc[:, 0], X2.iloc[:, 3], c = y_dbscan, cmap = "plasma")
plt.xlabel('Weight')
plt.ylabel('CALC')
plt.show()

#Hierarchial agloromerative clustering

kmeans_clustering1 = dataset_new.loc[:,['Weight','FAF']]
X = np.array(kmeans_clustering1)
hc = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage ='average')
y = hc.fit_predict(X)
plt.scatter(X[y == 0, 0],X[y == 0, 1], s = 40, c='blue', label ='Cluster 1')
plt.scatter(X[y == 1, 0],X[y == 1, 1], s = 40, c='red', label ='Cluster 2')
plt.scatter(X[y == 2, 0], X[y == 2, 1], s = 40, c='yellow', label ='Cluster 3')
plt.scatter(X[y == 3, 0], X[y == 3, 1], s = 40, c='green', label ='Cluster 4')
#plt.scatter(X[y == 4, 0], X[y == 4, 1], s = 40, c='grey', label ='Cluster 5')
#plt.scatter(X[y == 5, 0], X[y == 5, 1], s = 40, c='black', label ='Cluster 6')
plt.xlabel('Weight')
plt.ylabel('FAF ')
plt.legend()
plt.show()


# Regression
import numpy as np
#finding the correlation between the data values

import matplotlib.pyplot as plt
import seaborn as sns #used for making statistical graphs
plt.figure(figsize = (10,10))
sns.heatmap(dataset_new.corr(), annot=True)
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
#making regression algorithm 'Age' , 'Height', 'Gender','Weight','family_history_with_overweight','CAEC','FAF'
xpart = dataset_new[['Age' , 'Height', 'Gender','Weight','family_history_with_overweight','CAEC','FAF']].values
ypart = dataset_new['NObesity']

xtrain, xtest, ytrain, ytest = train_test_split(xpart, ypart, train_size = 0.75, random_state=101)


regressionmodel = LinearRegression()
regressionmodel.fit(xtrain, ytrain)

ypredicted= regressionmodel.predict(xtest)
#calculating rmse and mae for regressionmodel
import math
rootmeansquarederror = math.sqrt(mean_squared_error(ytest, ypredicted))
meanabosoluteerror = mean_absolute_error(ytest, ypredicted)
print('RMSE and MAE for regressionmodel')
print("RMSE = ", rootmeansquarederror)
print("MAE = ",meanabosoluteerror)

xpart1 = dataset_new[['Age' , 'Weight','family_history_with_overweight','CAEC']].values
ypart1 = dataset_new['NObesity']

xtrain1, xtest1, ytrain1, ytest1 = train_test_split(xpart1, ypart1, train_size = 0.75, random_state=101)


regressionmodel1 = LinearRegression()
regressionmodel1.fit(xtrain1, ytrain1)

ypredicted1= regressionmodel1.predict(xtest1)
#calculating rmse and mae regressionmodel1
import math
rootmeansquarederror1 = math.sqrt(mean_squared_error(ytest1, ypredicted1))
meanabosoluteerror1 = mean_absolute_error(ytest1, ypredicted1)
print('RMSE and MAE for regressionmodel1')
print("RMSE = ", rootmeansquarederror1)
print("MAE = ",meanabosoluteerror1)

#classification
#setting optimal parameter for knn classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
hyper_parameter = {
                'n_neighbors': range(1,16),
                'p':range(1,5)
               }

xclasspart = dataset_new[['Age' , 'Height', 'Gender','Weight','family_history_with_overweight','CAEC','FAF']].values
yclasspart = dataset_new['NObesity']
search = GridSearchCV(KNeighborsClassifier(), hyper_parameter)
search.fit(xclasspart, yclasspart)
print('Best Parameters for KNN classifier')
print(search.best_params_)
xtrain3, xtest3, ytrain3, ytest3 = train_test_split(xclasspart, yclasspart, train_size = 0.75, random_state = 101)
knmodel = KNeighborsClassifier(n_neighbors = search.best_params_['n_neighbors'], p = search.best_params_['p'])
knmodel.fit(xtrain3, ytrain3)

#making classification report of knn clasifier
from sklearn.metrics import classification_report

ypredicted2 = knmodel.predict(xtest3)
print('Classification Report')
print(classification_report(ytest3, ypredicted2))


