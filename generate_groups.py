import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os

# reading csv that has <userid>, <numeric_feature_1>, <numeric_feature_2>...
df = pd.read_csv('{}\\your_csv.csv'.format(os.getcwd()))

# list the names of the columns that has your numeric features
feature_cols = [
    'numeric_feature_1',
    'numeric_feature_2',
    'numeric_feature_3',
    'numeric_feature_4',
    'numeric_feature_5',
    'numeric_feature_6',
    'numeric_feature_7'
    ]

# filling null values with the mean, however consider each of your feature's
# characteristics and consider other methods for filling null values
# or even consider dropping them
df_clean = df.copy()
for col in feature_cols:
    df_clean[col].fillna(df_clean[col].mean(), axis=0, inplace=True)

# feature scaling with min-max normalization
# https://en.wikipedia.org/wiki/Feature_scaling

df_features = df_clean.copy()[feature_cols]
scaler = MinMaxScaler()
df_features[feature_cols] = scaler.fit_transform(df_features[feature_cols])

# defining n_init that goes to KMeans. Explanation:
# The n_init parameter specifies the number of times the KMeans algorithm
# will be executed with different centroid initializations. A bigger number
# for n_init increases the probability of finding a more optimal solution,
# because the algorithm is more likely to start with initial centroids
# many different. However, it also increases the computational cost,
# because the algorithm is executed multiple times.
n_init_kmeans = 20

# plotting the sum of squared distances (wcssd) to define how many clusters
# are needed. wcssd = within cluster sum of squared distances
wcssd = []

for num_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=num_clusters, n_init=n_init_kmeans)
    kmeans.fit(df_features)
    wcssd.append(kmeans.inertia_)

# plotting the wcssd to apply the elbow method and define ideal n clusters
plt.plot(range(1, 11), wcssd, 'o-')
plt.xlabel('N Clusters')
plt.ylabel('Within Cluster Sum of Squared Distances')
plt.title('Elbow Method For Optimal Clusters')
plt.show()

# setting the number of clusters to be used next by KMeans
num_clusters = 3

# applying KMeans
df_cluster = df_clean.copy()
kmeans = KMeans(n_clusters=num_clusters, n_init=n_init_kmeans)
df_cluster['cluster'] = kmeans.fit_predict(df_features)

# function that creates groups A, B and Control
def split_into_groups(df_, n_clusters):
    
    # creates the column in which we will store the group information
    df = df_.copy()
    df['group'] = None

    # iterate over each cluster
    for cluster in range(n_clusters):
        # isolate the indices within this cluster
        cluster_data_indices = df[df['cluster'] == cluster].index

        # random sampling to get Group A users
        group_a_indices = np.random.choice(cluster_data_indices, size=len(cluster_data_indices)//3, replace=False)
        
        # as Group A is already defined, let's isolate the remaining
        remaining_indices = list(set(cluster_data_indices) - set(group_a_indices))
        
        # among the remaining, we sample again to get Group B users
        group_b_indices = np.random.choice(remaining_indices, size=len(remaining_indices)//2, replace=False)
        
        # the remaining users are set to be Control
        control_indices = list(set(remaining_indices) - set(group_b_indices))

        # using the indexes previously sampled, now we fill in the 'group' column
        df.loc[group_a_indices, 'group'] = 'A'
        df.loc[group_b_indices, 'group'] = 'B'
        df.loc[control_indices, 'group'] = 'Control'
    return df

# applying our split groups function to the output df
df_group = split_into_groups(df_cluster, num_clusters)

# some validation steps to ensure we have correct proportions of clusters and groups 
# df_group['group'].value_counts()
# df_group['cluster'].value_counts()
# for n in range(num_clusters):
#     print(n)
#     print(df_group.loc[df_group['cluster'] == n, 'group'].value_counts())


# finally save the list of users with the original data for each feature
# (before feature scaling) plus the clusters and groups
df_group.to_csv(
    '{}\\lista_clusterizada_bq.csv'.format(os.getcwd()), 
    sep=',', 
    decimal='.',
    index=False
    )

