import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt


def load_data():
    df = pd.read_csv('user_pins_description.csv',sep='\t', encoding='utf-8')
#index problem
    df.pop('Unnamed: 0')
    return df

def clean(df):
#drop duplicates in pin_description
    df = df.drop_duplicates(subset=['user_id','pin_description'])
    df = df[df['pin_description'] != ' ']
    df = df.reset_index()
    df.pop('index')
    df = df.dropna()
    return df

#df['pin_description'] = df['pin_description'].str.replace('\+','')
#remove

def remove_non_alphanumeric_words(words):
    return re.sub('[^0-9A-Za-z \t]', ' ', words)

def tokenize(doc):
    # Stemmer
    snowball = SnowballStemmer('english')
    return [snowball.stem(token) for token in word_tokenize(doc.lower())]

def print_pin_list_for_each_user(df):
    user_pins = []
    user_list = df['user_id'].value_counts().index
    for user in user_list:
        a = ' '.join([text for text in df[df['user_id'] == user].pin_description.values])
        user_pins.append(a)
    return user_pins


def db_scan(X):

# Default: DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, n_jobs=1)
# eps - The maximum distance between two samples for them to be considered as in the same neighborhood.
# min_samples - The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
# metric - The metric to use when calculating distance between instances in a feature array: euclidean
# algorithm - auto - The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors.
# leaf_size - Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
# p - The power of the Minkowski metric to be used to calculate distance between points.

    db = DBSCAN(eps=0.3, min_samples=3, metric='cosine', algorithm='brute', n_jobs=1).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    return db, n_clusters_, labels, core_samples_mask

def plot_dbscan(db, n_clusters_, labels, core_samples_mask):
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)

    plt.ylim([-1,1])
    plt.xlim([-1,1])
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.savefig('cluster.png')
    return



if __name__ == '__main__':
    df = load_data()
    df = clean(df)
    # Remove punctuation, and strage symbols
    df['pin_description'] = df['pin_description'].apply(lambda x: remove_non_alphanumeric_words(x))
    # Fix 3 and 2 empty spaces
    df['pin_description'] = df['pin_description'].str.replace('   ',' ')
    df['pin_description'] = df['pin_description'].str.replace('  ',' ')
    # new_df is a new DF that aggregates the pins descriptions into one
    #new_df = df.groupby('user_id').agg({'pin_description':lambda x:' '.join(x)})
    df.reset_index(level=0, inplace=True)


    vect = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
    X = vect.fit_transform(df['pin_description'])
    X = X.todense()

    db, n_clusters, labels, core_samples_mask = db_scan(X)


    #Tokenize and stem/lemmatize the document.
    #df['pin_description_token_stem'] = df['pin_description'].apply(lambda x: tokenize(x))
