import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from sklearn.metrics.pairwise import linear_kernel
from copy import copy
import random
import feedparser
from bs4 import BeautifulSoup




def load_data(filename):
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
#index problem
    df.pop('Unnamed: 0')
    return df

def clean_pins(df):
    # drop duplicates in pin_description
    df = df.drop_duplicates(subset=['user_id','pin_description'])
    df = df[df['pin_description'] != ' ']
    df = df.reset_index()
    df.pop('index')
    df.reset_index(level=0, inplace=True)
    df = df.dropna()
    # Remove punctuation, and strage symbols
    df['pin_description'] = df['pin_description'].apply(lambda x: remove_non_alphanumeric_words(x))
    # Fix 3 and 2 empty spaces
    df['pin_description'] = df['pin_description'].str.replace('   ', ' ')
    df['pin_description'] = df['pin_description'].str.replace('  ', ' ')
    df = df[df['pin_description'] != ' ']


    return df

def clean_products(df):
    # drop duplicates in pin_description
    df = df.drop_duplicates(subset='image_id')
    df = df[df['new_prod_description'] != ' ']
    df = df.reset_index()
    df.pop('index')
    df = df.dropna()
    df['new_prod_description'] = df['new_prod_description'].apply(lambda x: remove_non_alphanumeric_words(x))
    #df['product_price'] = df['product_price'].apply(lambda x: remove_dollar(x))
    df['product_price'] = df['product_price'].astype(float)
    df['new_prod_description'] = df['new_prod_description'].str.replace('   ', ' ')
    df['new_prod_description'] = df['new_prod_description'].str.replace('  ', ' ')
    df = df[df['new_prod_description'] != ' ']
    return df



def remove_non_alphanumeric_words(words):
    return re.sub('[^0-9A-Za-z \t]', ' ', words)

# remove dollars and converts string to floats
def remove_dollar(words):
    if '-' in words:
# for the case there are two values for price: $5.00 - $15.00, minimum value is chosen
        return min([float(i[1:]) for i in words.split(' - ') if '$' in i ])
    elif '$' in words:
        return float(words[1:])
    return words

def tokenize(doc):
    # Stemmer
    snowball = SnowballStemmer('english')
    return [snowball.stem(token) for token in word_tokenize(doc.lower())]

def print_pin_list_for_each_user(df):
    user_pins = []
    user_list = df['user_id'].value_counts().index
    return [df[df['user_id'] == user].pin_description.values for user in user_list]

def product_list_descriptions(df, price_limit=30):
    prod_desc = []
    df = df[df['product_price'] <= price_limit]
    prod_list = df['image_id'].value_counts().index
    for product in prod_list:
        a = ' '.join([desc for desc in df[df['image_id'] == product].new_prod_description.values])
        prod_desc.append(a)
    return prod_desc


def db_scan(X, eps, min_samples):

# Default: DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, n_jobs=1)
# eps - The maximum distance between two samples for them to be considered as in the same neighborhood.
# min_samples - The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
# metric - The metric to use when calculating distance between instances in a feature array: euclidean
# algorithm - auto - The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors.
# leaf_size - Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
# p - The power of the Minkowski metric to be used to calculate distance between points.

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', algorithm='brute', n_jobs=1).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print('Estimated number of clusters: %d' % n_clusters_)

    return db, n_clusters_, labels, core_samples_mask

# Get request to Pinterest to obtain users information
def get_feed_user(user_friend):
    d = feedparser.parse("https://pinterest.com/" + user_friend + "/feed.rss/")
    l = len(d['entries'])
    database = []
    for pin_info in xrange(l):
        soup = BeautifulSoup(d['entries'][pin_info]['summary'])
        pin_description = soup.findAll('p')[1].text
        pin_link = soup.find('a')['href']
        user_id = d['entries'][pin_info]['summary_detail']['base'].split('/')[3]
        entry = (user_id, pin_description, pin_link)
        database.append(entry)
    df = pd.DataFrame(database, columns=["user_id", "pin_description", "pink_link"])
    return df



if __name__ == '__main__':
    # Clean data
    # df = load_data('../user_pins_description.csv')
    user_friend = str(raw_input('Enter Pinterest username: '))
    price_limit = float(input('Enter maximum gift price: '))
    # get user pins and cleans it
    df = get_feed_user(user_friend)
    df = clean_pins(df)
    df_prod = load_data('../amazon_products_description_final.csv')
    df_prod = clean_products(df_prod)
    # Initiation of NLP
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    # Products list limited by chosen price limit
    prod_doc = product_list_descriptions(df_prod, price_limit)

    # Lists and counters initialization
    total_probs = []
    total_topics = []
    user_clusters_results = []
    user_clusters = []
    check_clusters = []
    user_prefered_topics = []
    recommendations = []
    count_empty = 0


    # LDA implementation
    contador = 0
    for user in df.user_id.unique():
        contador += 1
        print "Username: {}".format(user)
        # make a list of all the pins of the user
        user_pins = print_pin_list_for_each_user(df[df['user_id'] == user])[0]
        descriptions = []
        for pin in user_pins:
            # After cleaning, if charaters are not alphanumeric the pins my be empty
            if pin.strip() == '':
                count_empty += 1
            else:
                raw = pin.lower()
                tokens = tokenizer.tokenize(raw)
                stopped_tokens = [i for i in tokens if not i in en_stop]
                stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
                descriptions.append(stemmed_tokens)
        # If all the pins are empty stop the code
        if descriptions == []:
            print "The user {} pin descriptions are empty! Sorry, no products can be recommended".format(user)
            break
        dictionary = corpora.Dictionary(descriptions)
        corpus = [dictionary.doc2bow(description) for description in descriptions]
        if len(user_pins)/3 == 0:
            n_topics = 1
        else:
            n_topics = len(user_pins)/2
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=n_topics, id2word = dictionary, passes=100)
        topics = [' '.join([ldamodel.show_topics()[i][1].split('"')[j] for j in xrange(len(ldamodel.show_topics()[i][1].split('"'))) if (j%2 != 0)]) for i in xrange(len(ldamodel.show_topics()))]
        #print "Topics:"
        #print topics
        probs = [', '.join([ldamodel.show_topics()[i][1].split('"')[j].replace(' +','')[:-1] for j in xrange(len(ldamodel.show_topics()[i][1].split('"'))) if (j%2 == 0)]) for i in xrange(len(ldamodel.show_topics()))]
        proba = [[j.replace(' ','') for j in i.split(', ')][:-1] for i in probs]
        final_probs = [[float(j) for j in i] for i in proba]
        total_topics.append(topics)

        # DBSCAN
        vect = TfidfVectorizer()
        # topics vectorization
        X = vect.fit_transform(topics)
        X = X.todense()
        db, n_clusters, labels, core_samples_mask = db_scan(X, eps=0.7, min_samples=2)
        # Add exception in case clusters could not be formed because the topics were too "far away"
        if (n_clusters == 0) or (n_clusters == 1):
            eps = 0.79
            min_samples = 2
            db, n_clusters, labels, core_samples_mask = db_scan(X,eps,min_samples)
        # If clusters = 0, then it means that we have to use min_samples = 1 and have one main topic
        elif n_clusters == 0:
            min_samples = 1
            eps=0.6
            db, n_clusters, labels, core_samples_mask = db_scan(X,eps,min_samples)
        results = (db, n_clusters, labels, core_samples_mask, db.core_sample_indices_)
        user_clusters_results.append(results)
        real_labels = labels[labels+1 != 0]
        # In case all the topics are very different to each other limit to 5
        if (n_clusters == n_topics) :
            for k in xrange(5):
                clust = random.randint(1, n_topics)
                db.core_sample_indices_[real_labels == clust]
                user_clusters.append(db.core_sample_indices_[real_labels == clust])
                prefered_topic = topics[max([(final_probs[i][0],i) for i in db.core_sample_indices_[real_labels == clust]])[1]]
                user_prefered_topics.append(prefered_topic)
        else:
            # From the chosen clusters, chose the centroid (topics with highest probability)
            for clust in xrange(n_clusters):
                db.core_sample_indices_[real_labels == clust]
                user_clusters.append(db.core_sample_indices_[real_labels == clust])
                prefered_topic = topics[max([(final_probs[i][0],i) for i in db.core_sample_indices_[real_labels == clust]])[1]]
                user_prefered_topics.append(prefered_topic)

        # linear kernel - cosine similarity

        product_list = df_prod[df_prod['product_price'] <= price_limit]['image_id'].values
        product_list_description = df_prod[df_prod['product_price'] <= price_limit]['product_description'].values
        product_list_price = df_prod[df_prod['product_price'] <= price_limit]['product_price'].values
        product_list_links = df_prod[df_prod['product_price'] <= price_limit]['product_amazon_link'].values
        vect = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
        lista = []
        if len(user_prefered_topics) == 1:
            n = 6
        else:
            n = 4
        for topic in user_prefered_topics:
            docs = copy(prod_doc)
            docs.append(topic)
            X = vect.fit_transform(docs)
            cosine_similarity = linear_kernel(X, X)
            sorted_matrix = np.argsort(cosine_similarity[-1])[::-1][1:n]
            lista += list(sorted_matrix)
            lista = list(set(lista))
            if len(lista) < 5:
                continue
            else:
                break
        recommendations.append(lista)
        # Outputs the result
        print "Products recommendation:"
        print product_list[lista[:5]]
        print product_list_description[lista[:5]]
        print "Products prices:"
        print product_list_links[lista[:5]]
