import json_lines as jl
import pandas as pd
import numpy as np
import logging

from itertools import islice
from functools import partial
from collections import Counter

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def get_most_viewed(hist:list)->tuple:
    item_list = []
    for item in hist:
        if item['event_type']=='view':
            item_list.append(item['event_info'])
    try:
        return Counter(item_list).most_common(1)[0]
    except IndexError as e:
        return (None, 0)


def get_last_viewed(hist:list)->int:
    idx_hist = len(hist) - 1
    item = {'event_type': 'null'}
    while item['event_type'] != 'view' and idx_hist >= 0:
        item = hist[idx_hist]
        idx_hist -= 1
    if item['event_type'] == 'view':
        return item['event_info']
    else:
        return None


def join_item_info(df:pd.DataFrame, df_item:pd.DataFrame, col:str)->pd.DataFrame:
    df = (df
          .set_index(col)
          .join(df_item
                .add_suffix('_{}'.format(col))
                .set_index('item_id_{}'.format(col)), how='left')
          .reset_index()
          .rename(columns={'index': col}))
    return df


def remove_stopwords(s:str, stopwords:list)->str:
    domain = filter(lambda w: w not in stopwords, s.split(' '))
    return ' '.join(domain)


def preproc_domain(s:str)->str:
    if not s:
        return 'other'

    stopwords = ['electric', 'supplies', 'sets', 'covers', 'sets']
    remove_domain_stopwords = partial(remove_stopwords, stopwords=stopwords)

    domain = s.split('-')[1]
    domain = ' '.join(domain.split('_'))
    domain = domain.lower()
    domain = remove_domain_stopwords(domain)
    return domain


def preproc_search(s:str)->str:
    # TODO: improve search preprocessing
    return s.lower()


def get_last_searched(hist:list)->str:
    idx_hist = len(hist) - 1
    item = {'event_type': 'null'}
    while item['event_type'] != 'search' and idx_hist >= 0:
        item = hist[idx_hist]
        idx_hist -= 1
    if item['event_type'] == 'search':
        return preproc_search(item['event_info'])
    else:
        return ''


def get_search_cluster(s:str, df_clusters:pd.DataFrame,
                       embedder:SentenceTransformer)->int:
    s_embedding = embedder.encode([s])
    s_embedding = (s_embedding
                   /np.linalg.norm(s_embedding, axis=1, keepdims=True))
    idx = np.argmax(cosine_similarity(list(df_clusters['embedding']), s_embedding))
    return df_clusters.loc[idx, 'cluster']


def process_user_dataset(filename:str, line_batch_limit:int,
                         item_filename:str, clusters_filename:str,
                         embedder, logger)->str:
    item_usecols = ['item_id', 'price', 'condition', 'domain_id_preproc']
    df_item = pd.read_parquet(item_filename, columns=item_usecols)
    df_clusters = pd.read_parquet(clusters_filename)

    # Iteratively load train_dataset,
    # perform processing,
    # save in parquet format
    df = pd.DataFrame()
    parquet_counter = 0
    missing_id = df_item['item_id'].max() + 1

    with jl.open(filename) as f:
        list_json = take(line_batch_limit, f)
        while len(list_json) > 0:
            df = pd.DataFrame(list_json, index=range(0, len(list_json)))
            # FEATURE
            # Most viewed item
            df_most_viewed = pd.DataFrame(list(df['user_history'].apply(get_most_viewed)),
                                          columns=['most_viewed', 'times_most_viewed'])
            df = pd.concat([df, df_most_viewed], axis=1)
            df['most_viewed'] = df['most_viewed'].fillna(missing_id)
            col = 'most_viewed'
            df = join_item_info(df, df_item, col)

            # FEATURE
            # Last viewed item
            df['last_viewed'] = df['user_history'].apply(get_last_viewed)
            df['last_viewed'] = df['last_viewed'].fillna(missing_id)
            col = 'last_viewed'
            df = join_item_info(df, df_item, col)

            # FEATURE
            # Last searched item
            get_search_cluster_ = partial(get_search_cluster,
                                          df_clusters=df_clusters,
                                          embedder=embedder)
            df['last_searched'] = df['user_history'].apply(get_last_searched)
            # df['last_searched_cluster'] = df['last_searched'].apply(get_search_cluster_)

            df = df.astype({'last_viewed': 'int32',
                            'most_viewed': 'int32',
                            'condition_last_viewed': 'category',
                            'condition_most_viewed': 'category'})
            parquet_file_name = ("../../data/interim/{}_dataset_{}.parquet"
                                 .format(filename.split('/')[-1].split('_')[0],
                                         parquet_counter))
            df.drop(columns=['user_history']).to_parquet(parquet_file_name)

            list_json = take(line_batch_limit, f)
            parquet_counter += 1


    return parquet_file_name


def process_item_dataset(filename:str,
                         embedder:SentenceTransformer,
                         logger)->tuple:

    with jl.open(filename) as f:
        df_item = pd.DataFrame(f)


    # Create item clusters via word embeddings and agglomerative clustering
    df_item['domain_id_preproc'] = df_item['domain_id'].apply(preproc_domain)
    corpus = df_item['domain_id_preproc'].unique()
    corpus_embeddings = embedder.encode(corpus)
    corpus_embeddings = (corpus_embeddings
                         /np.linalg.norm(corpus_embeddings,
                                         axis=1, keepdims=True))
    embedding_mapper = {x[0]: x[1]
                        for x in
                        zip(corpus, corpus_embeddings)}

    item_file_parquet = "../../data/interim/item_data.parquet"
    df_item.to_parquet(item_file_parquet)

    # TODO: tune number of clusters
    clustering_model = AgglomerativeClustering(n_clusters=20)
    #affinity='cosine',
    #linkage='average',
    #distance_threshold=0.8)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    cluster_list = [(corpus[sentence_idx],
                     cluster,
                     embedding_mapper.get(corpus[sentence_idx]))
                      for (sentence_idx, cluster) in enumerate(cluster_assignment)]

    df_clusters = pd.DataFrame(cluster_list, columns=['domain_id_preproc', 'cluster', 'embedding'])
    clusters_file_parquet = "../../data/interim/item_domain_clusters_data.parquet"
    df_clusters.to_parquet(clusters_file_parquet)

    return (item_file_parquet, clusters_file_parquet)


def main(args):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = None

    embedder = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1')

    # ITEM DATA
    # Load item_data
    item_file = "../../data/raw/item_data.jl.gz"
    # item_filename, clusters_filename = process_item_dataset(item_file, embedder, logger)
    item_filename, clusters_filename = ("../../data/interim/item_data.parquet",
                                        "../../data/interim/item_domain_clusters_data.parquet")

    n_rows = 500_000
    # TRAIN DATA
    train_file = "../../data/raw/train_dataset.jl.gz"
    process_user_dataset(train_file, n_rows,
                         item_filename, clusters_filename, embedder, logger)

    # TEST DATA
    test_file = "../../data/raw/test_dataset.jl.gz"
    process_user_dataset(test_file, n_rows,
                         item_filename, clusters_filename, embedder, logger)


if __name__ == '__main__':
    main({})
