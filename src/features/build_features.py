import json_lines as jl
import pandas as pd
import numpy as np
import logging

from functools import reduce
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

    domain = s.split('-')[1]
    domain = ' '.join(domain.split('_'))
    domain = domain.lower()
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


def get_search_cluster(s:str, domain_clusters:list,
                       embedder:SentenceTransformer)->int:
    s_embedding = embedder.encode([s])
    s_embedding = (s_embedding
                   /np.linalg.norm(s_embedding, axis=1, keepdims=True))
    cluster = np.argmax(cosine_similarity(domain_clusters, s_embedding))
    return cluster


def mean_embedding(l:list)->float:
  all_embeddings = reduce(lambda x,y: np.vstack([x, np.asarray(y)]), l)
  return list(all_embeddings.mean(axis=0))


def process_user_dataset(filename:str, line_batch_limit:int,
                         embedder, logger,
                         additional_filenames:dict)->str:

    if (item_filename:=additional_filenames.get("parquet_item_filename")):
        item_usecols = ['item_id', 'price', 'condition', 'domain_id_preproc']
        df_item = pd.read_parquet(item_filename, columns=item_usecols)
    else:
        raise ValueError("parquet_item_filename is expected in additional_filenames")

    if (clusters_filename:=additional_filenames.get("parquet_item_clusters_filename")):
        df_clusters = pd.read_parquet(clusters_filename)
    else:
        df_clusters = None

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
            df['last_searched'] = df['user_history'].apply(get_last_searched)

            # FEATURE
            # Last searched item cluster
            if df_clusters:
                domain_clusters = list(df_clusters
                                       [['cluster','embedding_cluster']]
                                       .groupby(by='cluster')
                                       .nth(-1)
                                       .reset_index()
                                       .sort_values(by='cluster')
                                       ['embedding_cluster']
                                       .values)
                get_search_cluster_ = partial(get_search_cluster,
                                              domain_clusters=domain_clusters,
                                              embedder=embedder)
                df['last_searched_cluster'] = df['last_searched'].apply(get_search_cluster_)

            # Saving results
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


def process_cluster_dataset(item_file:str, embedder)->str:
    col = 'domain_id_preproc'
    corpus = (pd.read_parquet(item_file, columns=[col])
              [col].unique())
    corpus_embeddings = embedder.encode(corpus)
    corpus_embeddings = (corpus_embeddings
                         /np.linalg.norm(corpus_embeddings,
                                         axis=1, keepdims=True))
    embedding_mapper = {x[0]: x[1]
                        for x in
                        zip(corpus, corpus_embeddings)}

    # TODO: tune number of clusters
    n_clusters = 50
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    #affinity='cosine',
    #linkage='average',
    #distance_threshold=0.8)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    cluster_list = [(corpus[sentence_idx],
                     cluster,
                     embedding_mapper.get(corpus[sentence_idx]))
                      for (sentence_idx, cluster) in enumerate(cluster_assignment)]

    clusters_file_parquet = "../../data/interim/item_domain_clusters_data.parquet"
    df_clusters = pd.DataFrame(cluster_list,
                               columns=['domain_id_preproc',
                                        'cluster', 'embedding_domain'])

    df_clusters = (df_clusters.set_index('cluster')
                   .join(df_clusters
                         [['cluster','embedding_domain']]
                         .groupby(by='cluster')
                         .agg({'embedding_domain': mean_embedding})
                         .rename(columns={'embedding_domain': 'embedding_cluster'}),
                         how='left')
                   .reset_index())

    df_clusters.to_parquet(clusters_file_parquet)

    return clusters_file_parquet


def process_item_dataset(filename:str,
                         embedder:SentenceTransformer,
                         logger)->str:

    with jl.open(filename) as f:
        df_item = pd.DataFrame(f)

    df_item['domain_id_preproc'] = df_item['domain_id'].apply(preproc_domain)

    item_file_parquet = "../../data/interim/item_data.parquet"
    df_item.to_parquet(item_file_parquet)

    return item_file_parquet


def main(args):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = None

    embedder = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')

    # ITEM DATA
    # Load item_data
    raw_item_filename = "../../data/raw/item_data.jl.gz"
    parquet_item_filename = process_item_dataset(raw_item_filename, embedder, logger)
    # parquet_item_file = "../../data/interim/item_data.parquet"

    # CLUSTERED ITEM DATA
    # parquet_item_cluster_filename = process_cluster_dataset(parquet_item_filename, embedder)

    n_rows = 500_000
    # TRAIN DATA
    raw_train_filename = "../../data/raw/train_dataset.jl.gz"
    process_user_dataset(raw_train_filename, n_rows, embedder, logger,
                         {"parquet_item_filename":
                          parquet_item_filename})

    # TEST DATA
    raw_test_filename = "../../data/raw/test_dataset.jl.gz"
    process_user_dataset(raw_test_filename, n_rows, embedder, logger,
                         {"parquet_item_filename":
                          parquet_item_filename})

if __name__ == '__main__':
    main({})
