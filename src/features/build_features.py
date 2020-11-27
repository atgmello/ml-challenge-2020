import json_lines as jl
import pandas as pd
import numpy as np
import logging

from os.path import exists
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


def convert_raw_to_parquet(raw_filename:str)->str:
    with jl.open(raw_filename) as f:
        list_json = [row for row in f]
        df_raw = pd.DataFrame(list_json,
                              index=range(0, len(list_json)))

    parquet_filename = (raw_filename
                        .replace('raw', 'interim')
                        .replace('jl.gz','parquet'))

    (df_raw.astype({'user_history': str}).to_parquet(parquet_filename)
     if 'user_history' in df_raw.columns
     else
     df_raw.to_parquet(parquet_filename))

    return parquet_filename


def preproc_user_history(s:str)->list:
    res = []
    try:
      res = json.loads(s.replace("'", '"'))
    except:
      res = [{'event_info':
                '100 WATERPROOF PROFISSIONAL AUDIO BLUETOOTH HEADSET EQUIPPED',
              'event_timestamp': '2019-10-01T16:58:42.553-0400',
              'event_type': 'search'}]
    return res


def get_most_viewed(hist:list, n:int=2)->tuple:
    filtered_hist = filter(lambda x: x['event_type']=='view', hist)
    item_list = list(reduce(lambda x, y:
                            x + [y['event_info']],
                            filtered_hist,
                            []))
    try:
        most_common = Counter(item_list).most_common(n)
        res = [item for tup in most_common for item in tup]
        while len(res) < 2*n:
            res.extend([None, 0])
        return res
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
    return ' '.join([w for w in nltk.word_tokenize(s.lower())
                     if not re.search('\d', w)
                     and len(w) > 2])


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


def generate_top_title(s:str, stopwords:list, n:int=20)->str:
    counter = Counter([w for w in nltk.word_tokenize(s.lower())
                      if w not in stopwords
                      and not re.search('\d', w)
                      and len(w) > 2]).most_common(n)
    title = ' '.join([w[0] for w in counter])
    return title


def get_domain_id(item:int, df_item:pd.DataFrame)->str:
    domain_id = df_item[df_item['item_id']==item]['domain_id'].values
    return domain_id[0] if len(domain_id) > 0 else None


def process_user_dataset(filename:str,
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

    missing_id = df_item['item_id'].max() + 1

    df = pd.DataFrame(list_json, index=range(0, len(list_json)))
    df['user_history'] = df['user_history'].apply(preproc_user_history)

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
    processed_filename = (filename.replace('.parquet', '_features.parquet'))

    df.drop(columns=['user_history']).to_parquet(processed_filename)

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

    df_item = pd.read_parquet(filename)

    df_item['domain_id_preproc'] = df_item['domain_id'].apply(preproc_domain)

    item_file_parquet = "../../data/interim/item_data.parquet"
    df_item.to_parquet(item_file_parquet)

    return item_file_parquet


def join_domain_title(row:list)->str:
    return (' '.join(' '.join(row['domain_id']
                              .lower()
                              .split('-')[1:])
                     .split('_'))
            + ' '
            + row['title'])


def generate_domain_data(filename:str,
                         embedder:SentenceTransformer,
                         path:str,
                         logger)->str:
    df_item = pd.read_parquet(filename)

    custom_stopwords = ['kit', '', '+', '-', 'und', 'unidade', 'unidad']
    stopwords = (nltk.corpus.stopwords.words('portuguese')
                 + nltk.corpus.stopwords.words('spanish')
                 + custom_stopwords)

    generate_top_title_ = partial(generate_top_title,
                                  stopwords=stopwords)
    df_domain = pd.DataFrame(df_item[['domain_id', 'title']]
                             .groupby(by='domain_id')
                             .agg(' '.join)
                             ['title']
                             .swifter
                             .apply(generate_top_title_)).reset_index()

    df_domain['title'] = (df_domain
                          [['domain_id','title']]
                          .swifter
                          .apply(join_domain_title, axis=1)
                          .values)

    df_domain['embedding_title'] = list(embedder.encode(list(df_domain['title'])))
    df_domain['domain_code'] = list(range(len(df_domain)))

    domain_file_parquet = path + "/data/interim/domain_data.parquet"
    df_domain.to_parquet(domain_file_parquet)

    return domain_file_parquet


def main(args):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = None

    embedder = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')

    # ITEM DATA
    # Load item_data

    parquet_item_filename = "../../data/interim/item_data.parquet"
    if not exists(parquet_item_filename):
        raw_item_filename = "../../data/raw/item_data.jl.gz"
        convert_raw_to_parquet(raw_item_filename)

    process_item_dataset(parquet_item_filename, embedder, logger)
    # parquet_item_file = "../../data/interim/item_data.parquet"

    # CLUSTERED ITEM DATA
    # parquet_item_cluster_filename = process_cluster_dataset(parquet_item_filename, embedder)

    # TRAIN DATA
    parquet_train_filename = "../../data/interim/train_dataset.parquet"
    if not exists(parquet_train_filename):
        raw_train_filename = "../../data/raw/train_dataset.jl.gz"
        convert_raw_to_parquet(raw_train_filename)

    process_user_dataset(parquet_train_filename, n_rows, embedder, logger,
                         {"parquet_item_filename":
                          parquet_item_filename})

    # TEST DATA
    parquet_test_filename = "../../data/interim/test_dataset.parquet"
    if not exists(parquet_test_filename):
        raw_test_filename = "../../data/raw/test_dataset.jl.gz"
        convert_raw_to_parquet(raw_test_filename)

    process_user_dataset(parquet_test_filename, n_rows, embedder, logger,
                         {"parquet_item_filename":
                          parquet_item_filename})

if __name__ == '__main__':
    main({})
