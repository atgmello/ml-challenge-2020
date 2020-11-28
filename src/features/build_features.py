import json_lines as jl
import pandas as pd
import numpy as np
import swifter
import logging
import time
import json
import nltk
import re

from os.path import exists
from functools import reduce
from itertools import islice, cycle
from functools import partial
from collections import Counter
from pynndescent import NNDescent

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def token_sliding_window(s:str, size:int):
    tokens = s.split(' ')
    for i in range(len(tokens) - size + 1):
        yield ' '.join(tokens[i:i+size])


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


def get_last_viewed(hist:list, n:int=2)->int:
    idx_hist = len(hist) - 1
    item_list = reversed(list(filter(lambda x:
                                     x.get("event_type")=="view",
                                     hist)))
    last_n = [x['event_info'] for x in take(n, item_list)]
    num_missing = n - len(last_n)
    last_n.extend(take(num_missing, cycle([None])))
    return last_n


def get_most_searched_ngram(hist:list, n:int=2, m:int=2)->list:
    """
    n: number of grams (bigram, trigram, ngram)
    m: number of most common
    """
    searched_items = reduce(lambda x, y:
                            x + [preproc_search(y['event_info'])] if y['event_type']=='search'
                            else x,
                            hist, [])
    searched_ngram = reduce(lambda x, y:
                            x + list(token_sliding_window(y, n)),
                            searched_items, [])
    sorted_cycle = (sorted(take(m, cycle(Counter(searched_ngram)
                                         .most_common(m))),
                           key=lambda x: x[1],
                           reverse=True))
    common_ngrams_counts = [item
                            for tup in sorted_cycle
                            for item in tup]

    num_missing = 2*m - len(common_ngrams_counts)
    if len(common_ngrams_counts) > 0:
        common_ngrams_counts.extend(take(num_missing,
                                         cycle(common_ngrams_counts)))
    else:
        common_ngrams_counts.extend(take(num_missing,
                                         cycle(['None', 0])))

    return common_ngrams_counts


def join_item_info(df:pd.DataFrame, df_item:pd.DataFrame, col:str)->pd.DataFrame:
    return (df
            .set_index(col)
            .join(df_item
                  .add_suffix('_{}'.format(col))
                  .set_index('item_id_{}'.format(col)), how='left')
            .reset_index()
            .rename(columns={'index': col}))


def remove_stopwords(s:str, stopwords:list)->str:
    return ' '.join(filter(lambda w: w not in stopwords, s.split(' ')))


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
        return None


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

    if additional_filenames.get("parquet_item_filename"):
        item_filename = additional_filenames.get("parquet_item_filename")
        item_usecols = ['item_id', 'price', 'condition', 'domain_id']
        df_item = pd.read_parquet(item_filename, columns=item_usecols)
    else:
        raise ValueError("parquet_item_filename is expected in additional_filenames")

    if additional_filenames.get("parquet_item_clusters_filename"):
        clusters_filename = additional_filenames.get("parquet_item_clusters_filename")
        df_clusters = pd.read_parquet(clusters_filename)
    else:
        df_clusters = None

    if additional_filenames.get("parquet_domain_filename"):
        domain_filename = additional_filenames.get("parquet_domain_filename")
        df_domain = pd.read_parquet(domain_filename)
    else:
        df_domain = None

    print("Reading data...")
    df = pd.read_parquet(filename)
    print("Preprocessing user_history...")
    df['user_history'] = df['user_history'].swifter.apply(preproc_user_history)
    df = df.rename_axis('user_id').reset_index()

    start = time.time()

    # FEATURE
    # Get the domain_id for the bought_item
    # It can be used as a new target
    if 'item_bought' in df.columns:
        print("Feature\nBought item domain_id...")
        df = (df.set_index('item_bought')
              .join(df_item[['item_id','domain_id']]
                    .set_index('item_id'),
                    how='left')
              .rename_axis('item_bought')
              .reset_index()
              .rename(columns={'domain_id': 'domain_id_item_bought'}))

    print("Feature\nMost viewed item...")
    # FEATURE
    # Most viewed item
    n_most = 2
    cols_feat_most_viewed = [c for n in range(1, n_most+1)
                             for c in
                             [f'most_viewed_{n}',
                              f'most_viewed_count_{n}']]
    df[cols_feat_most_viewed] = list(df['user_history'].swifter.apply(get_most_viewed))

    for i in range(1, n_most+1):
        # Join to get more information about the viewed item
        df = join_item_info(df, df_item, f'most_viewed_{i}')

        # Fills NaN with the last most viewed item
        if i > 1:
          df[f'most_viewed_{i}'] = (df[f'most_viewed_{i}']
                                    .fillna(df[f'most_viewed_{i-1}']))

    print("Feature\nLast viewed item...")
    # FEATURE
    # Last viewed item
    get_last_viewed_ = partial(get_last_viewed, n=n_last)
    df[cols_feat_last_viewed] = list(df['user_history'].swifter.apply(get_last_viewed_))
    n_last_viewed = 2
    cols_feat_last_viewed = [f'last_viewed_{i}' for i in range(1, n_last_viewed+1)]

    for c in cols_feat_last_viewed:
        df = join_item_info(df, df_item, c)

    print("Feature\nLast searched item...")
    # FEATURE
    # Last searched item
    idx_missing = df[cols_feat_last_viewed[0]].isna().values
    df['last_searched'] = None
    df.loc[idx_missing,'last_searched'] = (df.loc[idx_missing, 'user_history']
                                            .swifter.apply(get_last_searched))
    idx_missing = idx_missing & ~df['last_searched'].isna().values

    df['last_searched_embedding'] = None
    df.loc[idx_missing,'last_searched_embedding'] = [[x]
                                                     for x in
                                                     (embedder
                                                      .encode(list(df.loc[idx_missing,
                                                                          'last_searched'])))]

    print("Feature\nLast searched item domain...")
    # FEATURE
    # Last searched item domain
    print("Building index...")
    data = np.array([np.array(x) for x in df_domain['embedding_title'].values])
    index = NNDescent(data, metric='cosine')
    print("Querying data...")
    query_data = np.array([np.array(x[0])
                           for x in
                           df.loc[idx_missing,
                                  'last_searched_embedding'].values])
    closest_domain = index.query(query_data, k=1)
    df['last_searched_domain'] = None
    df.loc[idx_missing,'last_searched_domain'] = [df_domain.loc[idx[0],'domain_id']
                                                  for idx in closest_domain[0]]
    df['last_searched_domain_distance'] = None
    df.loc[idx_missing,'last_searched_domain_distance'] = [dist[0]
                                                           for dist in
                                                           closest_domain[1]]

    print("Feature\nMost searched ngrams...")
    # FEATURE
    # Most searched ngrams
    n_most = 2
    cols_feat_most_searched = [c for n in range(1, n_most+1)
                               for c in
                               [f'most_searched_ngram_{n}',
                                f'most_searched_ngram_count_{n}']]
    df[cols_feat_most_searched] = None
    get_most_searched_ngram_ = partial(get_most_searched_ngram, m=n_most)
    df.loc[idx_missing, cols_feat_most_searched] = list(df.loc[idx_missing,'user_history']
                                                        .swifter.apply(get_most_searched_ngram_))

    cols_search = [f'most_searched_ngram_{n}' for n in range(1, n_most+1)]
    for c in cols_search:
        df[f'{c}_embedding'] = None
        df.loc[idx_missing,f'{c}_embedding'] = [[x]
                                                for x in
                                                (embedder.
                                                 encode(list(df.loc[idx_missing,c])))]

    print("Feature\nMost searched ngrams domains...")
    # FEATURE
    # Most searched ngrams domain
    for c in cols_search:
        df[f'domain_id_{c}'] = None
        df[f'domain_id_{c}_distance'] = None
        query_data = np.array([np.array(x[0])
                               for x in
                               df.loc[idx_missing,f'{c}_embedding'].values])
        closest_domain = index.query(query_data, k=1)
        df.loc[idx_missing,f'domain_id_{c}'] = [df_domain.loc[idx[0],'domain_id']
                                                for idx in closest_domain[0]]
        df.loc[idx_missing,f'domain_id_{c}_distance'] = [dist[0]
                                                         for dist in
                                                         closest_domain[1]]

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
        df['last_searched_cluster'] = df['last_searched'].swifter.apply(get_search_cluster_)

    # Saving results
    processed_filename = (filename.replace('.parquet', '_features.parquet'))

    print(time.time() - start)
    df.drop(columns=['user_history']).to_parquet(processed_filename)

    return processed_filename


def process_cluster_dataset(item_file:str, embedder, path:str)->str:
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

    clusters_file_parquet = path + "/data/interim/item_domain_clusters_data.parquet"
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
                         path:str,
                         logger)->str:
    df_item = pd.read_parquet(filename)

    domain_mapper = {x[1]: x[0]
                     for x in
                     enumerate(sorted(df_item['domain_id'].dropna().unique()))}
    df_item['domain_code'] = df_item['domain_id'].map(domain_mapper)

    item_file_parquet = path + "/data/interim/item_data.parquet"
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

    nltk.download('stopwords')
    nltk.download('punkt')

    if args.embedder=='bert':
        embedder = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')
    else:
        embedder = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1')

    if args.environment=='colab':
        path = './drive/MyDrive/ml-data-challange-2020'
    else:
        path = '../../'

    # ITEM DATA
    # Load item_data
    parquet_item_filename = path + "/data/interim/item_data.parquet"
    if not exists(parquet_item_filename):
        raw_item_filename = path + "/data/raw/item_data.jl.gz"
        convert_raw_to_parquet(raw_item_filename)

    process_item_dataset(parquet_item_filename, embedder, path, logger)
    # parquet_item_file = "../../data/interim/item_data.parquet"

    # CLUSTERED ITEM DATA
    # parquet_item_cluster_filename = process_cluster_dataset(parquet_item_filename,
    #                                                         embedder, path)

    # DOMAIN ITEM DATA
    parquet_domain_filename = path + "/data/interim/domain_data.parquet"
    if not exists(parquet_domain_filename):
        generate_domain_data(parquet_item_filename, embedder, path, logger)

    # TEST DATA
    parquet_test_filename = path + "/data/interim/test_dataset.parquet"
    if not exists(parquet_test_filename):
        raw_test_filename = path + "/data/raw/test_dataset.jl.gz"
        convert_raw_to_parquet(raw_test_filename)

    process_user_dataset(parquet_test_filename, embedder, logger,
                         {"parquet_item_filename":
                          parquet_item_filename,
                          "parquet_domain_filename":
                          parquet_domain_filename})

    # TRAIN DATA
    parquet_train_filename = path + "/data/interim/train_dataset.parquet"
    if not exists(parquet_train_filename):
        raw_train_filename = path + "/data/raw/train_dataset.jl.gz"
        convert_raw_to_parquet(raw_train_filename)

    process_user_dataset(parquet_train_filename, embedder, logger,
                         {"parquet_item_filename":
                          parquet_item_filename,
                          "parquet_domain_filename":
                          parquet_domain_filename})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    env_msg = """
    Sets the environment where the code is going to run. Accepts 'local' or 'colab'.
    """
    parser.add_argument("--environment", help=env_msg,
                        choices = ['local', 'colab'],
                        default='colab')

    embedder_msg = """
    Pre-trained model to be used. Either 'bert' or 'roberta'
    """
    parser.add_argument("--embedder", help=embedder_msg,
                        choices = ['bert', 'roberta'],
                        default = 'bert')

    args = parser.parse_args()
    main(args)
