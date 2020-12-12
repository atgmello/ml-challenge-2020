import json_lines as jl
import pandas as pd
import numpy as np
import swifter
import logging
import time
import json
import nltk
import re
import os

from functools import reduce
from itertools import islice, cycle
from functools import partial
from collections import Counter
from pynndescent import NNDescent

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


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


def get_most_viewed(hist:list, item_domain_mapper:dict=None,
                           n:int=2)->str:
    viewed_list = list(reduce(lambda x, y:
                              x + [y['event_info']]
                              if y['event_type']=='view'
                              else x,
                              hist, []))
    try:
        if item_domain_mapper:
            viewed_list = [item_domain_mapper[x]
                           for x in viewed_list]
        most_common = Counter(viewed_list).most_common(n)
        most_common_flat = [item for tup in most_common for item in tup]
        while len(most_common_flat) < 2*n:
            most_common_flat.extend([None, 0])
        return most_common_flat
    except IndexError as e:
        return (None, 0)


def get_last(hist:list, n:int, event_type:str)->int:
    item_list = reversed(list(filter(lambda x:
                                     x.get("event_type")==event_type,
                                     hist)))
    last_n = ([x['event_info']
               for x in take(n, item_list)]
              if event_type=="view"
              else
              [preproc_search(x['event_info'])
               for x in take(n, item_list)])

    num_missing = n - len(last_n)
    (last_n.extend(take(num_missing, cycle([None])))
     if event_type=="view"
     else
     last_n.extend(take(num_missing, cycle(last_n))))
    return last_n


def get_most_searched_ngram(hist:list, n:int=2, m:int=2)->list:
    """
    n: number of grams (bigram, trigram, ngram)
    m: number of most common
    """
    searched_items = reduce(lambda x, y:
                            x + [preproc_search(y['event_info'])]
                            if y['event_type']=='search'
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


def generate_top_title(s:str, stopwords:list, n:int=10)->str:
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
                         output_filepath:str,
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

    if logger: logger.info("Reading data...")
    df = pd.read_parquet(filename)
    if logger: logger.info("Preprocessing user_history...")
    df['user_history'] = df['user_history'].swifter.apply(preproc_user_history)
    df = df.rename_axis('user_id').reset_index()

    start = time.time()

    # FEATURE
    # Get the domain_id for the bought_item
    # It can be used as a new target
    if 'item_bought' in df.columns:
        if logger: logger.info("Feature\nBought item domain_id...")
        df = (df.set_index('item_bought')
              .join(df_item[['item_id','domain_id']]
                    .set_index('item_id'),
                    how='left')
              .rename_axis('item_bought')
              .reset_index()
              .rename(columns={'domain_id': 'domain_id_item_bought'}))

    if logger: logger.info("Feature\nMost viewed item...")
    # FEATURE
    # Most viewed item
    n_most_item = 2
    cols_feat_most_viewed_item = [c for n in range(1, n_most_item+1)
                                  for c in
                                  [f'most_viewed_item_{n}',
                                   f'count_most_viewed_item_{n}']]
    get_most_viewed_item = partial(get_most_viewed,
                                    n=n_most_item)
    df[cols_feat_most_viewed_item] = list(df['user_history']
                                          .swifter.apply(get_most_viewed_item))

    for i in range(1, n_most_item+1):
        # Join to get more information about the viewed item
        df = join_item_info(df, df_item, f'most_viewed_item_{i}')

        # Fills NaN with the last most viewed item
        if i > 1:
            df[f'most_viewed_item_{i}'] = (df[f'most_viewed_item_{i}']
                                           .fillna(df[f'most_viewed_item_{i-1}']))

    # FEATURE
    # Most viewed domain
    if logger: logger.info("Feature\nMost viewed domain...")
    n_most_domain = 2
    cols_feat_most_viewed_domain = [c for n in range(1, n_most_domain+1)
                                    for c in
                                    [f'most_viewed_domain_{n}',
                                     f'count_most_viewed_domain_{n}']]
    item_domain_mapper = {x[0]: x[1]
                          for x in df_item[['item_id','domain_id']].values}
    get_most_viewed_domain = partial(get_most_viewed,
                                      item_domain_mapper=item_domain_mapper,
                                      n=n_most_domain)
    df[cols_feat_most_viewed_domain] = list(df['user_history']
                                            .swifter.apply(get_most_viewed_domain))

    if logger: logger.info("Feature\nLast viewed item...")
    # FEATURE
    # Last viewed item
    n_last_viewed = 2
    get_last_viewed = partial(get_last, n=n_last_viewed, event_type="view")
    cols_feat_last_viewed = [f'last_viewed_item_{i}' for i in range(1, n_last_viewed+1)]
    df[cols_feat_last_viewed] = list(df['user_history'].swifter.apply(get_last_viewed))

    for c in cols_feat_last_viewed:
        df = join_item_info(df, df_item, c)

    if logger: logger.info("Feature\nLast searched item...")
    # FEATURE
    # Last searched item
    n_last_searched = 2
    idx_missing = df[cols_feat_last_viewed[0]].isna().values
    get_last_searched = partial(get_last, n=n_last_searched, event_type="search")
    cols_feat_last_searched = [f'last_searched_{i}' for i in range(1, n_last_searched+1)]
    df[cols_feat_last_searched] = None
    df.loc[idx_missing,cols_feat_last_searched] = list(df.loc[idx_missing, 'user_history']
                                                        .swifter.apply(get_last_searched))
    # Domain knowledge for dealing with blank strings:
    # replace with `cellphones` which is the domain with
    # the most bought items.
    for c in cols_feat_last_searched:
        df.loc[idx_missing,c] = (df.loc[idx_missing,c]
                                 .apply(lambda s: 'cellphones'
                                        if s==''
                                        else s))
    idx_missing = idx_missing & ~df[cols_feat_last_searched[0]].isna().values

    cols_feat_last_searched_emb = [f'last_searched_embedding_{i}' for i in range(1, n_last_searched+1)]
    df[cols_feat_last_searched_emb] = None
    embeddings = embedder.encode(df.loc[idx_missing,cols_feat_last_searched]
                                 .values
                                 .reshape(1, -1)[0])
    for i,c in enumerate(cols_feat_last_searched_emb):
        df.loc[idx_missing, c] = [[x] for x in
                                  embeddings[i::len(cols_feat_last_searched_emb)]]

    if logger: logger.info("Feature\nLast searched item domain...")
    # FEATURE
    # Last searched item domain
    if logger: logger.info("Building index...")
    data = np.array([np.array(x) for x in df_domain['embedding_title'].values])
    index = NNDescent(data, metric='cosine')
    if logger: logger.info("Querying data...")
    query_data = np.array([np.array(x[0])
                           for x in
                           (df.loc[idx_missing,
                                   cols_feat_last_searched_emb]
                            .values
                            .reshape(1, -1))[0]])
    closest_domain = index.query(query_data, k=1)
    cols_feat_last_searched_domain = [f'domain_id_last_searched_{i}'
                                      for i in range(1, n_last_searched+1)]
    df[cols_feat_last_searched_domain] = None
    df.loc[idx_missing,cols_feat_last_searched_domain] = np.reshape([df_domain.loc[idx[0],'domain_id']
                                                                     for idx in closest_domain[0]],
                                                                    (sum(idx_missing),
                                                                     len(cols_feat_last_searched_domain)))

    cols_feat_last_searched_dist = [f'domain_id_last_searched_distance_{i}'
                                    for i in range(1, n_last_searched+1)]
    df[cols_feat_last_searched_domain] = None
    df.loc[idx_missing,cols_feat_last_searched_dist] = np.reshape([dist[0]
                                                                   for dist in closest_domain[1]],
                                                                  (sum(idx_missing),
                                                                   len(cols_feat_last_searched_domain)))

    if logger: logger.info("Feature\nMost searched ngrams...")
    # FEATURE
    # Most searched ngrams
    n_most_ngrams = 2
    cols_feat_most_searched = [c for n in range(1, n_most_ngrams+1)
                               for c in
                               [f'most_searched_ngram_{n}',
                                f'count_most_searched_ngram_{n}']]
    df[cols_feat_most_searched] = None
    get_most_searched_ngram_ = partial(get_most_searched_ngram, m=n_most_ngrams)
    df.loc[idx_missing, cols_feat_most_searched] = list(df.loc[idx_missing,'user_history']
                                                        .swifter.apply(get_most_searched_ngram_))

    cols_search = [f'most_searched_ngram_{n}' for n in range(1, n_most_ngrams+1)]
    for c in cols_search:
        df[f'{c}_embedding'] = None
        df.loc[idx_missing,f'{c}_embedding'] = [[x]
                                                for x in
                                                (embedder.
                                                 encode(list(df.loc[idx_missing,c])))]

    if logger: logger.info("Feature\nMost searched ngrams domains...")
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
    processed_filename = os.path.join(output_filepath,
                                      (filename.split('/')[-1]
                                       .replace('.parquet',
                                                '_features.parquet')))

    if logger: logger.info(f"Elapsed time: {time.time() - start}")
    if logger: logger.info("Saving file...")
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

    clusters_file_parquet = os.path.join(path, "item_domain_clusters_data.parquet")
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
                         output_filepath:str,
                         logger=None)->str:
    df_item = pd.read_parquet(filename)

    if logger: logger.info("Generating domain codes...")
    domain_mapper = {x[1]: x[0]
                     for x in
                     enumerate(sorted(df_item['domain_id'].dropna().unique()))}
    df_item['domain_code'] = df_item['domain_id'].map(domain_mapper)

    if logger: logger.info("Saving file...")
    item_file_parquet = os.path.join(output_filepath, "item_data.parquet")
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
                         output_filepath:str,
                         logger=None)->str:
    df_item = pd.read_parquet(filename)

    if logger: logger.info("Building stopword list...")
    custom_stopwords = ['kit', '', '+', '-', 'und', 'unidade', 'unidad']
    stopwords = (nltk.corpus.stopwords.words('portuguese')
                 + nltk.corpus.stopwords.words('spanish')
                 + custom_stopwords)

    if logger: logger.info("Generating title aggregation...")
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

    if logger: logger.info("Embedding aggregated titles...")
    df_domain['embedding_title'] = list(embedder.encode(list(df_domain['title'])))

    if logger: logger.info("Generating domain codes...")
    df_domain['domain_code'] = list(range(len(df_domain)))

    if logger: logger.info("Saving file...")
    domain_file_parquet = os.path.join(output_filepath, "domain_data.parquet")
    df_domain.to_parquet(domain_file_parquet)

    return domain_file_parquet


def enhance_datasets(input_filepath:str,
                     output_filepath:str,
                     logger=None,
                     embedder_name:str='roberta'):
    nltk.download('stopwords')
    nltk.download('punkt')

    intermediate_filepath = input_filepath.replace('raw', 'interim')

    if embedder_name=='bert':
        embedder = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')
    else:
        embedder = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1')

    # ITEM DATA
    # Load item_data
    parquet_item_filename = os.path.join(intermediate_filepath, "item_data.parquet")
    if not os.path.exists(parquet_item_filename):
        if logger: logger.info("Generating intermediate item_data...")
        raw_item_filename = os.path.join(input_filepath, "item_data.jl.gz")
        convert_raw_to_parquet(raw_item_filename)

    if logger: logger.info("Enhancing item_data...")
    process_item_dataset(parquet_item_filename, output_filepath, logger)

    # DOMAIN ITEM DATA
    parquet_domain_filename = os.path.join(output_filepath, "domain_data.parquet")
    if not os.path.exists(parquet_domain_filename):
        if logger: logger.info("Generating domain_data...")
        generate_domain_data(parquet_item_filename, embedder, output_filepath, logger)

    # TRAIN DATA
    parquet_train_filename = os.path.join(intermediate_filepath, "train_dataset.parquet")
    if not os.path.exists(parquet_train_filename):
        if logger: logger.info("Generating intermediate train_dataset...")
        raw_train_filename = os.path.join(input_filepath, "train_dataset.jl.gz")
        convert_raw_to_parquet(raw_train_filename)

    if logger: logger.info("Enhancing train_dataset...")
    process_user_dataset(parquet_train_filename,
                         output_filepath,
                         embedder, logger,
                         {"parquet_item_filename":
                          parquet_item_filename,
                          "parquet_domain_filename":
                          parquet_domain_filename})

    # TEST DATA
    parquet_test_filename = os.path.join(intermediate_filepath, "test_dataset.parquet")
    if not os.path.exists(parquet_test_filename):
        if logger: logger.info("Generating intermediate test_dataset...")
        raw_test_filename = os.path.join(input_filepath, "test_dataset.jl.gz")
        convert_raw_to_parquet(raw_test_filename)

    if logger: logger.info("Enhancing test_dataset...")
    process_user_dataset(parquet_test_filename,
                         output_filepath,
                         embedder, logger,
                         {"parquet_item_filename":
                          parquet_item_filename,
                          "parquet_domain_filename":
                          parquet_domain_filename})

    if logger: logger.info("All done!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    env_msg = """
    Sets the environment where the code is going to run. Accepts 'local' or 'colab'.
    """
    parser.add_argument("--environment", help=env_msg,
                        choices = ['local', 'colab'],
                        default='local')

    embedder_msg = """
    Pre-trained model to be used. Either 'bert' or 'roberta'
    """
    parser.add_argument("--embedder", help=embedder_msg,
                        choices = ['bert', 'roberta'],
                        default = 'roberta')

    args = parser.parse_args()

    if args.environment=='colab':
        input_filepath = './drive/MyDrive/ml-data-challange-2020/data/raw'
    else:
        input_filepath = './data/raw'

    output_filepath = input_filepath.replace('raw', 'processed')

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    enhance_datasets(input_filepath, output_filepath,
                     logger, args.embedder)
