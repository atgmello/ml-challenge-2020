import pandas as pd
import numpy as np
import logging
import swifter
import click
import gzip
import json
import os

from functools import partial, reduce
from src.models import ndcg_score
from collections import Counter
from datetime import datetime


def predict_simple(row:pd.Series, cols_domain:list,
                   cols_item:list, df_most_bought:pd.DataFrame,
                   available_domains:list,
                   most_bought_items:list)->list:
    """
    No ordering on the domains/items.

    It's important that `cols_item` and `cols_domain` are aligned.
    i.e. if the first elem from `cols_item` is `last_viewed_1`
    then the first elem from `cols_domain` should be `domain_id_last_viewed_1`.
    """
    valid_domains = [d for d in row[cols_domain].unique()
                     if d in available_domains]

    pred_list = list(set(reduce(lambda x, y: x + [row[y]]
                                if not np.isnan(row[y]) else x,
                                cols_item, [])))

    # Interleave top 10 items from each viewed/searched domain
    # and then flatten. We use this in order to recommend the
    # top items from the viewed/searched domains.
    top_items = [i
                 for items in
                 zip(*[df_most_bought.loc[c]
                       .head(10).index.values
                       for c in valid_domains])
                 for i in items]
    num_missing_items = 10 - len(pred_list)
    pred_list.extend(top_items[:num_missing_items])

    # In case we have not reached 10 items in our recomendation
    # list, we just return the top bought items overall.
    num_missing_items = 10 - len(pred_list)
    pred_list.extend(most_bought_items[:num_missing_items])

    pred_list = [int(x) for x in pred_list]

    return pred_list


def predict_vote(row:pd.Series, cols_domain:list,
                 cols_item:list, df_most_bought:pd.DataFrame,
                 available_domains:list,
                 most_bought_items:list)->list:
    """
    No ordering on the domains/items;
    With voting.

    It's important that `cols_item` and `cols_domain` are aligned.
    i.e. if the first elem from `cols_item` is `last_viewed_1`
    then the first elem from `cols_domain` should be `domain_id_last_viewed_1`.
    """
    valid_domains = [d for d in row[cols_domain]
                     if d in available_domains]
    try:
        top_domain = Counter(valid_domains).most_common(1)[0][0]
    except IndexError as e:
        top_domain = 'MLB-CELLPHONES'

    pred_list = list(set(reduce(lambda x, y: x + [row[y]]
                                if not np.isnan(row[y]) else x,
                                cols_item, [])))

    # Interleave top 10 items from each viewed/searched domain
    # and then flatten. We use this in order to recommend the
    # top items from the viewed/searched domains.
    top_items = (df_most_bought.loc[top_domain]
                 .head(10).index.values)

    num_missing_items = 10 - len(pred_list)
    pred_list.extend(top_items[:num_missing_items])

    # In case we have not reached 10 items in our recomendation
    # list, we just return the top bought items overall.
    num_missing_items = 10 - len(pred_list)
    pred_list.extend(most_bought_items[:num_missing_items])

    pred_list = [int(x) for x in pred_list]

    return pred_list


def predict_ordered(row:pd.Series, cols_domain:list,
                    cols_item:list, df_most_bought:pd.DataFrame,
                    available_domains:list,
                    most_bought_items:list)->list:
    """
    Order domain/items by domains with most sold items.

    It's important that `cols_item` and `cols_domain` are aligned.
    i.e. if the first elem from `cols_item` is `last_viewed_1`
    then the first elem from `cols_domain` should be `domain_id_last_viewed_1`.
    """
    valid_domains = [d for d in row[cols_domain].unique()
                     if d in available_domains]
    num_bought_domain = [df_most_bought.loc[v,'index_sum'].values[0]
                         for v in valid_domains]

    sorted_items = sorted(zip(row[cols_item],num_bought_domain),
                          key=lambda t: t[1], reverse=True)
    pred_list = list(filter(lambda i: not np.isnan(i),
                            set([x[0] for x in sorted_items])))

    # Interleave top 10 items from each viewed/searched domain
    # and then flatten. We use this in order to recommend the
    # top items from the viewed/searched domains.

    sorted_domains = [d[0]
                      for d in
                      sorted(zip(valid_domains, num_bought_domain),
                             key=lambda t: t[1], reverse=True)]
    top_items = [i
                 for items in
                 zip(*[df_most_bought.loc[c]
                       .head(10).index.values
                       for c in valid_domains])
                 for i in items]
    num_missing_items = 10 - len(pred_list)
    pred_list.extend(top_items[:num_missing_items])

    # In case we have not reached 10 items in our recomendation
    # list, we just return the top bought items overall.
    num_missing_items = 10 - len(pred_list)
    pred_list.extend(most_bought_items[:num_missing_items])

    pred_list = [int(x) for x in pred_list]

    return pred_list


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('prediction_policy', type=click.STRING, default='vote')
def make_prediction(input_filepath:str,
                    output_filepath:str,
                    prediction_policy:str):

    logger = logging.getLogger(__name__)
    logger.info('Loading data...')

    train_filename = 'train_dataset_features.parquet'
    test_filename = 'test_dataset_features.parquet'

    cols_load = ['item_bought', 'domain_id_item_bought']
    cols_feat_domain = []

    cols_item = [f'{i}_viewed_item_{j}' for i in ['most','last'] for j in range(1,3)]
    cols_load.extend(cols_item)

    cols_item_domain = [f'domain_id_{c}' for c in cols_item]
    cols_load.extend(cols_item_domain)
    cols_feat_domain.extend(cols_item_domain)

    cols_domain = [f'most_viewed_domain_{i}' for i in range(1,3)]
    cols_load.extend(cols_domain)
    cols_feat_domain.extend(cols_domain)

    cols_ngram_domain = [f'domain_id_most_searched_ngram_{i}' for i in range(1,3)]
    cols_load.extend(cols_ngram_domain)
    cols_feat_domain.extend(cols_ngram_domain)

    cols_searched_domain = [f'domain_id_last_searched_{i}' for i in range(1,3)]
    cols_load.extend(cols_searched_domain)
    cols_feat_domain.extend(cols_searched_domain)

    cols_feat_domain.extend(['domain_id_forest'])

    df_train = pd.read_parquet(os.path.join(input_filepath, train_filename),
                               columns=cols_load)
    df_test = pd.read_parquet(os.path.join(input_filepath, test_filename),
                              columns=cols_load[2:]+['user_id', 'domain_id_forest'])

    logger.info('Creating helper intermediate results...')
    df_most_bought = (df_train[['domain_id_item_bought','item_bought']]
                      .reset_index()
                      .groupby(by=['domain_id_item_bought','item_bought'])
                      .count()
                      .sort_values(by=['domain_id_item_bought','index'], ascending=False))

    # Add information about the number of items bought per domain
    df_most_bought = df_most_bought.join(df_most_bought
                                         .reset_index()[['domain_id_item_bought','index']]
                                         .groupby(by='domain_id_item_bought')
                                         .sum()
                                         .sort_values(by='index', ascending=False),
                                         how='left', rsuffix='_sum')

    most_bought_items = [i[0]
                         for i in
                         (df_most_bought
                          .sort_values(by='index', ascending=False)
                          .head(10).values)]

    available_domains = (df_most_bought
                         .reset_index()
                         ['domain_id_item_bought']
                         .unique())

    pred_dict = {'ordered': predict_ordered,
                 'simple': predict_simple,
                 'vote': predict_vote}

    predict_ = partial(pred_dict[prediction_policy],
                       cols_domain=cols_feat_domain,
                       cols_item=cols_item,
                       df_most_bought=df_most_bought,
                       available_domains=available_domains,
                       most_bought_items=most_bought_items)

    df_test = df_test.set_index('user_id').sort_index()

    logger.info("Predicting with '%s' heuristic...", prediction_policy)
    y_pred = df_test.apply(predict_, axis=1).values
    df_y_pred = pd.DataFrame(list(y_pred))
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    submission_filename = f'{prediction_policy}_{now}.csv'

    logger.info('Saving results...')
    df_y_pred.to_csv(os.path.join(output_filepath, submission_filename),
                     index=False, header=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    make_prediction()
