import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType


import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

from sim4rec.recommenders.ucb import UCB
from sim4rec.utils import pandas_to_spark

class BaseGraph:
    def __init__(self, seed=42, top_k: float=2.0):
        self.seed = seed
        np.random.seed(seed)
        self.log: Optional[DataFrame] = None
        self.scalar = StandardScaler()
        self.top_k = top_k

    def fit(self, log, user_features=None, item_features=None):
        raise NotImplemented()
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        raise NotImplemented()
    def join_log(self, log):
        # keep a running total of 
        if self.log:
            self.log.union(log.select('user_idx', 'item_idx', 'relevance'))
        else:
            self.log = log.select('user_idx', 'item_idx', 'relevance')


class LabelSpreadingRecommender(BaseGraph):
    def __init__(self, seed=42, top_k: float = 2.0, **kwargs):
        super().__init__(seed, top_k)
        self.model = LabelSpreading(**kwargs)
    def fit(self, log, user_features=None, item_features=None):
        self.join_log(log)

    def predict(self, log, k, users:DataFrame, items:DataFrame, user_features=None, item_features=None, filter_seen_items=True):
        
        #add log to logs
        self.join_log(log)
        
        # create cross of users and items, add relevance -1 to 'unseen'
        cross: pd.DataFrame = (
            users
            .join(items)
            .drop('__iter')
            .withColumn('relevance', sf.lit(-1)) # we might need to filter out seen cases in log
            .toPandas()
        )

        # join the two together and start processing
        pd_log = (
            self.log
            .join(user_features, on='user_idx')
            .join(item_features, on='item_idx')
            .drop('__iter')
            .toPandas()
        )
        pd_log = pd.concat([pd_log, cross])
        pd_log = pd.get_dummies(pd_log)
        pd_log['scaled_price'] = self.scalar.fit_transform(pd_log[['price']])
        
        y = pd_log['relevance']
        x = pd_log.drop(['relevance', 'price', 'user_idx', 'item_idx'], axis=1)
        # print(x.isna().sum())
        # print(x.describe())
        self.model.fit(x,y)

        # get probs
        cross = pd.get_dummies(cross)
        cross['scaled_price'] = self.scalar.transform(cross[['price']])
        cross['prob'] = self.model.predict_proba(cross.drop(['relevance', 'price', 'user_idx', 'item_idx'], axis=1))[:,np.where(self.model.classes_ == 1)[0][0]]
        
        # get k * top_k probabilities
        cross = (
            cross
            .groupby('user_idx')
            .apply(lambda x: x.nlargest(int(k*self.top_k), 'prob'))
            .reset_index(drop=True)
        )
        
        cross['relevance'] = cross['prob'] * cross['price']
        cross = (
            cross
            .groupby('user_idx')
            .apply(lambda x: x.nlargest(k, 'relevance'))
            .reset_index(drop=True)
            .sort_values(by=['user_idx', 'relevance'], ascending=[True, False])
        )
        
        return pandas_to_spark(cross)