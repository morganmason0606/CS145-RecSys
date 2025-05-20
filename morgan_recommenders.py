import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

from sim4rec.recommenders.ucb import UCB

class BaseRecommender:
    def __init__(self, seed=None):
        self.seed = seed
        np.random.seed(seed)
        self.log = None
    def fit(self, log, user_features=None, item_features=None):
        """
        No training needed for random recommender.
        
        Args:
            log: Interaction log
            user_features: User features (optional)
            item_features: Item features (optional)
        """
        # No training needed
        raise NotImplemented()
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        raise NotImplemented()


    
import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sim4rec.utils import pandas_to_spark
class LRRecommender(BaseRecommender):
    def __init__(self, seed=None, C=1.0):
        super().__init__(seed)
        self.model = LogisticRegression(
            penalty='l2', 
            C=C
        )
        self.scalar = StandardScaler()
    def preprocess_data(self, log, user_features, item_features): 
        pd_log = log.join(
            user_features, 
            on='user_idx'
        ).join(
            item_features, 
            on='item_idx'
        ).drop(
            'user_idx', 'item_idx', '__iter'
        ).toPandas()

        pd_log = pd.get_dummies(pd_log)
        pd_log['scaled_price'] = self.scalar.fit_transform(pd_log[['price']])
        return pd_log
    def fit(self, log:DataFrame, user_features=None, item_features=None):
        if self.log:
            self.log.union(log.select('user_idx', 'item_idx', 'relevance'))
        else:
            self.log = log.select('user_idx', 'item_idx', 'relevance')
        
        if user_features and item_features:
            pd_log = self.preprocess_data(log, user_features, item_features)

            y = pd_log['relevance']
            x = pd_log.drop(['relevance', 'price'], axis=1)

            self.model.fit(x,y)
    def predict(self, log, k, users:DataFrame, items:DataFrame, user_features=None, item_features=None, filter_seen_items=True):
        cross = (
            users
            .join(items)
            .drop('__iter')
            .toPandas()
        )
        cross = pd.get_dummies(cross)
        cross['scaled_price'] = self.scalar.transform(cross[['price']])

        cross['prob'] = self.model.predict_proba(cross.drop(['user_idx', 'item_idx', 'price'], axis=1))[:,np.where(self.model.classes_ == 1)[0][0]]
        
        cross = (
            cross
            .sort_values(by=['user_idx', 'prob'], ascending=[True, False])
            .groupby('user_idx')
            .head(2*k)
        )
        cross['relevance'] = cross['prob'] * cross["price"] 
        cross = (
            cross
            .sort_values(by=['user_idx', 'relevance'], ascending=[True, False])
            .groupby('user_idx')
            .head(k)
        )
       
        return pandas_to_spark(cross)
        