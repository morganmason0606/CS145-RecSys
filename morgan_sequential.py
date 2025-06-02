import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

from sim4rec.recommenders.ucb import UCB
from typing import Tuple, Optional

import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sim4rec.utils import pandas_to_spark
from sklearn.ensemble import RandomForestClassifier

class BaseSequential: 
    def __init__(self, seed=None, top_k: float = 2.0):
        self.seed = seed
        np.random.seed(seed)
        self.log: Optional[DataFrame] = None
        self.scalar = StandardScaler()
        self.top_k = top_k
        self.iter = 0
    def fit(self, log, user_featuers=None, item_features=None):
        raise NotImplemented()
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        raise NotImplemented()
    def join_log(self, log):
        if self.log:
            self.log.union(
                log.select(
                    'user_idx', 'item_idx', 'relevance'
                ).withColumn(
                    'iter', sf.lit(self.iter)
                )
            )    
        else:
            self.log = (
                log
                .select('user_idx', 'item_idx' ,'relevance')
                .withColumn('iter', sf.lit(self.iter))
            )
        self.iter +=1
    
    def preprocess_data(self, log, user_features, item_features) -> pd.DataFrame:
        self.join_log(log)
        pd_log = (
            self.log
            .join(user_features, on='user_idx')
            .join(item_features, on='item_idx')
            .drop('user_idx', 'item_idx', '__iter')
        ).toPandas()
