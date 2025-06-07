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

class BaseRecommender:
    def __init__(self, seed=None, top_k: float=2.0):
        self.seed = seed
        np.random.seed(seed)
        self.log: Optional[DataFrame] = None
        self.scalar = StandardScaler()
        self.top_k = top_k # > 1, the top k*top_k probs you want to consider before sorting on expected price
    
    def fit(self, log, user_features=None, item_features=None):
        raise NotImplemented()
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        raise NotImplemented()
    
    def join_log(self, log):
        # keep a running total of 
        if self.log:
            self.log = self.log.union(log.select('user_idx', 'item_idx', 'relevance'))
        else:
            self.log = log.select('user_idx', 'item_idx', 'relevance')
        print(log.count(), self.log.count())
        # log.head(5)
        # self.log.head(5)
    
    def preprocess_data(self, log, user_features, item_features) -> pd.DataFrame: 
        self.join_log(log)
        pd_log = self.log.join(
            user_features, 
            on='user_idx'
        ).join(
            item_features, 
            on='item_idx'
        ).drop(
            '__iter'
        ).toPandas()

        pd_log = pd.get_dummies(pd_log)
        pd_log['scaled_price'] = self.scalar.fit_transform(pd_log[['price']])
        return pd_log.drop(['user_idx', 'item_idx'], axis=1)
    
    def prepare_predict(self, users, items) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """cross, x"""
        cross = (
            users
            .join(items)
            .drop('__iter')
            .toPandas()
        )
        cross = pd.get_dummies(cross)
        cross['scaled_price'] = self.scalar.transform(cross[['price']])
        x = cross.drop(['user_idx', 'item_idx', 'price'], axis=1)
        return (cross, x)

    def finalize_predict(self, cross: pd.DataFrame, k) -> DataFrame:
        """expect cross to have prob, price, user_idx"""
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

class LRRecommender(BaseRecommender):
    def __init__(self, seed=None, top_k=2.0, C=1e-4, penalty='l2'):
        super().__init__(seed, top_k)
        self.model = LogisticRegression(
            penalty=penalty, 
            C=C
        )

    def fit(self, log:DataFrame, user_features=None, item_features=None):
        
        if user_features and item_features:
            pd_log = self.preprocess_data(log, user_features, item_features)
            # print(pd_log.head(5))
            y = pd_log['relevance']
            x = pd_log.drop(['relevance', 'price'], axis=1)

            y.head(5)
            x.head(5)

            self.model.fit(x,y)
    def predict(self, log, k, users:DataFrame, items:DataFrame, user_features=None, item_features=None, filter_seen_items=True):

        cross, x = self.prepare_predict(users, items)
        cross['prob'] = self.model.predict_proba(x)[:,np.where(self.model.classes_ == 1)[0][0]]

        cross.head(5)
        x.head(5)

        fin = self.finalize_predict(cross, k)
        fin.head(5)
        return self.finalize_predict(cross, k)

class RFRecommender(BaseRecommender):
    def __init__(self, seed=None, top_k=2.0, n_estimators=16):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
        )
    def fit(self, log, user_features, item_features):
        if user_features and item_features: 
            pd_log = self.preprocess_data(log, user_features, item_features)
            
            y = pd_log['relevance']
            x = pd_log.drop(['relevance', 'price'], axis=1)

            self.model.fit(x,y)
    def predict(self, log, k, users:DataFrame, items:DataFrame, user_features=None, item_features=None, filter_seen_items=True):

        cross, x = self.prepare_predict(users, items)
        cross['prob'] = self.model.predict_proba(x)[:,np.where(self.model.classes_ == 1)[0][0]]
               
        return self.finalize_predict(cross, k)


from xgboost import XGBClassifier
class XGBModel(BaseRecommender):
    def __init__(self, seed=None, top_k=2.0, n_estimators=2, max_depth=2, learning_rate=1):
        super().__init__(seed, top_k)
        
        self.model : XGBClassifier = XGBClassifier( 
            objective='binary:logistic', 
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            learning_rate=learning_rate
        )
    def fit(self, log, user_features=None, item_features=None):
        if user_features and item_features:

            pd_log = self.preprocess_data(log, user_features, item_features)
            y = pd_log['relevance']
            x = pd_log.drop(['relevance', 'price'], axis=1)
            self.model.fit(x,y)
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True): 
        cross, x = self.prepare_predict(users, items)
        cross['prob'] = self.model.predict_proba(x)[:,np.where(self.model.classes_ == 1)[0][0]]
        
        return self.finalize_predict(cross, k)



import torch
import torch.nn as nn
import torch.optim as optim

class NNRecommender(BaseRecommender):
    def __init__(self, seed=None, top_k=2.0,  hidden=24, epochs=500, learning_rate=0.001):
        super().__init__(seed, top_k)
        self.NN = NN(hidden)
        self.learning_rate = learning_rate
        self.epochs = epochs
    def fit(self, log, user_features=None, item_features=None):
        if user_features and item_features:

            pd_log = self.preprocess_data(log, user_features, item_features)

            y = pd_log['relevance']
            x = pd_log.drop(['relevance', 'price'], axis=1)
            x = self.get_x(x)
            y = torch.tensor(y, dtype=torch.float).unsqueeze(1)

            criterion = nn.BCELoss()  
            optimizer = optim.Adam(self.NN.parameters(), lr=self.learning_rate)
            self.NN.train()
            prev_loss = None
            losses = []
            for epoch in range(self.epochs):

                outputs = self.NN(x)
                loss = criterion(outputs, y)

                optimizer.zero_grad() 
                loss.backward()       
                optimizer.step()
                losses.append(loss)
            print(losses)
    
    def get_x(self, x):
        bool_cols = x.select_dtypes(include='bool').columns
        for col in bool_cols:
            x[col] = x[col].astype(int)
        x = torch.tensor(x.values, dtype=torch.float)
        return x


    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        self.NN.eval()
        cross, x = self.prepare_predict(users, items)
        
        x = self.get_x(x)
        with torch.no_grad():
            cross['prob'] = self.NN(x)
       
        return self.finalize_predict(cross, k)
        
class NN(nn.Module):
    def __init__(self, hidden=24):
        super(NN, self).__init__()
        self.input = 48
        self.hidden = hidden
        self.fc1 = nn.Linear(self.input, self.hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):


        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


class DNNRecommender(BaseRecommender):
    def __init__(self, seed=None, top_k=2.0,  hidden1=24, hidden2=8, epochs=100, learning_rate=0.001):
        super().__init__(seed, top_k)
        self.NN = DNN(hidden1,hidden2)
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, log, user_features=None, item_features=None):
        if user_features and item_features:

            pd_log = self.preprocess_data(log, user_features, item_features)

            y = pd_log['relevance']
            x = pd_log.drop(['relevance', 'price'], axis=1)
            x = self.get_x(x)
            y = torch.tensor(y, dtype=torch.float).unsqueeze(1)

            criterion = nn.BCELoss()  
            optimizer = optim.Adam(self.NN.parameters(), lr=self.learning_rate)
            self.NN.train()
            prev_loss = None
            for epoch in range(self.epochs):

                outputs = self.NN(x)
                loss = criterion(outputs, y)

                optimizer.zero_grad() 
                loss.backward()       
                optimizer.step()
    
    def get_x(self, x):
        bool_cols = x.select_dtypes(include='bool').columns
        for col in bool_cols:
            x[col] = x[col].astype(int)
        x = torch.tensor(x.values, dtype=torch.float)
        return x


    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        self.NN.eval()
        cross, x = self.prepare_predict(users, items)
        
        x = self.get_x(x)
        with torch.no_grad():
            cross['prob'] = self.NN(x)
       
        return self.finalize_predict(cross, k)

class DNN(nn.Module):
    def __init__(self, hidden1=24, hidden2=8):
        super(DNN, self).__init__()
        self.input = 48
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self.fc1 = nn.Linear(self.input, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fco = nn.Linear(self.hidden2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):


        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fco(out)
        out = self.sigmoid(out)

        return out


class BetterBase:
    def __init__(self, seed=None, top_k: float=2.0):
        self.seed = seed
        np.random.seed(seed)
        self.log: Optional[DataFrame] = None
        self.scalar = StandardScaler()
        self.top_k = top_k # > 1, the top k*top_k probs you want to consider before sorting on expected price
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
    def preprocess_data(self, log, user_features, item_features) -> pd.DataFrame:  
        """expects user_featuers, item_features"""       
        pd_log = self.log.join(
            user_features, 
            on='user_idx'
        ).join(
            item_features, 
            on='item_idx'
        ).drop(
            'user_idx', 'item_idx', '__iter'
        ).toPandas()

        pd_log['scaled_price'] = self.scalar.fit_transform(pd_log[['price']])

        pd_log['cross'] = pd_log['segment'].astype(str) + '_' + pd_log['category'].astype(str)
        pd_log = pd.get_dummies(pd_log, columns=['cross'], prefix='cross_col')
        cross_col = [col for col in pd_log.columns if col.startswith('cross_col')]
        for col in cross_col:
            pd_log[col]*= pd_log['scaled_price']
         
        return pd_log.drop(['segment', 'category'], axis=1)
    
    def prepare_predict(self, users, items) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """cross, x"""
        cross = (
            users
            .join(items)
            .drop('__iter')
            .toPandas()
        )

        cross['scaled_price'] = self.scalar.transform(cross[['price']])
        cross['cross'] = cross['segment'].astype(str) + '_' + cross['category'].astype(str)
        cross = pd.get_dummies(cross, columns=['cross'], prefix='cross_col')
        cross_col = [col for col in cross.columns if col.startswith('cross_col')]
        for col in cross_col:
            cross[col]*= cross['scaled_price']

        x = cross.drop(['user_idx', 'item_idx', 'price', 'segment', 'category'], axis=1)
        return (cross, x)

    def finalize_predict(self, cross: pd.DataFrame, k) -> DataFrame:
        """expect cross to have prob, price, user_idx"""
        cross = (
            cross
            .sort_values(by=['user_idx', 'prob'], ascending=[True, False])
            .groupby('user_idx')
            .head(int(k*self.top_k))
        )
        cross['relevance'] = cross['prob'] * cross["price"] 
        cross = (
            cross
            .sort_values(by=['user_idx', 'relevance'], ascending=[True, False])
            .groupby('user_idx')
            .head(k)
        )
        
        return pandas_to_spark(cross)

class BLRRecommender(BetterBase):
    def __init__(self, seed=None, top_k=2.0, C=1.0, penalty='l2'):
        super().__init__(seed, top_k)
        self.model = LogisticRegression(
            penalty=penalty, 
            C=C
        )

    def fit(self, log:DataFrame, user_features=None, item_features=None):
        
        if user_features and item_features:
            self.join_log(log)

            pd_log = self.preprocess_data(log, user_features, item_features)

            y = pd_log['relevance']
            x = pd_log.drop(['relevance', 'price'], axis=1)

            self.model.fit(x,y)
    def predict(self, log, k, users:DataFrame, items:DataFrame, user_features=None, item_features=None, filter_seen_items=True):

        cross, x = self.prepare_predict(users, items)
        cross['prob'] = self.model.predict_proba(x)[:,np.where(self.model.classes_ == 1)[0][0]]
               
        return self.finalize_predict(cross, k)
