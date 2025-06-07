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
from sim4rec.utils import pandas_to_spark

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


FEATURES = ['user_attr_0', 'user_attr_1', 'user_attr_2', 'user_attr_3',
        'user_attr_4', 'user_attr_5', 'user_attr_6', 'user_attr_7',
        'user_attr_8', 'user_attr_9', 'user_attr_10', 'user_attr_11',
        'user_attr_12', 'user_attr_13', 'user_attr_14', 'user_attr_15',
        'user_attr_16', 'user_attr_17', 'user_attr_18', 'user_attr_19',
        'item_attr_0', 'item_attr_1', 'item_attr_2', 'item_attr_3',
        'item_attr_4', 'item_attr_5', 'item_attr_6', 'item_attr_7',
        'item_attr_8', 'item_attr_9', 'item_attr_10', 'item_attr_11',
        'item_attr_12', 'item_attr_13', 'item_attr_14', 'item_attr_15',
        'item_attr_16', 'item_attr_17', 'item_attr_18', 'item_attr_19', 'scaled_price',
        'segment_budget', 'segment_mainstream', 'segment_premium',
        'category_books', 'category_clothing', 'category_electronics',
        'category_home']
class BaseSequential: 
    def __init__(self, seed=None, top_k: float = 2.0):
        self.seed = seed
        np.random.seed(seed)
        self.log: Optional[DataFrame] = None
        self.scalar = StandardScaler()
        self.top_k = top_k
        self.iter = 0
    def fit(self, log, user_features=None, item_features=None):
        raise NotImplemented()
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        raise NotImplemented()
    def join_log(self, log):
        if self.log:
            self.log = self.log.union(
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
        ).toPandas()
        pd_log['scaled_price'] = self.scalar.fit_transform(pd_log[['price']])
        return pd_log
    
    def prepare_predict(self, users, items) -> pd.DataFrame:
        """cross, x"""
        cross = (
            users
            .join(items)
            .drop('__iter')
            .toPandas()
        )
        cross = pd.get_dummies(cross)
        cross['scaled_price'] = self.scalar.transform(cross[['price']])
        return cross

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



class RNNRec(BaseSequential):
    def __init__(self, seed=42, top_k = 2.0, h=32, early_break=3):
        super().__init__(seed, top_k)
        self.hidden = h
        self.inlen = 3
        self.model = None
        self.epochs = 25
        self.early_break = early_break

    def fit(self, log, user_features=None, item_features=None):
        pd_log = self.preprocess_data(log, user_features, item_features)
        pd_log = pd.get_dummies(pd_log)

        grouped_data = pd_log.groupby(['user_idx'])
        input_dim = len(FEATURES)
        model = RNNModel(input_dim, self.hidden, 3, 1)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.model = model
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            for user_idx, user_df in grouped_data:
                user_df = user_df.sort_values('iter')
                x = torch.tensor(user_df[FEATURES].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)  # shape: (1, seq_len, input_dim)
                y = torch.tensor(user_df['relevance'].values, dtype=torch.float32)  # shape: (seq_len,)
                optimizer.zero_grad()
                output, _ = model(x)
                output = output.squeeze()  # shape: (seq_len,)
                if output.dim() == 0:
                    output = output.unsqueeze(0)
                # print(output, y)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}, Loss: {total_loss}")
            if total_loss < self.early_break:
                print('early break')
                break
        self.model = model 

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        self.model.eval()

        pd_log = (
            self.log
            .join(user_features, on='user_idx')
            .join(item_features, on='item_idx')
        ).toPandas()
        pd_log['scaled_price'] = self.scalar.transform(pd_log[['price']])
        pd_log = pd.get_dummies(pd_log)

        # Prepare candidate item features
        items_pd = items.toPandas()
        items_pd = pd.get_dummies(items_pd)
        items_pd['scaled_price'] = self.scalar.transform(items_pd[['price']])
        
        # Prepare user features
        users_pd = users.toPandas()
        users_pd = pd.get_dummies(users_pd)
        
        # For each user, get their sequence and hidden state
        results = []
        grouped_data = pd_log.groupby(['user_idx'])
        for user_idx, user_df in grouped_data:
            # If user_idx is a tuple, extract the first element
            if isinstance(user_idx, tuple):
                user_idx_val = user_idx[0]
            else:
                user_idx_val = user_idx
            user_row = users_pd[users_pd['user_idx'] == user_idx_val]
            if user_row.empty:
                continue
            user_df = user_df.sort_values('iter')
            x = torch.tensor(user_df[FEATURES].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)  # (1, seq_len, input_dim)
            # print(x)
            with torch.no_grad():
                _, hn = self.model(x)

            # print(hn)
            user_feat = user_row.iloc[0].to_dict()
            for _, item_row in items_pd.iterrows():
                feature_dict = {**user_feat, **item_row.to_dict()}
                feature_vec = [feature_dict.get(f, 0.0) for f in FEATURES]
                feature_tensor = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
                # print(feature_tensor)
                with torch.no_grad():
                    prob, _ = self.model(feature_tensor, hn)
                results.append({
                    'user_idx': user_idx_val,
                    'item_idx': item_row['item_idx'],
                    'prob': prob.item(),
                    'price': item_row['price'] if 'price' in item_row else 1.0
                })
        cross = pd.DataFrame(results)
        # print(cross.head(5))
        fin = self.finalize_predict(cross, k)
        return fin

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=1):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, h=None):

        if h is None:
            h = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device)
        out, hn = self.rnn(x, h)
        # out = self.fc(out[:, -1, :])
        out = self.fc(out)
        out = torch.sigmoid(out)

        return out, hn


class LSTMRec(BaseSequential):
    def __init__(self, seed=42, top_k=2.0, h=32, early_break=3):
        super().__init__(seed, top_k)
        self.hidden = h
        self.inlen = 3
        self.model = None
        self.epochs = 25
        self.early_break = early_break

    def fit(self, log, user_features=None, item_features=None):
        pd_log = self.preprocess_data(log, user_features, item_features)
        pd_log = pd.get_dummies(pd_log)
        grouped_data = pd_log.groupby(['user_idx'])
        input_dim = len(FEATURES)
        model = LSTMModel(input_dim, self.hidden, 3, 1)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.model = model
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            for user_idx, user_df in grouped_data:
                user_df = user_df.sort_values('iter')
                x = torch.tensor(user_df[FEATURES].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)
                y = torch.tensor(user_df['relevance'].values, dtype=torch.float32)
                optimizer.zero_grad()
                output, _ = model(x)
                output = output.squeeze()
                if output.dim() == 0:
                    output = output.unsqueeze(0)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}, Loss: {total_loss}")
            if total_loss < self.early_break:
                print('early break')
                break
        self.model = model

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        self.model.eval()
        pd_log = (
            self.log
            .join(user_features, on='user_idx')
            .join(item_features, on='item_idx')
        ).toPandas()
        pd_log['scaled_price'] = self.scalar.transform(pd_log[['price']])
        pd_log = pd.get_dummies(pd_log)
        items_pd = items.toPandas()
        items_pd = pd.get_dummies(items_pd)
        items_pd['scaled_price'] = self.scalar.transform(items_pd[['price']])
        users_pd = users.toPandas()
        users_pd = pd.get_dummies(users_pd)
        results = []
        grouped_data = pd_log.groupby(['user_idx'])
        for user_idx, user_df in grouped_data:
            if isinstance(user_idx, tuple):
                user_idx_val = user_idx[0]
            else:
                user_idx_val = user_idx
            user_row = users_pd[users_pd['user_idx'] == user_idx_val]
            if user_row.empty:
                continue
            user_df = user_df.sort_values('iter')
            print(user_df.head(2))
            x = torch.tensor(user_df[FEATURES].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, hn = self.model(x)
            user_feat = user_row.iloc[0].to_dict()
            for _, item_row in items_pd.iterrows():
                feature_dict = {**user_feat, **item_row.to_dict()}
                feature_vec = [feature_dict.get(f, 0.0) for f in FEATURES]
                feature_tensor = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    prob, _ = self.model(feature_tensor, hn)
                results.append({
                    'user_idx': user_idx_val,
                    'item_idx': item_row['item_idx'],
                    'prob': prob.item(),
                    'price': item_row['price'] if 'price' in item_row else 1.0
                })
        cross = pd.DataFrame(results)
        fin = self.finalize_predict(cross, k)
        return fin

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, h=None):
        if h is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device)
            h = (h0, c0)
        out, (hn, cn) = self.lstm(x, h)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out, (hn, cn)
