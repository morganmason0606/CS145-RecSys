import pandas as pd
from pyspark.sql import DataFrame


class MMRecommender:
    """
    Template class for implementing a custom recommender.
    
    This class provides the basic structure required to implement a recommender
    that can be used with the Sim4Rec simulator. Students should extend this class
    with their own recommendation algorithm.
    """
    
    def __init__(self, seed=None):
        """
        Initialize recommender.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        # Add your initialization logic here

    
    @staticmethod
    def pyspark_data_to_pandas(log: DataFrame, user_features: DataFrame, item_features: DataFrame, drop_first:bool=True) ->pd.DataFrame:
        data = (   
            log.join(user_features, on="user_idx", how="left")
            .join(item_features, on="item_idx", how="left")
            .drop("user_idx", "item_idx")
        ).toPandas()
        return pd.get_dummies(data, columns=['category', 'segment'], drop_first=drop_first)
    


    def fit(self, log, user_features=None, item_features=None):
        """
        Train the recommender model based on interaction history.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            user_features: User features dataframe (optional)
            item_features: Item features dataframe (optional)
        """
        # Implement your training logic here
        raise NotImplementedError("fit method not implemented")

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations for users.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            k: Number of items to recommend
            users: User dataframe
            items: Item dataframe
            user_features: User features dataframe (optional)
            item_features: Item features dataframe (optional)
            filter_seen_items: Whether to filter already seen items
        
        Returns:
            DataFrame: Recommendations with user_idx, item_idx, and relevance columns
        """
        # Example of a random recommender implementation:
        # Cross join users and items
        raise NotImplementedError("predict method not implemented")
