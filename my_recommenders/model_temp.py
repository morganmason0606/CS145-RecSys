import pandas as pd
from pyspark.sql import DataFrame


class MMRecommender:
    """
    Template class for implementing a custom recommender.

    This class provides the basic structure required to implement a recommender
    that can be used with the Sim4Rec simulator. Students should extend this class
    with their own recommendation algorithm.
    """

    columns = [
        "relevance",
        "user_attr_0",
        "user_attr_1",
        "user_attr_2",
        "user_attr_3",
        "user_attr_4",
        "user_attr_5",
        "user_attr_6",
        "user_attr_7",
        "user_attr_8",
        "user_attr_9",
        "user_attr_10",
        "user_attr_11",
        "user_attr_12",
        "user_attr_13",
        "user_attr_14",
        "user_attr_15",
        "user_attr_16",
        "user_attr_17",
        "user_attr_18",
        "user_attr_19",
        "segment",
        "item_attr_0",
        "item_attr_1",
        "item_attr_2",
        "item_attr_3",
        "item_attr_4",
        "item_attr_5",
        "item_attr_6",
        "item_attr_7",
        "item_attr_8",
        "item_attr_9",
        "item_attr_10",
        "item_attr_11",
        "item_attr_12",
        "item_attr_13",
        "item_attr_14",
        "item_attr_15",
        "item_attr_16",
        "item_attr_17",
        "item_attr_18",
        "item_attr_19",
        "category",
        "price",
    ]
    categorical_cols = ["category", "segment"]
    scale_col = ["price"]

    def __init__(self, seed=None):
        """
        Initialize recommender.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        # Add your initialization logic here

    @staticmethod
    def pyspark_data_to_pandas(
        log: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
        drop_first: bool = True,
    ) -> pd.DataFrame:
        data = (
            log.join(user_features, on="user_idx", how="left").join(
                item_features, on="item_idx", how="left"
            )
        ).toPandas()
        return pd.get_dummies(
            data, columns=["category", "segment"], drop_first=drop_first
        )

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

    def predict(
        self,
        log,
        k,
        users,
        items,
        user_features=None,
        item_features=None,
        filter_seen_items=True,
    ):
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
