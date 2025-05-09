from . import model_temp
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler
from pyspark.ml.classification import LogisticRegression
import re



class logistic_model(model_temp.MMRecommender):
    """
    Logistic Regression Recommender.
    
    This class implements a logistic regression model for recommendation tasks.
    It uses the MMRecommender as a base class and extends it with logistic regression
    functionality.
    """
    
    def __init__(self, seed=None, C=1.0):
        """
        Initialize the logistic model recommender.
        
        Args:
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)
        # Add your initialization logic here
    
        # store all previous iterations -> to better retrain 
        # - can also experiemnt with only storing last n iterations of data 
        self.pipeline = Pipeline(
            stages=[
                indexers:=StringIndexer(
                    inputCols=logistic_model.scale_col, 
                    outputCols=[f"{col}_index" for col in logistic_model.categorical_cols]
                ),
                ohencoders:=OneHotEncoder(
                    inputCols=[f"{col}_index" for col in logistic_model.categorical_cols], 
                    outputCols=[f"{col}_ohe" for col in logistic_model.categorical_cols]
                ),
                ss_va:=VectorAssembler(
                    inputCols=[logistic_model.scale_col], outputCol=f"{logistic_model.scale_col}_vec"
                ),
                scaler:=StandardScaler(
                    inputCol=f"{logistic_model.scale_col}_vec",
                    outputCol=f"{logistic_model.scale_col}_scaled"
                ), 
                va:=VectorAssembler(
                    inputCols=
                    [c for c in logistic_model.columns if re.match(r".*_attr_.*", c)] + 
                    [f"{col}_ohe" for col in logistic_model.scale_col] + 
                    [f"{logistic_model.scale_col}_scaled"]
                    , outputCol="features"),
                lr:=LogisticRegression(
                    featuresCol="features", 
                    labelCol="relevance",
                    predictionCol = 'prediction',
                    regParam=C
                )
            ]
        )

    def fit(self, log, user_features=None, item_features=None):
        """
        fit once per iteration
        Train the recommender model based on interaction history.
        - can retrain every round when getting new data 

        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            -  grows per iteration
            user_features: User features dataframe (optional)
            item_features: Item features dataframe (optional)
        """
        data = (
            log
            .join(user_features, on="user_idx", how="left")
            .join(item_features, on="item_idx", how="left")
        )
        self.pipeline.fit(data)


    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations for users using logistic regression.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns; grows per iteration
            k: Number of items to recommend
            users: User dataframe, list of user_idxs; [id]
            items: Item dataframe
            user_features: User features dataframe (optional) [id, ... ]
            item_features: Item features dataframe (optional)
            filter_seen_items: Whether to filter already seen items; 
        
        Returns:
            DataFrame with recommended items for each user
            - for each user top k products 

        """
        # Implement your prediction logic here
        data = user_features.crossJoin(items)
        results = self.pipeline.transform(data).select("user_idx", "item_idx", "price", "prediction")
        results = results.withColumn('expected_revenue', results['prediction'] * results['price'])
        