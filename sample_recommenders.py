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
    def __init__(self, seed=None):
        super().__init__(seed)
        self.model = LogisticRegression(
            penalty='l2', 
            C=1.0
        )
        self.scalar = StandardScaler()
    def fit(self, log:DataFrame, user_features=None, item_features=None):
        # log.show(5)
        # user_features.show(5)
        # item_features.show(5)

        if user_features and item_features:
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
            pd_log['price'] = self.scalar.fit_transform(pd_log[['price']])

            y = pd_log['relevance']
            x = pd_log.drop(['relevance'], axis=1)

            self.model.fit(x,y)
    def predict(self, log, k, users:DataFrame, items:DataFrame, user_features=None, item_features=None, filter_seen_items=True):
        cross = users.join(
            items
        ).drop('__iter').toPandas().copy()

        cross = pd.get_dummies(cross)
        cross['orig_price'] = cross['price']
        cross['price'] = self.scalar.transform(cross[['price']])
        cross.head(10).to_csv("atransformed.csv")

        cross['prob'] = self.model.predict_proba(cross.drop(['user_idx', 'item_idx', 'orig_price'], axis=1))[:,np.where(self.model.classes_ == 1)[0][0]]
        
        cross['relevance'] = cross['prob'] * cross["orig_price"] 
        # cross.head(10).to_csv("apred.csv")
        
        cross = cross.sort_values(by=['user_idx', 'relevance'], ascending=[True, False])
        cross = cross.groupby('user_idx').head(k)
        cross.head(10).to_csv("afin.csv")

        cross['price'] = cross['orig_price']
       
        return pandas_to_spark(cross)
        

class RandomRecommender:
    """
    Random recommendation algorithm.
    Recommends random items with uniform probability.
    """
    
    def __init__(self, seed=None):
        """
        Initialize random recommender.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
    def fit(self, log, user_features=None, item_features=None):
        """
        No training needed for random recommender.
        
        Args:
            log: Interaction log
            user_features: User features (optional)
            item_features: Item features (optional)
        """
        # No training needed
        pass
        
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate random recommendations.
        
        Args:
            log: Interaction log
            k: Number of items to recommend
            users: User dataframe
            items: Item dataframe
            user_features: User features (optional)
            item_features: Item features (optional)
            filter_seen_items: Whether to filter already seen items
            
        Returns:
            DataFrame: Recommendations with random relevance scores
        """
        # Cross join users and items
        recs = users.crossJoin(items)
        
        # Filter out already seen items if needed
        if filter_seen_items and log is not None:
            seen_items = log.select("user_idx", "item_idx")
            recs = recs.join(
                seen_items,
                on=["user_idx", "item_idx"],
                how="left_anti"
            )
        
        # Add random relevance scores
        recs = recs.withColumn(
            "relevance",
            sf.rand(seed=self.seed)
        )
        
        # Rank items by relevance for each user
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        recs = recs.withColumn("rank", sf.row_number().over(window))
        
        # Filter top-k recommendations
        recs = recs.filter(sf.col("rank") <= k).drop("rank")
        
        return recs


class PopularityRecommender:
    """
    Popularity-based recommendation algorithm.
    Recommends items with the highest historical interaction rate.
    """
    
    def __init__(self, alpha=1.0, seed=None):
        """
        Initialize popularity recommender.
        
        Args:
            alpha: Smoothing parameter for popularity calculation
            seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.seed = seed
        self.item_popularity = None
        
    def fit(self, log, user_features=None, item_features=None):
        """
        Calculate item popularity from interaction log.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            user_features: User features (optional)
            item_features: Item features (optional)
        """
        if log is None or log.count() == 0:
            return
            
        # Calculate item popularity as the fraction of positive interactions
        self.item_popularity = log.groupBy("item_idx").agg(
            sf.sum("relevance").alias("pos_count"),
            sf.count("relevance").alias("total_count")
        )
        
        # Apply smoothing to handle cold start
        self.item_popularity = self.item_popularity.withColumn(
            "relevance",
            (sf.col("pos_count") + self.alpha) / (sf.col("total_count") + 2 * self.alpha)
        ).select("item_idx", "relevance")
        
        # Cache for faster access
        self.item_popularity.cache()
        
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations based on item popularity.
        
        Args:
            log: Interaction log
            k: Number of items to recommend
            users: User dataframe
            items: Item dataframe
            user_features: User features (optional)
            item_features: Item features (optional)
            filter_seen_items: Whether to filter already seen items
            
        Returns:
            DataFrame: Recommendations with popularity-based relevance scores
        """
        # If no popularity data, use uniform popularity
        if self.item_popularity is None:
            # Create items with uniform popularity
            items_with_pop = items.withColumn("relevance", sf.lit(0.5))
            
            # Generate recommendations for each user
            recs = users.crossJoin(items_with_pop)
        else:
            # Join items with popularity scores
            items_with_pop = items.join(
                self.item_popularity,
                on="item_idx",
                how="left"
            ).fillna(0.5, subset=["relevance"])  # Default score for new items
            
            # Generate recommendations for each user
            recs = users.crossJoin(items_with_pop)
        
        # Filter out already seen items if needed
        if filter_seen_items and log is not None:
            seen_items = log.select("user_idx", "item_idx")
            recs = recs.join(
                seen_items,
                on=["user_idx", "item_idx"],
                how="left_anti"
            )
        
        # Rank items by popularity for each user
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        recs = recs.withColumn("rank", sf.row_number().over(window))
        
        # Filter top-k recommendations
        recs = recs.filter(sf.col("rank") <= k).drop("rank")
        
        return recs


class ContentBasedRecommender:
    """
    Content-based recommendation algorithm.
    Recommends items similar to those the user has previously interacted with.
    """
    
    def __init__(self, similarity_threshold=0.0, seed=None):
        """
        Initialize content-based recommender.
        
        Args:
            similarity_threshold: Minimum similarity score to consider
            seed: Random seed for reproducibility
        """
        self.similarity_threshold = similarity_threshold
        self.seed = seed
        self.user_profiles = None
        
    def _create_feature_vectors(self, df, feature_prefix, output_col="features"):
        """
        Create feature vectors from dataframe columns.
        
        Args:
            df: Input dataframe
            feature_prefix: Prefix of feature columns
            output_col: Name of the output feature vector column
            
        Returns:
            DataFrame: Dataframe with added feature vector column
        """
        # Get feature columns
        feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
        
        # Create vector assembler
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol=output_col
        )
        
        # Create feature vectors
        return assembler.transform(df)
    
    def _cosine_similarity_udf(self):
        """
        Create UDF for computing cosine similarity between two vectors.
        
        Returns:
            function: UDF for cosine similarity calculation
        """
        def cosine_similarity(v1, v2):
            """Safely compute cosine similarity between two Spark ML Vectors.

            If either vector is None or has zero norm, return 0.0 to avoid errors
            and undefined behaviour.
            """
            # Guard against nulls coming from missing profiles or item features
            if v1 is None or v2 is None:
                return 0.0

            # Compute dot product
            dot = v1.dot(v2)

            # Compute L2 norms
            norm1 = float(v1.norm(2))
            norm2 = float(v2.norm(2))

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot / (norm1 * norm2))
            
        return sf.udf(cosine_similarity, DoubleType())
        
    def _vector_to_dense_array_udf(self):
        """
        Create UDF for converting ML Vector to a dense array (list of doubles).
        Handles both dense and sparse vectors.
        """
        def vector_to_dense_list(vector):
            if vector is None:
                return None
            return vector.toArray().tolist()
            
        return sf.udf(vector_to_dense_list, ArrayType(DoubleType()))

    def fit(self, log, user_features=None, item_features=None):
        """
        Create user profiles based on historical interactions.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            user_features: User features (optional)
            item_features: Item features dataframe with feature columns
        """
        if log is None or log.count() == 0 or item_features is None:
            return
            
        # Create item feature vectors
        items_with_vector_features = self._create_feature_vectors(
            item_features,
            feature_prefix="item_attr_",
            output_col="item_features_vec"  # Renamed output column
        )
        
        # Convert item feature vectors to dense arrays
        # This UDF handles both DenseVector and SparseVector types correctly
        vec_to_array_udf = self._vector_to_dense_array_udf()
        items_with_array_features = items_with_vector_features.withColumn(
            "item_features_arr",  # New column with dense array features
            vec_to_array_udf(sf.col("item_features_vec"))
        )
        
        # Join log with item features (now as dense arrays)
        user_items = log.join(
            items_with_array_features.select("item_idx", "item_features_arr"),  # Use the array column
            on="item_idx"
        )
        
        # Create user profiles by averaging item features weighted by relevance
        # First create a weighted feature vector for each interaction
        # 'item_features_arr' is ArrayType(DoubleType), suitable for transform
        user_items = user_items.withColumn(
            "weighted_features",
            sf.expr("transform(item_features_arr, x -> x * relevance)")
        )
        
        # Determine the number of features robustly
        feature_cols_for_size = [col for col in item_features.columns if col.startswith("item_attr_")]
        num_features = len(feature_cols_for_size)

        if num_features == 0:
            # No item attributes found to create profiles.
            self.user_profiles = None
            return

        # Step 1: Aggregate to get sum_of_relevances and list_of_weighted_features per user
        aggregation_expressions = [
            sf.sum("relevance").alias("total_relevance_for_user"),
            sf.collect_list("weighted_features").alias("collected_weighted_features_list"),
            sf.count("item_idx").alias("interaction_count")
        ]
        user_aggregated_data = user_items.groupBy("user_idx").agg(*aggregation_expressions)

        # Step 2: Sum the arrays in collected_weighted_features_list
        # If collected_weighted_features_list is empty, aggregate returns its zero element.
        sum_expr_str = f"""
            aggregate(
                collected_weighted_features_list,
                array_repeat(0.0D, {num_features}), /* zero element: array of doubles */
                (acc, x) -> transform(arrays_zip(acc, x), pair -> pair.acc + pair.x) /* merge function */
            )
        """
        profiles_with_sum_array = user_aggregated_data.withColumn(
            "sum_weighted_features_arr",
            sf.expr(sum_expr_str)
        )
        
        # Step 3: Normalize the summed_weighted_features_array by total_relevance_for_user
        profiles_with_normalized_array = profiles_with_sum_array.withColumn(
            "user_profile_arr",
            sf.when(
                sf.col("total_relevance_for_user").isNotNull() & (sf.col("total_relevance_for_user") > 0.0),
                sf.expr(f"transform(sum_weighted_features_arr, val -> val / total_relevance_for_user)")
            ).otherwise(
                sf.expr(f"array_repeat(0.0D, {num_features})") # Default to zero array
            )
        )

        # Step 4: Convert the user_profile_array (ArrayType(DoubleType)) to a VectorUDT
        # Define a UDF for this conversion. It defaults to a zero vector of appropriate length.
        def array_to_vector_udf_func(arr):
            if arr and isinstance(arr, list) and len(arr) == num_features:
                return Vectors.dense(arr)
            return Vectors.dense([0.0] * num_features)
            
        array_to_vector_udf = sf.udf(array_to_vector_udf_func, VectorUDT())
        
        final_user_profiles = profiles_with_normalized_array.withColumn(
            "user_profile",
            array_to_vector_udf(sf.col("user_profile_arr"))
        )
        
        # Cache for faster access, selecting necessary columns
        self.user_profiles = final_user_profiles.select(
            "user_idx", "user_profile", "interaction_count", "total_relevance_for_user"
        ).cache()
        
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations based on content similarity.
        
        Args:
            log: Interaction log
            k: Number of items to recommend
            users: User dataframe
            items: Item dataframe
            user_features: User features (optional)
            item_features: Item features with feature columns
            filter_seen_items: Whether to filter already seen items
            
        Returns:
            DataFrame: Recommendations with similarity-based relevance scores
        """
        # If no user profiles or item features, return random recommendations
        if self.user_profiles is None or item_features is None:
            random_rec = RandomRecommender(seed=self.seed)
            return random_rec.predict(log, k, users, items, filter_seen_items=filter_seen_items)
            
        # Create item feature vectors
        items_with_features = self._create_feature_vectors(
            item_features,
            feature_prefix="item_attr_",
            output_col="item_features"
        )
        
        # Join users with their profiles
        users_with_profiles = users.join(
            self.user_profiles.select("user_idx", "user_profile"),
            on="user_idx",
            how="left"
        )
        
        # For users without profiles, use random recommendations
        users_with_profiles = users_with_profiles.withColumn(
            "has_profile",
            sf.col("user_profile").isNotNull()
        )
        
        # Keep user_profile as vector only if profile exists; otherwise set to null. This avoids creating
        # a literal DenseVector, which PySpark cannot handle natively with sf.lit().
        users_with_profiles = users_with_profiles.withColumn(
            "user_profile",
            sf.when(
                sf.col("has_profile"),
                sf.col("user_profile")
            ).otherwise(sf.lit(None).cast(VectorUDT()))
        )
        
        # Generate candidate recommendations
        recs = users_with_profiles.crossJoin(items_with_features)
        
        # Filter out already seen items if needed
        if filter_seen_items and log is not None:
            seen_items = log.select("user_idx", "item_idx")
            recs = recs.join(
                seen_items,
                on=["user_idx", "item_idx"],
                how="left_anti"
            )
        
        # Calculate similarity between user profile and item features
        cosine_sim = self._cosine_similarity_udf()
        recs = recs.withColumn(
            "relevance",
            sf.when(
                sf.col("has_profile"),
                cosine_sim(sf.col("user_profile"), sf.col("item_features"))
            ).otherwise(sf.rand(seed=self.seed))
        )
        
        # Filter by similarity threshold
        recs = recs.filter(sf.col("relevance") >= self.similarity_threshold)
        
        # Rank items by similarity for each user
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        recs = recs.withColumn("rank", sf.row_number().over(window))
        
        # Filter top-k recommendations
        recs = recs.filter(sf.col("rank") <= k).drop("rank", "has_profile", "user_profile", "item_features")
        
        return recs


class HybridRecommender:
    """
    Hybrid recommendation algorithm.
    Combines multiple recommendation algorithms.
    """
    
    def __init__(
        self,
        recommenders,
        weights=None,
        seed=None
    ):
        """
        Initialize hybrid recommender.
        
        Args:
            recommenders: List of recommender objects
            weights: Weights for each recommender (optional)
            seed: Random seed for reproducibility
        """
        self.recommenders = recommenders
        
        if weights is None:
            self.weights = [1.0 / len(recommenders)] * len(recommenders)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        self.seed = seed
        
    def fit(self, log, user_features=None, item_features=None):
        """
        Train all underlying recommenders.
        
        Args:
            log: Interaction log
            user_features: User features (optional)
            item_features: Item features (optional)
        """
        for recommender in self.recommenders:
            recommender.fit(log, user_features, item_features)
            
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations by combining predictions from multiple recommenders.
        
        Args:
            log: Interaction log
            k: Number of items to recommend
            users: User dataframe
            items: Item dataframe
            user_features: User features (optional)
            item_features: Item features (optional)
            filter_seen_items: Whether to filter already seen items
            
        Returns:
            DataFrame: Recommendations with combined relevance scores
        """
        # Generate recommendations from each recommender
        all_recs = []
        
        for i, (recommender, weight) in enumerate(zip(self.recommenders, self.weights)):
            # Generate recommendations
            recs = recommender.predict(
                log, 
                k=k*2,  # Request more recommendations to have more candidates
                users=users, 
                items=items,
                user_features=user_features,
                item_features=item_features,
                filter_seen_items=filter_seen_items
            )
            
            # Add recommender index and weight
            recs = recs.withColumn("recommender_idx", sf.lit(i))
            recs = recs.withColumn("weight", sf.lit(weight))
            
            all_recs.append(recs)
            
        # Combine all recommendations
        if not all_recs:
            return users.crossJoin(items).withColumn("relevance", sf.lit(0.5))
            
        combined_recs = all_recs[0]
        for recs in all_recs[1:]:
            combined_recs = combined_recs.unionByName(recs)
            
        # Aggregate relevance scores by user-item pair
        combined_recs = combined_recs.groupBy("user_idx", "item_idx").agg(
            sf.sum(sf.col("relevance") * sf.col("weight")).alias("relevance"),
            sf.collect_list("recommender_idx").alias("recommenders")
        )
        
        # Join back with item data
        result = combined_recs.join(items, on="item_idx")
        
        # Rank items by aggregated relevance for each user
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        result = result.withColumn("rank", sf.row_number().over(window))
        
        # Filter top-k recommendations
        result = result.filter(sf.col("rank") <= k).drop("rank", "recommenders")
        
        return result
        
