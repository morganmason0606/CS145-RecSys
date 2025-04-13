import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

from sim4rec.recommenders.ucb import UCB

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
            dot = v1.dot(v2)
            norm1 = float(v1.norm(2))
            norm2 = float(v2.norm(2))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot / (norm1 * norm2))
            
        return sf.udf(cosine_similarity, DoubleType())
        
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
        items_with_features = self._create_feature_vectors(
            item_features,
            feature_prefix="item_attr_",
            output_col="item_features"
        )
        
        # Join log with item features
        user_items = log.join(
            items_with_features.select("item_idx", "item_features"),
            on="item_idx"
        )
        
        # Create user profiles by averaging item features weighted by relevance
        # First create a weighted feature vector for each interaction
        user_items = user_items.withColumn(
            "weighted_features",
            sf.expr("transform(item_features.values, x -> x * relevance)")
        )
        
        # Group by user and compute average feature vector
        self.user_profiles = user_items.groupBy("user_idx").agg(
            sf.avg(sf.col("relevance")).alias("avg_relevance"),
            sf.count("item_idx").alias("interaction_count"),
            
            # Compute average feature vector using custom aggregation
            sf.expr("""
                map_from_arrays(
                    sequence(0, size(collect_list(weighted_features)[0]) - 1),
                    aggregate(
                        collect_list(weighted_features),
                        array_repeat(0.0, size(collect_list(weighted_features)[0])),
                        (acc, x) -> transform(
                            arrays_zip(acc, x),
                            a -> a.acc + a.x
                        ),
                        acc -> transform(
                            acc,
                            x -> x / CAST(sum(relevance) AS DOUBLE)
                        )
                    )
                )
            """).alias("feature_map")
        )
        
        # Convert map to vector
        self.user_profiles = self.user_profiles.withColumn(
            "user_profile",
            sf.expr("vector(map_values(feature_map))")
        ).drop("feature_map")
        
        # Cache for faster access
        self.user_profiles.cache()
        
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
        
        users_with_profiles = users_with_profiles.withColumn(
            "user_profile",
            sf.when(sf.col("has_profile"), sf.col("user_profile"))
             .otherwise(sf.lit(Vectors.dense([0.0] * items_with_features.first()["item_features"].size)))
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
        
