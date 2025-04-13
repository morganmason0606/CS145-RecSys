import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame

from sim4rec.modules import RealDataGenerator, SDVDataGenerator, CompositeGenerator
from sim4rec.utils import pandas_to_spark

class CompetitionDataGenerator:
    """
    Data generator for the recommendation competition.
    Creates synthetic user and item data with configurable properties.
    """
    
    def __init__(
        self, 
        spark_session=None,
        seed=42,
        n_users=10000,
        n_items=1000,
        n_user_features=20,
        n_item_features=15,
        user_feature_mean=0.0,
        user_feature_std=1.0,
        item_feature_mean=0.0,
        item_feature_std=1.0,
        user_segments=None,
        user_segment_weights=None,
        user_segment_means=None,
        item_categories=None,
        item_category_weights=None,
        item_category_means=None, 
        **kwargs
    ):
        """
        Initialize the competition data generator.
        
        Args:
            spark_session: SparkSession for data processing
            seed: Random seed for reproducibility
            n_users: Number of users to generate
            n_items: Number of items to generate
            n_user_features: Number of user features
            n_item_features: Number of item features
            user_feature_mean: Mean of user feature distribution
            user_feature_std: Standard deviation of user feature distribution
            item_feature_mean: Mean of item feature distribution
            item_feature_std: Standard deviation of item feature distribution
            user_segments: List of user segment names (optional)
            user_segment_weights: List of weights for each user segment (optional)
            user_segment_means: List of feature means for each user segment (optional)
            item_categories: List of item category names (optional)
            item_category_weights: List of weights for each item category (optional)
            item_category_means: List of feature means for each item category (optional)
        """
        # Set random seed for reproducibility
        self.seed = seed
        np.random.seed(self.seed)
        
        # Initialize Spark session if not provided
        if spark_session is None:
            self.spark = SparkSession.builder \
                .appName("RecSysCompetition") \
                .master("local[*]") \
                .config("spark.driver.memory", "4g") \
                .config("spark.sql.shuffle.partitions", "8") \
                .getOrCreate()
        else:
            self.spark = spark_session
            
        # Set data generation parameters
        self.n_users = n_users
        self.n_items = n_items
        self.n_user_features = n_user_features
        self.n_item_features = n_item_features
        self.user_feature_mean = user_feature_mean
        self.user_feature_std = user_feature_std
        self.item_feature_mean = item_feature_mean
        self.item_feature_std = item_feature_std
        
        # Set user segments if provided
        if user_segments is not None:
            self.user_segments = user_segments
            self.user_segment_weights = user_segment_weights if user_segment_weights else [1/len(user_segments)] * len(user_segments)
            self.user_segment_means = user_segment_means if user_segment_means else [self.user_feature_mean] * len(user_segments)
        else:
            self.user_segments = ["default"]
            self.user_segment_weights = [1.0]
            self.user_segment_means = [self.user_feature_mean]
            
        # Set item categories if provided
        if item_categories is not None:
            self.item_categories = item_categories
            self.item_category_weights = item_category_weights if item_category_weights else [1/len(item_categories)] * len(item_categories)
            self.item_category_means = item_category_means if item_category_means else [self.item_feature_mean] * len(item_categories)
        else:
            self.item_categories = ["default"]
            self.item_category_weights = [1.0]
            self.item_category_means = [self.item_feature_mean]
            
        # Initialize data frames
        self.users_df = None
        self.items_df = None
        self.history_df = None
        
        # Initialize data generators
        self.user_generators = []
        self.item_generator = None
        self.composite_user_generator = None
        
    def generate_users(self):
        """
        Generate synthetic user data based on configured segments.
        
        Returns:
            DataFrame: Spark DataFrame containing user data
        """
        # Calculate users per segment
        users_per_segment = [int(w * self.n_users) for w in self.user_segment_weights]
        # Ensure we have exactly n_users
        users_per_segment[-1] += self.n_users - sum(users_per_segment)
        
        # Generate users for each segment
        all_users = []
        user_idx_offset = 0
        
        for i, (segment, n_seg_users, seg_mean) in enumerate(zip(self.user_segments, users_per_segment, self.user_segment_means)):
            # Generate user features for this segment
            seg_users = pd.DataFrame(
                data=np.random.normal(seg_mean, self.user_feature_std, size=(n_seg_users, self.n_user_features)),
                columns=[f"user_attr_{j}" for j in range(self.n_user_features)]
            )
            
            # Add user_id and segment
            seg_users["user_idx"] = np.arange(user_idx_offset, user_idx_offset + n_seg_users)
            seg_users["segment"] = segment
            
            # Update offset and add to list
            user_idx_offset += n_seg_users
            all_users.append(seg_users)
            
        # Combine all segments
        users_df_pd = pd.concat(all_users, ignore_index=True)
        
        # Convert to Spark DataFrame
        self.users_df = pandas_to_spark(users_df_pd)
        
        return self.users_df
        
    def generate_items(self):
        """
        Generate synthetic item data based on configured categories.
        
        Returns:
            DataFrame: Spark DataFrame containing item data
        """
        # Calculate items per category
        items_per_category = [int(w * self.n_items) for w in self.item_category_weights]
        # Ensure we have exactly n_items
        items_per_category[-1] += self.n_items - sum(items_per_category)
        
        # Generate items for each category
        all_items = []
        item_idx_offset = 0
        
        for i, (category, n_cat_items, cat_mean) in enumerate(zip(self.item_categories, items_per_category, self.item_category_means)):
            # Generate item features for this category
            cat_items = pd.DataFrame(
                data=np.random.normal(cat_mean, self.item_feature_std, size=(n_cat_items, self.n_item_features)),
                columns=[f"item_attr_{j}" for j in range(self.n_item_features)]
            )
            
            # Add item_id and category
            cat_items["item_idx"] = np.arange(item_idx_offset, item_idx_offset + n_cat_items)
            cat_items["category"] = category
            
            # Add price for each item (for revenue calculation)
            cat_items["price"] = np.random.gamma(shape=5.0, scale=10.0, size=n_cat_items)
            
            # Update offset and add to list
            item_idx_offset += n_cat_items
            all_items.append(cat_items)
            
        # Combine all categories
        items_df_pd = pd.concat(all_items, ignore_index=True)
        
        # Convert to Spark DataFrame
        self.items_df = pandas_to_spark(items_df_pd)
        
        return self.items_df
    
    def generate_initial_history(self, interaction_density=0.01):
        """
        Generate initial interaction history between users and items.
        
        Args:
            interaction_density: Fraction of all possible user-item pairs to include
            
        Returns:
            DataFrame: Spark DataFrame containing interaction history
        """
        # Calculate total number of interactions
        n_interactions = int(interaction_density * self.n_users * self.n_items)
        
        # Generate random user-item pairs
        history_df_pd = pd.DataFrame()
        history_df_pd["user_idx"] = np.random.randint(0, self.n_users, size=n_interactions)
        history_df_pd["item_idx"] = np.random.randint(0, self.n_items, size=n_interactions)
        
        # Remove duplicates
        history_df_pd = history_df_pd.drop_duplicates(subset=["user_idx", "item_idx"])
        
        # Get user and item features for dot product calculation
        users_df_pd = self.users_df.toPandas()
        items_df_pd = self.items_df.toPandas()
        
        # Extract feature matrices
        user_features = users_df_pd[[col for col in users_df_pd.columns if col.startswith("user_attr_")]].values
        item_features = items_df_pd[[col for col in items_df_pd.columns if col.startswith("item_attr_")]].values
        
        # Calculate relevance (based on dot product of user and item features)
        users_matrix = user_features[history_df_pd["user_idx"].values]
        items_matrix = item_features[history_df_pd["item_idx"].values]
        
        # Simple dot product between user and item features
        dot_products = np.sum(users_matrix * items_matrix, axis=1)
        
        # Convert to binary relevance (1 if dot product is positive, 0 otherwise)
        history_df_pd["relevance"] = np.where(dot_products > 0, 1, 0)
        
        # Convert to Spark DataFrame
        self.history_df = pandas_to_spark(history_df_pd)
        
        return self.history_df
    
    def setup_data_generators(self):
        """
        Set up Sim4Rec data generators for users and items.
        
        Returns:
            tuple: (composite_user_generator, item_generator)
        """
        # Create a generator for each user segment
        for i, segment in enumerate(self.user_segments):
            users_segment_df = self.users_df.filter(f"segment = '{segment}'")
            
            # Create a data generator for this segment
            segment_gen = RealDataGenerator(
                label=f"users_{segment}",
                seed=self.seed + i
            )
            segment_gen.fit(users_segment_df)
            
            self.user_generators.append(segment_gen)
        
        # Create a composite generator for all user segments
        self.composite_user_generator = CompositeGenerator(
            generators=self.user_generators,
            label="users_composite",
            weights=self.user_segment_weights
        )
        
        # Generate users data
        self.composite_user_generator.generate(self.n_users)
        
        # Create a generator for items
        self.item_generator = RealDataGenerator(
            label="items_real",
            seed=self.seed + len(self.user_segments)
        )
        self.item_generator.fit(self.items_df)
        
        # Generate items data
        self.item_generator.generate(self.n_items)
        
        return self.composite_user_generator, self.item_generator
    
    def save_data(self, output_dir="competition_data"):
        """
        Save generated data to Parquet files.
        
        Args:
            output_dir: Directory to save data files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.users_df is not None:
            self.users_df.write.parquet(f"{output_dir}/users.parquet", mode="overwrite")
            
        if self.items_df is not None:
            self.items_df.write.parquet(f"{output_dir}/items.parquet", mode="overwrite")
            
        if self.history_df is not None:
            self.history_df.write.parquet(f"{output_dir}/history.parquet", mode="overwrite")
    
    def load_data(self, input_dir="competition_data"):
        """
        Load data from Parquet files.
        
        Args:
            input_dir: Directory to load data files from
        """
        self.users_df = self.spark.read.parquet(f"{input_dir}/users.parquet")
        self.items_df = self.spark.read.parquet(f"{input_dir}/items.parquet")
        self.history_df = self.spark.read.parquet(f"{input_dir}/history.parquet") 