import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as sf
from pyspark.ml import Transformer, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.param.shared import HasOutputCol, HasInputCol
from pyspark.sql import Window 

from sim4rec.modules import Simulator, EvaluateMetrics
from sim4rec.response import BernoulliResponse, NoiseResponse
from sim4rec.utils import pandas_to_spark

class RankingMetrics:
    """
    Implementation of ranking metrics for recommendation evaluation.
    """
    
    def __init__(self, k=5, user_col="user_idx", item_col="item_idx", 
                 relevance_col="relevance", response_col="response", 
                 rank_col="rank", price_col="price", revenue_col="revenue"):
        """
        Initialize ranking metrics evaluator.
        
        Args:
            k: Cutoff for position-aware metrics
            user_col: User identifier column
            item_col: Item identifier column
            relevance_col: Column with predicted relevance scores
            response_col: Column with actual user responses (ground truth)
            rank_col: Column with ranking position (1-based)
            price_col: Column with item prices
            revenue_col: Column with revenue values
        """
        self.k = k
        self.user_col = user_col
        self.item_col = item_col
        self.relevance_col = relevance_col
        self.response_col = response_col
        self.rank_col = rank_col
        self.price_col = price_col
        self.revenue_col = revenue_col
        
    def evaluate(self, df: DataFrame) -> Dict[str, float]:
        """
        Calculate ranking metrics on the given DataFrame.
        
        Args:
            df: DataFrame with recommendations and responses
            
        Returns:
            Dict of metric names to values
        """
        # Add rank column if not present
        if self.rank_col not in df.columns:
            window = Window.partitionBy(self.user_col).orderBy(sf.desc(self.relevance_col))
            df = df.withColumn(self.rank_col, sf.row_number().over(window))
        
        # Convert to pandas for easier computation
        pdf = df.toPandas()
        
        # Calculate metrics
        metrics = {}
        
        # Precision@K
        metrics["precision_at_k"] = self._precision_at_k(pdf)
        
        # Recall@K (if we have all relevant items, otherwise skip)
        # metrics["recall_at_k"] = self._recall_at_k(pdf)
        
        # NDCG@K
        metrics["ndcg_at_k"] = self._ndcg_at_k(pdf)
        
        # MRR - Mean Reciprocal Rank
        metrics["mrr"] = self._mrr(pdf)
        
        # Hit Rate
        metrics["hit_rate"] = self._hit_rate(pdf)
        
        # Discounted Revenue
        metrics["discounted_revenue"] = self._discounted_revenue(pdf)
        
        return metrics
    
    def _precision_at_k(self, df: pd.DataFrame) -> float:
        """
        Calculate Precision@K: fraction of recommended items that were relevant.
        
        Args:
            df: Pandas DataFrame with recommendations
            
        Returns:
            Precision@K value
        """
        # Filter to top-K items
        df_topk = df[df[self.rank_col] <= self.k]
        
        # Calculate precision for each user
        user_precisions = df_topk.groupby(self.user_col).apply(
            lambda x: x[self.response_col].mean()
        )
        
        # Return average precision across users
        return user_precisions.mean() if not user_precisions.empty else 0.0
    
    def _ndcg_at_k(self, df: pd.DataFrame) -> float:
        """
        Calculate NDCG@K: Normalized Discounted Cumulative Gain.
        
        Args:
            df: Pandas DataFrame with recommendations
            
        Returns:
            NDCG@K value
        """
        def dcg(relevance_scores):
            """Calculate DCG for a list of relevance scores."""
            return np.sum(
                relevance_scores / np.log2(np.arange(2, len(relevance_scores) + 2))
            )
        
        # Group by user
        ndcg_values = []
        
        for user_id, group in df.groupby(self.user_col):
            # Only consider top-K items
            group = group[group[self.rank_col] <= self.k].copy()
            
            if group.empty:
                continue
                
            # Get actual responses
            responses = group.sort_values(self.rank_col)[self.response_col].values
            
            # Calculate DCG
            dcg_value = dcg(responses)
            
            # Calculate ideal DCG (sort by response)
            ideal_responses = np.sort(responses)[::-1]
            idcg_value = dcg(ideal_responses)
            
            # Calculate NDCG
            if idcg_value > 0:
                ndcg = dcg_value / idcg_value
                ndcg_values.append(ndcg)
        
        # Return average NDCG
        return np.mean(ndcg_values) if ndcg_values else 0.0
    
    def _mrr(self, df: pd.DataFrame) -> float:
        """
        Calculate MRR: Mean Reciprocal Rank.
        
        Args:
            df: Pandas DataFrame with recommendations
            
        Returns:
            MRR value
        """
        # Group by user
        reciprocal_ranks = []
        
        for user_id, group in df.groupby(self.user_col):
            # Find the first relevant item
            first_relevant = group[group[self.response_col] > 0].sort_values(self.rank_col)
            
            if first_relevant.empty:
                reciprocal_ranks.append(0.0)
            else:
                rank = first_relevant.iloc[0][self.rank_col]
                reciprocal_ranks.append(1.0 / rank)
        
        # Return average reciprocal rank
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def _hit_rate(self, df: pd.DataFrame) -> float:
        """
        Calculate Hit Rate: fraction of users for whom at least one recommended item was relevant.
        
        Args:
            df: Pandas DataFrame with recommendations
            
        Returns:
            Hit Rate value
        """
        # Filter to top-K items
        df_topk = df[df[self.rank_col] <= self.k]
        
        # Calculate hit rate for each user
        user_hits = df_topk.groupby(self.user_col).apply(
            lambda x: (x[self.response_col] > 0).any()
        )
        
        # Return average hit rate across users
        return user_hits.mean() if not user_hits.empty else 0.0
    
    def _discounted_revenue(self, df: pd.DataFrame) -> float:
        """
        Calculate Discounted Revenue: revenue weighted by rank position.
        
        Args:
            df: Pandas DataFrame with recommendations
            
        Returns:
            Discounted Revenue value
        """
        # Filter to top-K items
        df_topk = df[df[self.rank_col] <= self.k].copy()
        
        # Calculate discount factor
        df_topk['discount'] = 1.0 / np.log2(df_topk[self.rank_col] + 1)
        
        # Calculate discounted revenue
        df_topk['discounted_revenue'] = df_topk[self.revenue_col] * df_topk['discount']
        
        # Return total discounted revenue
        return df_topk['discounted_revenue'].sum()

class RevenueResponse(Transformer, HasInputCol, HasOutputCol):
    """
    Revenue response model that calculates purchase amount based on
    relevance score and item price.
    """
    
    def __init__(self, inputCol=None, outputCol=None, priceCol="price"):
        """
        Initialize revenue response model.
        
        Args:
            inputCol: Input column name (relevance score)
            outputCol: Output column name (revenue)
            priceCol: Column containing item price
        """
        super().__init__()
        self._set(inputCol=inputCol, outputCol=outputCol)
        self.priceCol = priceCol
        
    def _transform(self, dataset):
        """
        Calculate revenue based on relevance score and item price.
        
        Args:
            dataset: Input dataframe with relevance scores and prices
            
        Returns:
            DataFrame with added revenue column
        """
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()
        
        # Check if there are multiple columns with the name of self.priceCol.
        price_occurrences = [col for col in dataset.columns if col == self.priceCol]
        if len(price_occurrences) > 1:
            # Rename all columns to ensure uniqueness.
            new_names = []
            counts = {}
            for col in dataset.columns:
                if col in counts:
                    counts[col] += 1
                    new_names.append(f"{col}_{counts[col]}")
                else:
                    counts[col] = 0
                    new_names.append(col)
            dataset = dataset.toDF(*new_names)
            
            # After renaming, the first occurrence is still named "price"
            # and the duplicates will be named like "price_1", "price_2", etc.
            # Remove the extra duplicate price columns.
            duplicate_price_cols = [col for col in dataset.columns if col.startswith(f"{self.priceCol}_")]
            for dup_col in duplicate_price_cols:
                dataset = dataset.drop(dup_col)
        
        # Now there should be only one "price" column.
        return dataset.withColumn(
            outputCol,
            sf.col(inputCol) * sf.col(self.priceCol)
        )


class CompetitionSimulator:
    """
    Simulator for the recommendation system competition.
    Extends Sim4Rec's simulator with revenue calculation.
    """
    
    def __init__(
        self,
        user_generator,
        item_generator,
        data_dir,
        log_df=None,
        conversion_noise_mean=0.1,
        conversion_noise_std=0.05,
        spark_session=None,
        seed=42
    ):
        """
        Initialize the competition simulator.
        
        Args:
            user_generator: User data generator (from Sim4Rec)
            item_generator: Item data generator (from Sim4Rec)
            data_dir: Directory to store simulation data
            log_df: Initial log dataframe (optional)
            conversion_noise_mean: Mean of noise added to relevance scores
            conversion_noise_std: Standard deviation of noise added to relevance scores
            spark_session: Spark session (optional)
            seed: Random seed for reproducibility
        """
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
            
        # Initialize Sim4Rec simulator
        self.simulator = Simulator(
            user_gen=user_generator,
            item_gen=item_generator,
            data_dir=data_dir,
            log_df=log_df,
            user_key_col="user_idx",
            item_key_col="item_idx",
            spark_session=self.spark
        )
        
        # Set random seed
        self.seed = seed
        np.random.seed(self.seed)
        
        # Set conversion noise parameters
        self.conversion_noise_mean = conversion_noise_mean
        self.conversion_noise_std = conversion_noise_std
        
        # Create response pipeline
        self.response_pipeline = self._create_response_pipeline()
        
        # Initialize evaluation metrics
        self.evaluator = EvaluateMetrics(
            userKeyCol="user_idx",
            itemKeyCol="item_idx",
            predictionCol="relevance",
            labelCol="response",
            mllib_metrics=[]
        )
        
        # Initialize ranking metrics
        self.ranking_evaluator = RankingMetrics(
            k=5,  # Default value, will be overridden in run_iteration
            user_col="user_idx",
            item_col="item_idx",
            relevance_col="relevance",
            response_col="response",
            price_col="price",
            revenue_col="revenue"
        )
        
        # Initialize metrics storage
        self.metrics_history = []
        self.revenue_history = []
        
    def _create_response_pipeline(self):
        """
        Create a pipeline for user response simulation.
        Simulates both conversion (click/purchase) and revenue.
        
        Returns:
            PipelineModel: Pipeline for simulating user responses
        """
        # Create noise response to add randomness to relevance scores
        # This simulates the inherent unpredictability of user behavior
        noise_response = NoiseResponse(
            mu=self.conversion_noise_mean,
            sigma=self.conversion_noise_std,
            outputCol="__noisy_relevance",
            seed=self.seed
        )
        
        # Create Bernoulli response to simulate binary purchase decisions
        bernoulli_response = BernoulliResponse(
            inputCol="__noisy_relevance",
            outputCol="response",
            seed=self.seed
        )
        
        # Create revenue response to calculate purchase amount
        revenue_response = RevenueResponse(
            inputCol="response",
            outputCol="revenue",
            priceCol="price"
        )
        
        # Create pipeline
        return PipelineModel(stages=[noise_response, bernoulli_response, revenue_response])
    
    def run_iteration(
        self,
        recommender,
        user_frac=0.1,
        k=5,
        filter_seen_items=True,
        iteration=None
    ):
        """
        Run a single iteration of the simulation.
        
        Args:
            recommender: Recommendation algorithm to evaluate
            user_frac: Fraction of users to sample
            k: Number of items to recommend to each user
            filter_seen_items: Whether to filter already seen items
            iteration: Iteration label (optional)
            
        Returns:
            tuple: (metrics, revenue, true_responses)
        """
        # Sample users
        users = self.simulator.sample_users(user_frac).cache()
        
        # Get item features (including prices)
        items = self.simulator.sample_items(1.0).cache() 
        
        # Generate recommendations
        recs = recommender.predict(
            log=self.simulator.log,
            k=k,
            users=users,
            items=items,
            filter_seen_items=filter_seen_items
        ).cache()
        
        # Update k for ranking metrics
        self.ranking_evaluator.k = k
        
        # Simulate user responses
        true_responses = self.simulator.sample_responses(
            recs_df=recs,
            user_features=users,
            item_features=items,
            action_models=self.response_pipeline
        ).cache()
        
        # Calculate basic metrics
        metrics = self.evaluator(true_responses)
        
        # Calculate ranking metrics
        ranking_metrics = self.ranking_evaluator.evaluate(true_responses)
        metrics.update(ranking_metrics)
        
        # Calculate total revenue
        total_revenue = true_responses.agg(sf.sum("revenue")).collect()[0][0]
        
        # Log responses in simulator
        log_update = true_responses.select(
            "user_idx", "item_idx", "relevance", "response"
        )
        
        # Convert the relevance column from DoubleType to LongType to match the schema
        # and drop the response column since it's not in the original schema
        log_update = log_update.select(
            "user_idx", 
            "item_idx", 
            sf.col("relevance").cast("long").alias("relevance")
        )
        
        # Add a placeholder for the __iter column that will be treated specially by update_log
        # This is to match the existing schema in the Simulator's log
        if self.simulator.log is not None:
            log_schema = self.simulator.log.schema
            iter_col_exists = any(field.name == self.simulator.ITER_COLUMN for field in log_schema.fields)
            
            if iter_col_exists:
                # We don't actually need to add the column here since update_log will handle it
                # But we need to make sure it's compatible with the schema check
                pass
        
        # Update the log
        self.simulator.update_log(log_update, iteration=iteration if iteration is not None else len(self.metrics_history))
        
        # Store metrics and revenue
        metrics["revenue"] = total_revenue
        self.metrics_history.append(metrics)
        self.revenue_history.append(total_revenue)
        
        # Clean up
        users.unpersist()
        recs.unpersist()
        true_responses.unpersist()
        
        return metrics, total_revenue, true_responses
    
    def run_simulation(
        self,
        recommender,
        n_iterations=10,
        user_frac=0.1,
        k=5,
        filter_seen_items=True,
        retrain=True
    ):
        """
        Run a multi-iteration simulation.
        
        Args:
            recommender: Recommendation algorithm to evaluate
            n_iterations: Number of iterations to run
            user_frac: Fraction of users to sample in each iteration
            k: Number of items to recommend to each user
            filter_seen_items: Whether to filter already seen items
            retrain: Whether to retrain the recommender after each iteration
            
        Returns:
            tuple: (metrics_history, revenue_history)
        """
        for i in range(n_iterations):
            # Run a single iteration
            metrics, revenue, true_responses = self.run_iteration(
                recommender=recommender,
                user_frac=user_frac,
                k=k,
                filter_seen_items=filter_seen_items,
                iteration=i
            )
            
            print(f"Iteration {i}: Revenue = {revenue:.2f}, Metrics = {metrics}")
            
            # Retrain the recommender if needed
            if retrain and i < n_iterations - 1:
                # Check if the log dataframe contains a 'response' column
                columns = [field.name for field in self.simulator.log.schema.fields]
                
                if 'response' in columns:
                    # If response column exists, use it for training by renaming it to relevance
                    training_log = self.simulator.log
                    
                    # If relevance column also exists, drop it first
                    if 'relevance' in columns:
                        training_log = training_log.drop("relevance")
                    
                    # Rename response to relevance and ensure it's binary (0 or 1)
                    training_log = training_log.withColumn(
                        "relevance", 
                        sf.when(sf.col("response") > 0, 1).otherwise(0)
                    ).drop("response")
                    
                    # Train the recommender
                    recommender.fit(log=training_log)
                else:
                    # If no response column, check if the existing relevance column needs to be binarized
                    training_log = self.simulator.log
                    
                    # Ensure relevance is binary (0 or 1)
                    training_log = training_log.withColumn(
                        "relevance",
                        sf.when(sf.col("relevance") > 0, 1).otherwise(0)
                    )
                    
                    # Train the recommender
                    recommender.fit(log=training_log)
                
        return self.metrics_history, self.revenue_history
    
    def compare_recommenders(
        self,
        recommenders,
        recommender_names=None,
        n_iterations=10,
        user_frac=0.1,
        k=5,
        filter_seen_items=True,
        retrain=True
    ):
        """
        Compare multiple recommendation algorithms.
        
        Args:
            recommenders: List of recommender algorithms to compare
            recommender_names: List of names for the recommenders (optional)
            n_iterations: Number of iterations to run
            user_frac: Fraction of users to sample in each iteration
            k: Number of items to recommend to each user
            filter_seen_items: Whether to filter already seen items
            retrain: Whether to retrain recommenders after each iteration
            
        Returns:
            pd.DataFrame: Comparison results
        """
        if recommender_names is None:
            recommender_names = [f"Recommender_{i+1}" for i in range(len(recommenders))]
            
        results = []
        
        for name, recommender in zip(recommender_names, recommenders):
            # Reset simulator
            data_dir = os.path.join(self.simulator.data_dir, name)
            simulator = CompetitionSimulator(
                user_generator=self.simulator.user_gen,
                item_generator=self.simulator.item_gen,
                data_dir=data_dir,
                log_df=self.simulator.log,
                conversion_noise_mean=self.conversion_noise_mean,
                conversion_noise_std=self.conversion_noise_std,
                spark_session=self.spark,
                seed=self.seed
            )
            
            # Run simulation
            metrics_history, revenue_history = simulator.run_simulation(
                recommender=recommender,
                n_iterations=n_iterations,
                user_frac=user_frac,
                k=k,
                filter_seen_items=filter_seen_items,
                retrain=retrain
            )
            
            # Store results
            results.append({
                "name": name,
                "total_revenue": sum(revenue_history),
                "avg_revenue_per_iteration": np.mean(revenue_history),
                "metrics_history": metrics_history,
                "revenue_history": revenue_history
            })
            
        # Convert to DataFrame for easy comparison
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("total_revenue", ascending=False).reset_index(drop=True)
        
        return results_df
    
    def train_test_split(
        self,
        recommender,
        train_iterations=5,
        test_iterations=5,
        user_frac=0.1,
        k=5,
        filter_seen_items=True,
        retrain=True
    ):
        """
        Run a simulation with explicit train-test split.
        First runs train_iterations to build up history and train the recommender,
        then runs test_iterations to evaluate performance without retraining.
        
        Args:
            recommender: Recommendation algorithm to evaluate
            train_iterations: Number of iterations to use for training
            test_iterations: Number of iterations to use for testing
            user_frac: Fraction of users to sample in each iteration
            k: Number of items to recommend to each user
            filter_seen_items: Whether to filter already seen items
            retrain: Whether to retrain during the training phase
            
        Returns:
            tuple: (train_metrics, test_metrics, train_revenue, test_revenue)
        """
        # Reset metrics storage
        self.metrics_history = []
        self.revenue_history = []
        
        print("Starting Training Phase:")
        # Training phase
        train_metrics_history = []
        train_revenue_history = []
        
        for i in range(train_iterations):
            # Run a single iteration
            metrics, revenue, true_responses = self.run_iteration(
                recommender=recommender,
                user_frac=user_frac,
                k=k,
                filter_seen_items=filter_seen_items,
                iteration=f"train_{i}"
            )
            
            print(f"Training Iteration {i}: Revenue = {revenue:.2f}")
            
            # Store training metrics
            train_metrics_history.append(metrics)
            train_revenue_history.append(revenue)
            
            # Retrain the recommender if needed
            if retrain and i < train_iterations - 1:
                # Use same retraining logic as in run_simulation
                columns = [field.name for field in self.simulator.log.schema.fields]
                
                if 'response' in columns:
                    # If response column exists, use it for training by renaming it to relevance
                    training_log = self.simulator.log
                    
                    # If relevance column also exists, drop it first
                    if 'relevance' in columns:
                        training_log = training_log.drop("relevance")
                    
                    # Rename response to relevance and ensure it's binary (0 or 1)
                    training_log = training_log.withColumn(
                        "relevance", 
                        sf.when(sf.col("response") > 0, 1).otherwise(0)
                    ).drop("response")
                    
                    # Train the recommender
                    recommender.fit(log=training_log)
                else:
                    # If no response column, check if the existing relevance column needs to be binarized
                    training_log = self.simulator.log
                    
                    # Ensure relevance is binary (0 or 1)
                    training_log = training_log.withColumn(
                        "relevance",
                        sf.when(sf.col("relevance") > 0, 1).otherwise(0)
                    )
                    
                    # Train the recommender
                    recommender.fit(log=training_log)
        
        # One final retraining using all training data
        columns = [field.name for field in self.simulator.log.schema.fields]
        training_log = self.simulator.log
        if 'response' in columns:
            # If relevance column exists, drop it first
            if 'relevance' in columns:
                training_log = training_log.drop("relevance")
            
            # Rename response to relevance and ensure it's binary (0 or 1)
            training_log = training_log.withColumn(
                "relevance", 
                sf.when(sf.col("response") > 0, 1).otherwise(0)
            ).drop("response")
        else:
            # Ensure relevance is binary (0 or 1)
            training_log = training_log.withColumn(
                "relevance",
                sf.when(sf.col("relevance") > 0, 1).otherwise(0)
            )
        
        # Train the recommender one last time on all training data
        recommender.fit(log=training_log)
        
        print("\nStarting Testing Phase:")
        # Testing phase (no retraining)
        test_metrics_history = []
        test_revenue_history = []
        
        for i in range(test_iterations):
            # Run a single iteration
            metrics, revenue, true_responses = self.run_iteration(
                recommender=recommender,
                user_frac=user_frac,
                k=k,
                filter_seen_items=filter_seen_items,
                iteration=f"test_{i}"
            )
            
            print(f"Testing Iteration {i}: Revenue = {revenue:.2f}")
            
            # Store testing metrics
            test_metrics_history.append(metrics)
            test_revenue_history.append(revenue)
        
        return train_metrics_history, test_metrics_history, train_revenue_history, test_revenue_history
        
    def compare_recommenders_with_train_test_split(
        self,
        recommenders,
        recommender_names=None,
        train_iterations=5,
        test_iterations=5,
        user_frac=0.1,
        k=5,
        filter_seen_items=True,
        retrain=True
    ):
        """
        Compare multiple recommendation algorithms using train-test split.
        
        Args:
            recommenders: List of recommender algorithms to compare
            recommender_names: List of names for the recommenders (optional)
            train_iterations: Number of iterations to use for training
            test_iterations: Number of iterations to use for testing
            user_frac: Fraction of users to sample in each iteration
            k: Number of items to recommend to each user
            filter_seen_items: Whether to filter already seen items
            retrain: Whether to retrain during the training phase
            
        Returns:
            pd.DataFrame: Comparison results
        """
        if recommender_names is None:
            recommender_names = [f"Recommender_{i+1}" for i in range(len(recommenders))]
            
        results = []
        
        for name, recommender in zip(recommender_names, recommenders):
            print(f"\nEvaluating {name}:")
            # Reset simulator
            data_dir = os.path.join(self.simulator.data_dir, name)
            
            # Clean up any existing simulator data directory
            import shutil
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
            
            simulator = CompetitionSimulator(
                # Access the user and item generators from the wrapped simulator's private attributes
                user_generator=self.simulator._user_gen,  # Corrected: using _user_gen (private attribute) 
                item_generator=self.simulator._item_gen,  # Corrected: using _item_gen (private attribute)
                data_dir=data_dir,
                log_df=self.simulator.log,
                conversion_noise_mean=self.conversion_noise_mean,
                conversion_noise_std=self.conversion_noise_std,
                spark_session=self.spark,
                seed=self.seed
            )
            
            # Run simulation with train-test split
            train_metrics, test_metrics, train_revenue, test_revenue = simulator.train_test_split(
                recommender=recommender,
                train_iterations=train_iterations,
                test_iterations=test_iterations,
                user_frac=user_frac,
                k=k,
                filter_seen_items=filter_seen_items,
                retrain=retrain
            )
            
            # Calculate average metrics
            train_avg_metrics = {}
            for metric_name in train_metrics[0].keys():
                values = [metrics[metric_name] for metrics in train_metrics]
                train_avg_metrics[f"train_{metric_name}"] = np.mean(values)
            
            test_avg_metrics = {}
            for metric_name in test_metrics[0].keys():
                values = [metrics[metric_name] for metrics in test_metrics]
                test_avg_metrics[f"test_{metric_name}"] = np.mean(values)
            
            # Store results
            results.append({
                "name": name,
                "train_total_revenue": sum(train_revenue),
                "test_total_revenue": sum(test_revenue),
                "train_avg_revenue": np.mean(train_revenue),
                "test_avg_revenue": np.mean(test_revenue),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "train_revenue": train_revenue,
                "test_revenue": test_revenue,
                **train_avg_metrics,
                **test_avg_metrics
            })
            
        # Convert to DataFrame for easy comparison
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("test_total_revenue", ascending=False).reset_index(drop=True)
        
        return results_df 