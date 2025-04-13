"""
Configuration module for the recommendation system competition.
"""

# Default competition settings
DEFAULT_CONFIG = {
    # Data generation settings
    "data_generation": {
        "seed": 42,
        "n_users": 10000,
        "n_items": 1000,
        "n_user_features": 20,
        "n_item_features": 20,
        "user_feature_mean": 0.0,
        "user_feature_std": 1.0,
        "item_feature_mean": 0.0,
        "item_feature_std": 1.0,
        "user_segments": ["budget", "mainstream", "premium"],
        "user_segment_weights": [0.3, 0.5, 0.2],
        "user_segment_means": [-0.5, 0.0, 0.5],
        "item_categories": ["electronics", "books", "clothing", "home"],
        "item_category_weights": [0.2, 0.3, 0.3, 0.2],
        "item_category_means": [0.3, -0.3, 0.0, 0.2],
        "initial_history_density": 0.001
    },
    
    # Simulation settings
    "simulation": {
        "data_dir": "competition_data",
        "conversion_noise_mean": 0.1,
        "conversion_noise_std": 0.05,
        "iterations": 10,
        "user_fraction": 0.1,
        "k": 5,
        "filter_seen_items": True,
        "retrain": True,
        # Train-test split parameters
        "train_iterations": 5,   # Number of iterations to use for training
        "test_iterations": 5,    # Number of iterations to use for testing
        "train_test_split": True # Whether to perform train-test split evaluation
    },
    
    # Competition settings
    "competition": {
        "max_submission_per_day": 3,
        "public_leaderboard_fraction": 0.4,
        "private_leaderboard_fraction": 0.6,
        "evaluation_metric": "revenue",  # Primary metric
        "evaluation_iterations": 20,
        "time_limit_seconds": 3600
    }
}

# Competition phases
COMPETITION_PHASES = {
    "development": {
        "description": "Development phase for testing algorithms",
        "public_leaderboard": True,
        "private_leaderboard": False,
        "max_submissions_total": 100,
        "feedback": "full",  # Provides full feedback on algorithm performance
        "data_access": "all"  # Provides access to all training data
    },
    "validation": {
        "description": "Validation phase with limited submissions",
        "public_leaderboard": True,
        "private_leaderboard": False,
        "max_submissions_total": 20, 
        "feedback": "partial",  # Provides limited feedback on algorithm performance
        "data_access": "partial"  # Provides access to partial training data
    },
    "final": {
        "description": "Final evaluation phase",
        "public_leaderboard": True,
        "private_leaderboard": True,
        "max_submissions_total": 5,
        "feedback": "minimal",  # Provides minimal feedback (only on public leaderboard)
        "data_access": "locked"  # No additional data access during this phase
    }
}

# Baseline recommender algorithms
BASELINE_RECOMMENDERS = [
    {
        "name": "Random",
        "class": "RandomRecommender",
        "description": "Recommends random items",
        "parameters": {
            "seed": 42
        }
    },
    {
        "name": "Popularity",
        "class": "PopularityRecommender",
        "description": "Recommends popular items based on interaction history",
        "parameters": {
            "alpha": 1.0,
            "seed": 42
        }
    },
    {
        "name": "UCB",
        "class": "UCB",
        "description": "Upper Confidence Bound algorithm for exploration-exploitation trade-off",
        "parameters": {
            "exploration_coef": 2.0,
            "sample": False,
            "seed": 42
        }
    },
    {
        "name": "ContentBased",
        "class": "ContentBasedRecommender",
        "description": "Recommends items similar to those the user has interacted with",
        "parameters": {
            "similarity_threshold": 0.0,
            "seed": 42
        }
    },
    {
        "name": "EnhancedUCB",
        "class": "EnhancedUCB",
        "description": "Enhanced UCB algorithm with price consideration",
        "parameters": {
            "exploration_coef": 2.0,
            "price_weight": 0.3,
            "sample": False,
            "seed": 42
        }
    },
    {
        "name": "Hybrid",
        "class": "HybridRecommender",
        "description": "Hybrid recommender combining Popularity and ContentBased",
        "parameters": {
            "recommenders": [
                {
                    "class": "PopularityRecommender",
                    "parameters": {"alpha": 1.0, "seed": 42}
                },
                {
                    "class": "ContentBasedRecommender",
                    "parameters": {"similarity_threshold": 0.0, "seed": 42}
                }
            ],
            "weights": [0.7, 0.3],
            "seed": 42
        }
    }
]

# Evaluation metrics
EVALUATION_METRICS = {
    "revenue": {
        "name": "Total Revenue",
        "description": "Total revenue generated from user purchases",
        "higher_is_better": True,
        "column": "revenue"
    },
    "discounted_revenue": {
        "name": "Discounted Revenue",
        "description": "Revenue discounted by position in the ranking (higher ranks get higher weight)",
        "higher_is_better": True,
        "column": None  # Calculated during evaluation
    },
    "conversion_rate": {
        "name": "Conversion Rate",
        "description": "Fraction of recommendations that resulted in a purchase",
        "higher_is_better": True,
        "column": "response"
    },
    "precision_at_k": {
        "name": "Precision@K",
        "description": "Fraction of recommended items that were relevant",
        "higher_is_better": True,
        "column": None  # Calculated during evaluation
    },
    "recall_at_k": {
        "name": "Recall@K",
        "description": "Fraction of relevant items that were recommended",
        "higher_is_better": True,
        "column": None  # Calculated during evaluation
    },
    "ndcg_at_k": {
        "name": "NDCG@K",
        "description": "Normalized Discounted Cumulative Gain at K",
        "higher_is_better": True,
        "column": None  # Calculated during evaluation
    },
    "mrr": {
        "name": "MRR",
        "description": "Mean Reciprocal Rank - average of reciprocal ranks of the first relevant item",
        "higher_is_better": True,
        "column": None  # Calculated during evaluation
    },
    "hit_rate": {
        "name": "Hit Rate",
        "description": "Fraction of users for whom at least one relevant item was recommended",
        "higher_is_better": True,
        "column": None  # Calculated during evaluation
    },
    # "areaUnderROC": {
    #     "name": "AUC-ROC",
    #     "description": "Area under the ROC curve for predicting user response",
    #     "higher_is_better": True,
    #     "column": None  # Uses Spark MLlib evaluator
    # },
    # "accuracy": {
    #     "name": "Accuracy",
    #     "description": "Fraction of correct predictions",
    #     "higher_is_better": True,
    #     "column": None  # Uses Spark MLlib evaluator
    # }
}

# Submission template
SUBMISSION_TEMPLATE = {
    "recommender_class": None,  # Class implementing the recommender interface
    "parameters": {},           # Parameters for the recommender
    "description": "",          # Description of the algorithm
    "team_name": "",            # Team name
    "members": []               # List of team members
}

# Submission validation criteria
SUBMISSION_VALIDATION = {
    "required_methods": ["fit", "predict"],
    "predict_params": ["log", "k", "users", "items", "filter_seen_items"],
    "fit_params": ["log", "user_features", "item_features"],
    "time_limit_seconds": 3600,
    "memory_limit_mb": 8192
} 