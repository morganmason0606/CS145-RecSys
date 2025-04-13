
# Recommendation System Competition Platform

A data mining course competition platform for designing recommendation algorithms that maximize revenue through multi-iteration learning.

## Overview

This platform uses the [Sim4Rec](https://github.com/sb-ai-lab/Sim4Rec) simulation framework to:

1. Generate synthetic users and items with various properties
2. Simulate user responses to recommendations over multiple iterations
3. Evaluate recommendation algorithms based on revenue generation
4. Compare performance across different algorithms

Students will compete to design recommendation algorithms that earn the most money by recommending the right items to the right users. The system implements a multi-iteration learning environment where algorithms can adapt over time based on user feedback.

## Problem Setting: Multi-Iteration Ranking Task

The competition focuses on a sequential recommendation problem where:

- Each recommendation algorithm interacts with users across multiple iterations
- Algorithms make top-k item recommendations to users in each iteration
- User responses (purchases) generate revenue based on item prices
- Algorithms can learn from past interactions to improve future recommendations
- The goal is to maximize cumulative revenue across all iterations

This setup realistically simulates how recommendation systems operate in production environments, where models continuously learn from user interactions.

## Training and Evaluation Setup

The platform uses a train-test split evaluation approach:

1. **Training Phase**:
   - Algorithms interact with users for a fixed number of training iterations
   - Recommenders can be retrained after each iteration based on new feedback
   - This phase allows algorithms to learn user preferences

2. **Testing Phase**:
   - Algorithms make recommendations for additional test iterations
   - No retraining occurs during testing phase
   - Performance on test iterations determines final evaluation

This approach tests both an algorithm's ability to learn from interactions and its generalization performance on unseen data.

## Metrics

The platform evaluates recommenders using the following metrics:

1. **Total Revenue (Primary)**: Sum of prices for all purchased items
   ```
   Revenue = Sum(price * response)
   ```
   where `response` is 1 for purchased items and 0 otherwise

2. **Discounted Revenue**: Revenue weighted by recommendation position
   ```
   Discounted Revenue = Sum(price * response * (1/log2(rank + 1)))
   ```

3. **Precision@K**: Fraction of recommended items that were relevant
   ```
   Precision@K = (# of recommended items that were purchased) / K
   ```

4. **NDCG@K**: Normalized Discounted Cumulative Gain, which measures ranking quality
   ```
   DCG = Sum(relevance_i / log2(i+1))
   IDCG = DCG of the ideal ranking
   NDCG = DCG / IDCG
   ```

5. **MRR**: Mean Reciprocal Rank, the average of reciprocal ranks of the first relevant item
   ```
   MRR = Average(1/rank of first relevant item)
   ```

6. **Hit Rate**: Fraction of users for whom at least one recommended item was relevant
   ```
   Hit Rate = (# of users with at least one purchased item) / (# of users)
   ```

## Installation

### Prerequisites

- Python 3.9
- Java 8+ (required for PySpark)

### Setup

1. Clone this repository
```bash
git clone https://github.com/FrancoTSolis/CS145-RecSys.git
cd CS145-RecSys
```

2. Install dependencies
```bash
pip install --upgrade pip wheel poetry
poetry install
pip install PyYAML==6.0.1
pip install pyspark==3.5.5 numpy matplotlib==3.7.5 seaborn==0.13.2 torch sdv==0.18.0 pyarrow
```

## Getting Started

The main execution flow is in `recommender_analysis_visualization.py`, which:

1. Generates synthetic user and item data
2. Performs exploratory data analysis
3. Sets up and evaluates baseline recommenders
4. Visualizes performance metrics

Run the analysis script to get started:
```bash
python recommender_analysis_visualization.py
```

This will:
- Generate a synthetic dataset with users and items
- Create visualizations of user segments, item categories, and interactions
- Run multiple baseline recommenders (Random, Popularity, Content-Based)
- Compare their performance using train-test evaluation
- Generate visualizations of recommender performance

## Developing Your Algorithm

1. Start with the `MyRecommender` class in `recommender_analysis_visualization.py`

2. Implement your recommendation algorithm with:
   - `__init__`: Initialize your algorithm and parameters
   - `fit`: Train your model on historical data
   - `predict`: Generate recommendations for users

3. Test your algorithm using the `run_recommender_analysis` function

## Checkpoints

The competition consists of three checkpoints, each focusing on a different type of recommendation algorithm:

1. **Content-based Recommender**: Leverage user and item attributes to make recommendations
2. **Sequence-based Recommender**: Use sequential patterns in user interactions
3. **Graph-based Recommender**: Exploit relationships between users and items

Detailed instructions for each checkpoint will be released later.

## Leaderboard

A competition leaderboard will track the performance of submitted algorithms. Submissions will be evaluated in hidden environments with the same setup but different random seeds to test robustness. More information on the leaderboard and submission process will be provided later.

## Baseline Recommenders

Several baseline algorithms are provided for comparison:

- `RandomRecommender`: Recommends random items
- `PopularityRecommender`: Recommends items based on popularity
- `ContentBasedRecommender`: Recommends items similar to previously liked items
- `EnhancedUCB`: Upper Confidence Bound algorithm with price consideration
- `HybridRecommender`: Combines multiple recommendation strategies

## Tips for Success

1. **Consider item prices**: Since revenue is the primary metric, recommending high-priced items that users are likely to purchase can be effective
2. **User segmentation**: Different user segments may have different preferences
3. **Content-based features**: Use user and item attributes for personalization
4. **Hybrid approaches**: Combine multiple recommendation strategies
5. **Exploration vs. exploitation**: Balance between recommending items you know users will like and discovering new items
6. **Iterative learning**: Update your model as new interaction data becomes available

## Questions and Support

For questions or support, please contact the course instructors or teaching assistants.

Good luck with your algorithms!
