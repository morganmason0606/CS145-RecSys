"""
Evaluation script for recommendation system competition.
"""

import os
import time
import json
import importlib.util
import inspect
import argparse
from typing import Dict, Any, Optional, List, Tuple, Union
import traceback

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame

from sim4rec.modules import Simulator, EvaluateMetrics

from config import DEFAULT_CONFIG, EVALUATION_METRICS
from data_generator import CompetitionDataGenerator
from simulator import CompetitionSimulator
import sample_recommenders


def load_submission_module(submission_path: str) -> Any:
    """
    Load the submission module from the given path.
    
    Args:
        submission_path: Path to the submission module
        
    Returns:
        module: The loaded module
    """
    # Get the module name from the file path
    module_name = os.path.basename(submission_path).replace('.py', '')
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, submission_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module


def validate_submission(module: Any) -> Tuple[bool, str]:
    """
    Validate that the submission module meets the requirements.
    
    Args:
        module: The submission module to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    # Check for get_recommender function
    if not hasattr(module, 'get_recommender'):
        return False, "Submission must provide a get_recommender function"
    
    # Get the recommender class
    try:
        recommender = module.get_recommender()
    except Exception as e:
        return False, f"Error initializing recommender: {str(e)}"
    
    # Check that recommender has required methods
    required_methods = ['fit', 'predict']
    for method in required_methods:
        if not hasattr(recommender, method):
            return False, f"Recommender must implement {method} method"
    
    # Check method signatures
    try:
        # Check fit method signature
        fit_sig = inspect.signature(recommender.fit)
        required_params = ['log', 'user_features', 'item_features']
        for param in required_params:
            if param not in fit_sig.parameters:
                return False, f"fit method must accept {param} parameter"
        
        # Check predict method signature
        predict_sig = inspect.signature(recommender.predict)
        required_params = ['log', 'k', 'users', 'items', 'user_features', 'item_features', 'filter_seen_items']
        for param in required_params:
            if param not in predict_sig.parameters:
                return False, f"predict method must accept {param} parameter"
    except Exception as e:
        return False, f"Error validating method signatures: {str(e)}"
    
    # Check metadata
    if not hasattr(module, 'SUBMISSION_METADATA'):
        return False, "Submission must provide SUBMISSION_METADATA dictionary"
    
    metadata = module.SUBMISSION_METADATA
    required_fields = ['team_name', 'members', 'description']
    for field in required_fields:
        if field not in metadata:
            return False, f"SUBMISSION_METADATA must include {field}"
    
    return True, ""


def load_baseline_recommender(recommender_config: Dict[str, Any]) -> Any:
    """
    Load a baseline recommender from the sample_recommenders module.
    
    Args:
        recommender_config: Configuration for the recommender
        
    Returns:
        object: An instance of the recommender
    """
    # Get the recommender class
    class_name = recommender_config['class']
    if not hasattr(sample_recommenders, class_name):
        raise ValueError(f"Unknown recommender class: {class_name}")
    
    recommender_class = getattr(sample_recommenders, class_name)
    
    # Create an instance with the specified parameters
    parameters = recommender_config.get('parameters', {})
    
    # Check if we need to instantiate nested recommenders (for HybridRecommender)
    if 'recommenders' in parameters:
        nested_recommenders = []
        for nested_config in parameters['recommenders']:
            nested_class = getattr(sample_recommenders, nested_config['class'])
            nested_recommender = nested_class(**nested_config.get('parameters', {}))
            nested_recommenders.append(nested_recommender)
        parameters = {**parameters, 'recommenders': nested_recommenders}
    
    return recommender_class(**parameters)


def evaluate_recommender(
    recommender,
    simulator: CompetitionSimulator,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a recommender using the simulator.
    
    Args:
        recommender: Recommender instance to evaluate
        simulator: Simulator to use for evaluation
        config: Evaluation configuration
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    # Get evaluation settings
    n_iterations = config['simulation']['iterations']
    user_frac = config['simulation']['user_fraction']
    k = config['simulation']['k']
    filter_seen_items = config['simulation']['filter_seen_items']
    retrain = config['simulation']['retrain']
    
    # Check if we should use train-test split
    use_train_test_split = config['simulation'].get('train_test_split', False)
    
    # Run the simulation
    start_time = time.time()
    try:
        if use_train_test_split:
            # Get train-test split parameters
            train_iterations = config['simulation'].get('train_iterations', n_iterations // 2)
            test_iterations = config['simulation'].get('test_iterations', n_iterations - train_iterations)
            
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
            
            # Combine metrics for compatibility with existing code
            metrics_history = train_metrics + test_metrics
            revenue_history = train_revenue + test_revenue
        else:
            # Run standard simulation
            metrics_history, revenue_history = simulator.run_simulation(
                recommender=recommender,
                n_iterations=n_iterations,
                user_frac=user_frac,
                k=k,
                filter_seen_items=filter_seen_items,
                retrain=retrain
            )
    except Exception as e:
        error_msg = f"Error during simulation: {str(e)}\n{traceback.format_exc()}"
        return {
            "success": False,
            "error": error_msg,
            "runtime_seconds": time.time() - start_time
        }
    
    # Calculate total revenue
    total_revenue = sum(revenue_history)
    avg_revenue = np.mean(revenue_history)
    
    # Calculate average metrics
    avg_metrics = {}
    for i, metrics in enumerate(metrics_history):
        for metric_name, value in metrics.items():
            if metric_name not in avg_metrics:
                avg_metrics[metric_name] = []
            avg_metrics[metric_name].append(value)
    
    for metric_name, values in avg_metrics.items():
        avg_metrics[metric_name] = np.mean(values)
    
    # Create results dictionary
    results = {
        "success": True,
        "total_revenue": total_revenue,
        "avg_revenue_per_iteration": avg_revenue,
        "avg_metrics": avg_metrics,
        "metrics_history": metrics_history,
        "revenue_history": revenue_history,
        "runtime_seconds": time.time() - start_time
    }
    
    # Add train-test specific results if used
    if use_train_test_split:
        # Calculate train metrics
        train_avg_metrics = {}
        for i, metrics in enumerate(train_metrics):
            for metric_name, value in metrics.items():
                if metric_name not in train_avg_metrics:
                    train_avg_metrics[metric_name] = []
                train_avg_metrics[metric_name].append(value)
        
        for metric_name, values in train_avg_metrics.items():
            train_avg_metrics[metric_name] = np.mean(values)
        
        # Calculate test metrics
        test_avg_metrics = {}
        for i, metrics in enumerate(test_metrics):
            for metric_name, value in metrics.items():
                if metric_name not in test_avg_metrics:
                    test_avg_metrics[metric_name] = []
                test_avg_metrics[metric_name].append(value)
        
        for metric_name, values in test_avg_metrics.items():
            test_avg_metrics[metric_name] = np.mean(values)
        
        # Add to results
        results.update({
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_revenue": train_revenue,
            "test_revenue": test_revenue,
            "train_total_revenue": sum(train_revenue),
            "test_total_revenue": sum(test_revenue),
            "train_avg_revenue": np.mean(train_revenue),
            "test_avg_revenue": np.mean(test_revenue),
            "train_avg_metrics": train_avg_metrics,
            "test_avg_metrics": test_avg_metrics
        })
    
    return results


def evaluate_submission(
    submission_path: str,
    output_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    data_dir: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate a submission.
    
    Args:
        submission_path: Path to the submission module
        output_path: Path to save the evaluation results
        config: Configuration for the evaluation
        data_dir: Directory with pre-generated data
        seed: Random seed for reproducibility
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    # Use default config if not provided
    if config is None:
        config = DEFAULT_CONFIG
    
    # Override seed if provided
    if seed is not None:
        config['data_generation']['seed'] = seed
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("RecSysCompetition") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    
    try:
        # Load the submission module
        submission_module = load_submission_module(submission_path)
        
        # Validate the submission
        is_valid, error_message = validate_submission(submission_module)
        if not is_valid:
            return {
                "success": False,
                "error": error_message
            }
        
        # Initialize data generator
        data_generator = CompetitionDataGenerator(
            spark_session=spark,
            **config['data_generation']
        )
        
        # Load or generate data
        if data_dir is not None and os.path.exists(data_dir):
            # Load pre-generated data
            data_generator.load_data(data_dir)
        else:
            # Generate new data
            data_generator.generate_users()
            data_generator.generate_items()
            data_generator.generate_initial_history(config['data_generation']['initial_history_density'])
            
            # Save data if output path is provided
            if output_path is not None:
                data_dir = os.path.join(output_path, "data")
                data_generator.save_data(data_dir)
        
        # Setup data generators for simulator
        user_generator, item_generator = data_generator.setup_data_generators()
        
        # Initialize simulator
        simulator_config = config['simulation']
        simulator = CompetitionSimulator(
            user_generator=user_generator,
            item_generator=item_generator,
            data_dir=os.path.join(simulator_config['data_dir'], "simulation"),
            log_df=data_generator.history_df,
            conversion_noise_mean=simulator_config['conversion_noise_mean'],
            conversion_noise_std=simulator_config['conversion_noise_std'],
            spark_session=spark,
            seed=config['data_generation']['seed']
        )
        
        # Get the recommender
        recommender = submission_module.get_recommender()
        
        # Evaluate the recommender
        results = evaluate_recommender(
            recommender=recommender,
            simulator=simulator,
            config=config
        )
        
        # Add metadata
        results['metadata'] = submission_module.SUBMISSION_METADATA
        
        # Save results if output path is provided
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            results_path = os.path.join(output_path, "results.json")
            
            # Convert numpy values to Python native types
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Save as JSON
            with open(results_path, 'w') as f:
                json.dump(results, f, default=convert_numpy, indent=2)
        
        return results
    
    finally:
        # Stop Spark session
        spark.stop()


def evaluate_all_baselines(
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    data_dir: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all baseline recommenders.
    
    Args:
        output_dir: Directory to save the evaluation results
        config: Configuration for the evaluation
        data_dir: Directory with pre-generated data
        seed: Random seed for reproducibility
        
    Returns:
        Dict[str, Dict[str, Any]]: Evaluation results for each baseline
    """
    # Use default config if not provided
    if config is None:
        config = DEFAULT_CONFIG
    
    # Override seed if provided
    if seed is not None:
        config['data_generation']['seed'] = seed
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("RecSysCompetition") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    
    try:
        # Initialize data generator
        data_generator = CompetitionDataGenerator(
            spark_session=spark,
            **config['data_generation']
        )
        
        # Load or generate data
        if data_dir is not None and os.path.exists(data_dir):
            # Load pre-generated data
            data_generator.load_data(data_dir)
        else:
            # Generate new data
            data_generator.generate_users()
            data_generator.generate_items()
            data_generator.generate_initial_history(config['data_generation']['initial_history_density'])
            
            # Save data
            data_dir = os.path.join(output_dir, "data")
            data_generator.save_data(data_dir)
        
        # Setup data generators for simulator
        user_generator, item_generator = data_generator.setup_data_generators()
        
        # Get baseline recommenders
        baselines = []
        from config import BASELINE_RECOMMENDERS
        for recommender_config in BASELINE_RECOMMENDERS:
            try:
                recommender = load_baseline_recommender(recommender_config)
                baselines.append((recommender_config['name'], recommender))
            except Exception as e:
                print(f"Error loading baseline {recommender_config['name']}: {str(e)}")
        
        # Initialize simulator
        simulator_config = config['simulation']
        all_results = {}
        
        for name, recommender in baselines:
            print(f"Evaluating baseline: {name}")
            
            # Create a separate simulator for each baseline
            simulator = CompetitionSimulator(
                user_generator=user_generator,
                item_generator=item_generator,
                data_dir=os.path.join(simulator_config['data_dir'], f"simulation_{name}"),
                log_df=data_generator.history_df,
                conversion_noise_mean=simulator_config['conversion_noise_mean'],
                conversion_noise_std=simulator_config['conversion_noise_std'],
                spark_session=spark,
                seed=config['data_generation']['seed']
            )
            
            # Evaluate the recommender
            try:
                results = evaluate_recommender(
                    recommender=recommender,
                    simulator=simulator,
                    config=config
                )
                all_results[name] = results
                
                # Save individual results
                baseline_dir = os.path.join(output_dir, f"baseline_{name}")
                os.makedirs(baseline_dir, exist_ok=True)
                
                # Convert numpy values to Python native types
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                # Save as JSON
                with open(os.path.join(baseline_dir, "results.json"), 'w') as f:
                    json.dump(results, f, default=convert_numpy, indent=2)
                
                print(f"  Total Revenue: {results['total_revenue']}")
                print(f"  Avg Revenue/Iteration: {results['avg_revenue_per_iteration']}")
                
            except Exception as e:
                error_msg = f"Error evaluating {name}: {str(e)}\n{traceback.format_exc()}"
                all_results[name] = {
                    "success": False,
                    "error": error_msg
                }
                print(f"  Error: {str(e)}")
        
        # Create a comparison table
        comparison = []
        for name, results in all_results.items():
            if results['success']:
                comparison.append({
                    'name': name,
                    'total_revenue': results['total_revenue'],
                    'avg_revenue_per_iteration': results['avg_revenue_per_iteration'],
                    'runtime_seconds': results['runtime_seconds']
                })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('total_revenue', ascending=False).reset_index(drop=True)
        
        # Save comparison table
        comparison_df.to_csv(os.path.join(output_dir, "baseline_comparison.csv"), index=False)
        
        # Also save as markdown table
        with open(os.path.join(output_dir, "baseline_comparison.md"), 'w') as f:
            f.write("# Baseline Recommender Comparison\n\n")
            f.write(comparison_df.to_markdown(index=False))
        
        return all_results
    
    finally:
        # Stop Spark session
        spark.stop()


def create_leaderboard(
    evaluations_dir: str,
    output_path: Optional[str] = None,
    metric: str = "total_revenue",
    use_test_metrics: bool = False
) -> pd.DataFrame:
    """
    Create a leaderboard from evaluation results.
    
    Args:
        evaluations_dir: Directory containing evaluation results
        output_path: Path to save the leaderboard
        metric: Metric to rank by
        use_test_metrics: Whether to use test metrics for ranking (if train-test split was used)
        
    Returns:
        pd.DataFrame: Leaderboard
    """
    # Load all evaluation results
    results = []
    for root, dirs, files in os.walk(evaluations_dir):
        for file in files:
            if file == "results.json":
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        evaluation = json.load(f)
                    
                    # Skip failed evaluations
                    if not evaluation.get('success', False):
                        continue
                    
                    # Get metadata
                    metadata = evaluation.get('metadata', {})
                    team_name = metadata.get('team_name', os.path.basename(root))
                    
                    # Extract metrics
                    row = {
                        'team_name': team_name,
                        'submission_path': root
                    }
                    
                    # Check if train-test split was used
                    has_train_test_split = 'train_total_revenue' in evaluation and 'test_total_revenue' in evaluation
                    
                    # Add primary metric
                    if use_test_metrics and has_train_test_split and 'test_' + metric in evaluation:
                        row[metric] = evaluation['test_' + metric]
                    elif use_test_metrics and has_train_test_split and metric in evaluation.get('test_avg_metrics', {}):
                        row[metric] = evaluation['test_avg_metrics'][metric]
                    elif metric in evaluation:
                        row[metric] = evaluation[metric]
                    elif metric in evaluation.get('avg_metrics', {}):
                        row[metric] = evaluation['avg_metrics'][metric]
                    
                    # Add secondary metrics
                    ranking_metrics = ["discounted_revenue", "precision_at_k", "recall_at_k", 
                                      "ndcg_at_k", "mrr", "hit_rate"]
                    
                    for metric_name, config in EVALUATION_METRICS.items():
                        if metric_name != metric:
                            if use_test_metrics and has_train_test_split and 'test_' + metric_name in evaluation:
                                row[metric_name] = evaluation['test_' + metric_name]
                            elif use_test_metrics and has_train_test_split and metric_name in evaluation.get('test_avg_metrics', {}):
                                row[metric_name] = evaluation['test_avg_metrics'][metric_name]
                            elif metric_name in evaluation:
                                row[metric_name] = evaluation[metric_name]
                            elif metric_name in evaluation.get('avg_metrics', {}):
                                row[metric_name] = evaluation['avg_metrics'][metric_name]
                    
                    # Add train-test specific metrics if available
                    if has_train_test_split:
                        for prefix in ['train_', 'test_']:
                            for suffix in ['total_revenue', 'avg_revenue']:
                                key = prefix + suffix
                                if key in evaluation:
                                    row[key] = evaluation[key]
                    
                    row['runtime_seconds'] = evaluation.get('runtime_seconds', None)
                    
                    results.append(row)
                except Exception as e:
                    print(f"Error loading {os.path.join(root, file)}: {str(e)}")
    
    # Create leaderboard dataframe
    leaderboard = pd.DataFrame(results)
    
    # Sort by the primary metric
    if metric in leaderboard.columns:
        higher_is_better = EVALUATION_METRICS.get(metric, {}).get('higher_is_better', True)
        leaderboard = leaderboard.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)
    
    # Save leaderboard if output path is provided
    if output_path is not None:
        leaderboard.to_csv(output_path, index=False)
        
        # Also save a formatted markdown file with more readable metrics
        md_output_path = output_path.replace('.csv', '.md')
        with open(md_output_path, 'w') as f:
            f.write("# Recommendation System Competition Leaderboard\n\n")
            
            # Create a more user-friendly table
            f.write("## Overall Results\n\n")
            
            # Format the primary metric column
            if metric in leaderboard.columns:
                metric_name = EVALUATION_METRICS.get(metric, {}).get('name', metric)
                f.write(f"Ranked by: **{metric_name}** ")
                
                if use_test_metrics:
                    f.write("(Test set performance)\n\n")
                else:
                    f.write("\n\n")
                
                # Create a simpler table with key metrics
                key_metrics = ["team_name", metric]
                
                # Add train/test specific metrics if available
                has_train_test = any(col.startswith('train_') or col.startswith('test_') for col in leaderboard.columns)
                if has_train_test:
                    key_metrics.extend(["train_total_revenue", "test_total_revenue"])
                
                key_metrics.extend(["conversion_rate", "ndcg_at_k", "mrr"])
                key_metrics = [m for m in key_metrics if m in leaderboard.columns]
                
                # Create header
                header = []
                for col in key_metrics:
                    if col in EVALUATION_METRICS:
                        header.append(EVALUATION_METRICS[col]['name'])
                    else:
                        header.append(col.replace('_', ' ').title())
                
                f.write("| " + " | ".join(header) + " |\n")
                f.write("| " + " | ".join(["---"] * len(header)) + " |\n")
                
                # Add rows
                for _, row in leaderboard.iterrows():
                    values = []
                    for col in key_metrics:
                        if pd.isna(row[col]):
                            values.append("N/A")
                        elif isinstance(row[col], (int, float)):
                            values.append(f"{row[col]:.4f}")
                        else:
                            values.append(str(row[col]))
                    
                    f.write("| " + " | ".join(values) + " |\n")
                
                # Add detailed metrics section
                f.write("\n## Detailed Metrics\n\n")
                for i, (_, row) in enumerate(leaderboard.iterrows()):
                    f.write(f"### {i+1}. {row['team_name']}\n\n")
                    
                    # Organize metrics in a logical way
                    metrics_table = []
                    
                    # First add revenue metrics
                    if has_train_test:
                        metrics_table.append(("Training Total Revenue", row.get('train_total_revenue', 'N/A')))
                        metrics_table.append(("Testing Total Revenue", row.get('test_total_revenue', 'N/A')))
                        metrics_table.append(("Training Avg Revenue", row.get('train_avg_revenue', 'N/A')))
                        metrics_table.append(("Testing Avg Revenue", row.get('test_avg_revenue', 'N/A')))
                    else:
                        metrics_table.append(("Total Revenue", row.get('total_revenue', 'N/A')))
                        metrics_table.append(("Avg Revenue", row.get('avg_revenue_per_iteration', 'N/A')))
                    
                    # Then add ranking metrics
                    for col in leaderboard.columns:
                        if col in ["team_name", "submission_path", "runtime_seconds"]:
                            continue
                        if col in ["total_revenue", "avg_revenue_per_iteration", "train_total_revenue", 
                                  "test_total_revenue", "train_avg_revenue", "test_avg_revenue"]:
                            continue
                            
                        metric_name = col
                        if col in EVALUATION_METRICS:
                            metric_name = EVALUATION_METRICS[col]['name']
                        
                        value = row[col]
                        if pd.isna(value):
                            value = "N/A"
                        elif isinstance(value, (int, float)):
                            value = f"{value:.6f}"
                        
                        metrics_table.append((metric_name, value))
                    
                    # Add runtime at the end
                    runtime = row.get('runtime_seconds', 'N/A')
                    if not pd.isna(runtime):
                        metrics_table.append(("Runtime (seconds)", f"{float(runtime):.2f}"))
                    
                    f.write("| Metric | Value |\n")
                    f.write("| --- | --- |\n")
                    for name, value in metrics_table:
                        f.write(f"| {name} | {value} |\n")
                    
                    f.write("\n")
    
    return leaderboard


def main():
    """
    Command line interface for evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate recommendation algorithms")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Evaluate submission
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a submission")
    eval_parser.add_argument("submission_path", help="Path to the submission module")
    eval_parser.add_argument("--output-path", help="Path to save the evaluation results")
    eval_parser.add_argument("--data-dir", help="Directory with pre-generated data")
    eval_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    eval_parser.add_argument("--train-test", action="store_true", help="Use train-test split evaluation")
    eval_parser.add_argument("--train-iterations", type=int, help="Number of iterations for training")
    eval_parser.add_argument("--test-iterations", type=int, help="Number of iterations for testing")
    
    # Evaluate all baselines
    baseline_parser = subparsers.add_parser("baselines", help="Evaluate all baseline recommenders")
    baseline_parser.add_argument("output_dir", help="Directory to save the evaluation results")
    baseline_parser.add_argument("--data-dir", help="Directory with pre-generated data")
    baseline_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    baseline_parser.add_argument("--train-test", action="store_true", help="Use train-test split evaluation")
    baseline_parser.add_argument("--train-iterations", type=int, help="Number of iterations for training")
    baseline_parser.add_argument("--test-iterations", type=int, help="Number of iterations for testing")
    
    # Create leaderboard
    leaderboard_parser = subparsers.add_parser("leaderboard", help="Create a leaderboard from evaluation results")
    leaderboard_parser.add_argument("evaluations_dir", help="Directory containing evaluation results")
    leaderboard_parser.add_argument("--output-path", help="Path to save the leaderboard")
    leaderboard_parser.add_argument("--metric", default="total_revenue", help="Metric to rank by")
    leaderboard_parser.add_argument("--use-test-metrics", action="store_true", help="Use test metrics for ranking (if train-test split was used)")
    
    args = parser.parse_args()
    
    if args.command == "evaluate":
        # Update config with train-test split options if provided
        config = DEFAULT_CONFIG.copy()
        if args.train_test:
            config['simulation']['train_test_split'] = True
        if args.train_iterations:
            config['simulation']['train_iterations'] = args.train_iterations
        if args.test_iterations:
            config['simulation']['test_iterations'] = args.test_iterations
        
        # Evaluate a submission
        results = evaluate_submission(
            submission_path=args.submission_path,
            output_path=args.output_path,
            config=config,
            data_dir=args.data_dir,
            seed=args.seed
        )
        
        if results['success']:
            print("Evaluation successful")
            print(f"Total Revenue: {results['total_revenue']}")
            
            # Print train-test results if available
            if 'train_total_revenue' in results and 'test_total_revenue' in results:
                print(f"Train Revenue: {results['train_total_revenue']}")
                print(f"Test Revenue: {results['test_total_revenue']}")
                print(f"Train/Test Revenue Ratio: {results['train_total_revenue'] / results['test_total_revenue']:.2f}")
            else:
                print(f"Avg Revenue/Iteration: {results['avg_revenue_per_iteration']}")
        else:
            print("Evaluation failed")
            print(f"Error: {results['error']}")
    
    elif args.command == "baselines":
        # Update config with train-test split options if provided
        config = DEFAULT_CONFIG.copy()
        if args.train_test:
            config['simulation']['train_test_split'] = True
        if args.train_iterations:
            config['simulation']['train_iterations'] = args.train_iterations
        if args.test_iterations:
            config['simulation']['test_iterations'] = args.test_iterations
        
        # Evaluate all baselines
        evaluate_all_baselines(
            output_dir=args.output_dir,
            config=config,
            data_dir=args.data_dir,
            seed=args.seed
        )
    
    elif args.command == "leaderboard":
        # Create leaderboard
        leaderboard = create_leaderboard(
            evaluations_dir=args.evaluations_dir,
            output_path=args.output_path,
            metric=args.metric,
            use_test_metrics=args.use_test_metrics
        )
        
        print(leaderboard)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 