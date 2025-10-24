#!/usr/bin/env python3
import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG-RerankerEval")

class RerankerEvaluator:
    """Evaluates answers using a cross-encoder reranker."""
    
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.logger = logger
        self.reranker = None
        self.model_name = model_name
        self._load_reranker()
        
    def _load_reranker(self):
        """Load the reranker model."""
        try:
            self.reranker = CrossEncoder(self.model_name)
            self.logger.info(f"Loaded reranker model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error loading reranker model: {str(e)}")
            raise
    
    def evaluate(self, answers_by_retriever, ground_truth):
        """Compare answers from all retrievers against ground truth using reranker."""
        self.logger.info("Evaluating answers with reranker")
        
        results = {
            "by_query": {},
            "aggregated": {
                "mean_score": {},
                "median_score": {},
                "wins": {},
                "top3": {},
                "mean_rank": {}
            }
        }
        
        # Initialize counters
        retrievers = list(answers_by_retriever.keys())
        retriever_scores = {r: [] for r in retrievers}
        retriever_ranks = {r: [] for r in retrievers}
        wins = {r: 0 for r in retrievers}
        top3_counts = {r: 0 for r in retrievers}
        
        # Find common queries 
        common_queries = set()
        for retriever, answers in answers_by_retriever.items():
            if not common_queries:
                common_queries = set(answers.keys())
            else:
                common_queries &= set(answers.keys())
        
        self.logger.info(f"Found {len(common_queries)} common queries across {len(retrievers)} retrievers")
        
        # Process each query
        for query_id in common_queries:
            if query_id not in ground_truth:
                self.logger.warning(f"Query {query_id} not found in ground truth, skipping")
                continue
                
            gt = ground_truth[query_id]
            query_results = {"scores": {}, "ranking": []}
            
            # Prepare pairs for reranking
            pairs = []
            pair_map = []  # To map back to retrievers
            
            for retriever in retrievers:
                answer = answers_by_retriever[retriever].get(query_id, {})
                if not answer:
                    continue
                
                if isinstance(answer, dict) and "answer" in answer:
                    answer_text = answer["answer"]
                else:
                    answer_text = answer
                
                # Skip empty answers
                if not answer_text or not answer_text.strip():
                    continue
                    
                pairs.append([gt, answer_text])
                pair_map.append(retriever)
            
            # Skip if no valid pairs
            if not pairs:
                self.logger.warning(f"No valid answers for query {query_id}")
                continue
                
            # Rerank with cross-encoder
            scores = self.reranker.predict(pairs)
            
            # Store scores and update counters
            for i, score in enumerate(scores):
                retriever = pair_map[i]
                query_results["scores"][retriever] = float(score)
                retriever_scores[retriever].append(float(score))
            
            # Rank retrievers by score
            ranked_retrievers = sorted(query_results["scores"].items(), 
                                      key=lambda x: x[1], reverse=True)
            
            query_results["ranking"] = [r for r, s in ranked_retrievers]
            
            # Update win count for the best retriever
            if ranked_retrievers:
                wins[ranked_retrievers[0][0]] += 1
            
            # Update top-3 counts
            for i, (retriever, _) in enumerate(ranked_retrievers[:3]):
                top3_counts[retriever] += 1
            
            # Update rank counters
            for i, (retriever, _) in enumerate(ranked_retrievers):
                rank = i + 1
                retriever_ranks[retriever].append(rank)
            
            # Store query results
            results["by_query"][query_id] = query_results
        
        # Calculate aggregated metrics
        for retriever in retrievers:
            scores = retriever_scores[retriever]
            ranks = retriever_ranks[retriever]
            
            if not scores:
                continue
                
            results["aggregated"]["mean_score"][retriever] = float(np.mean(scores))
            results["aggregated"]["median_score"][retriever] = float(np.median(scores))
            results["aggregated"]["wins"][retriever] = wins[retriever]
            results["aggregated"]["top3"][retriever] = top3_counts[retriever]
            results["aggregated"]["mean_rank"][retriever] = float(np.mean(ranks))
        
        return results

def create_comparison_tables(results, output_dir):
    """Create comparison tables for reranker results."""
    if not results:
        return
    
    aggregated = results["aggregated"]
    
    # Prepare data
    data = []
    for retriever in aggregated["mean_score"].keys():
        # Extract chunking and embedding type
        if "(" in retriever and ")" in retriever:
            chunking = retriever.split("(")[1].split(")")[0]
        else:
            chunking = "unknown"
            
        if "Content" in retriever:
            embedding = "content"
        elif "TF-IDF" in retriever:
            embedding = "tfidf"
        elif "Prefix" in retriever:
            embedding = "prefix"
        elif "Reranker" in retriever:
            if "TFIDF" in retriever:
                embedding = "reranker_tfidf"
            elif "Prefix" in retriever:
                embedding = "reranker_prefix"
            else:
                embedding = "reranker"
        else:
            embedding = "unknown"
        
        # Add to data
        data.append({
            "retriever": retriever,
            "chunking": chunking,
            "embedding": embedding,
            "mean_score": aggregated["mean_score"].get(retriever, 0),
            "median_score": aggregated["median_score"].get(retriever, 0),
            "wins": aggregated["wins"].get(retriever, 0),
            "top3": aggregated["top3"].get(retriever, 0),
            "mean_rank": aggregated["mean_rank"].get(retriever, 0)
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "reranker_comparison.csv"), index=False)
    
    # Create pivot tables
    if len(df["chunking"].unique()) > 1 and len(df["embedding"].unique()) > 1:
        # Mean score table
        score_pivot = df.pivot_table(index="chunking", columns="embedding", values="mean_score")
        score_pivot.to_csv(os.path.join(output_dir, "reranker_score_by_type.csv"))
        
        # Win count table
        win_pivot = df.pivot_table(index="chunking", columns="embedding", values="wins")
        win_pivot.to_csv(os.path.join(output_dir, "reranker_wins_by_type.csv"))
        
        # Mean rank table (lower is better)
        rank_pivot = df.pivot_table(index="chunking", columns="embedding", values="mean_rank")
        rank_pivot.to_csv(os.path.join(output_dir, "reranker_rank_by_type.csv"))

def load_answers(answers_dir):
    """Load answers from all retrievers into a structured dictionary."""
    answers_by_retriever = {}
    
    # List all answer files
    answer_files = [f for f in os.listdir(answers_dir) if f.endswith("_answers.json")]
    
    for file_name in answer_files:
        file_path = os.path.join(answers_dir, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            retriever_name = data.get("retriever_name", file_name.replace("_answers.json", ""))
            
            # Extract answers
            retriever_answers = {}
            for query_id, query_data in data.get("answers", {}).items():
                if isinstance(query_data, dict) and "answer" in query_data:
                    retriever_answers[query_id] = query_data["answer"]
                else:
                    retriever_answers[query_id] = query_data
            
            answers_by_retriever[retriever_name] = retriever_answers
            logger.info(f"Loaded {len(retriever_answers)} answers for {retriever_name}")
            
        except Exception as e:
            logger.error(f"Error loading {file_name}: {str(e)}")
    
    return answers_by_retriever

def load_ground_truth(gt_file):
    """Load ground truth answers."""
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        ground_truth = {}
        for query_id, query_data in gt_data.items():
            if isinstance(query_data, dict) and "answer" in query_data:
                ground_truth[query_id] = query_data["answer"]
            else:
                ground_truth[query_id] = query_data
        
        logger.info(f"Loaded {len(ground_truth)} ground truth answers from {gt_file}")
        return ground_truth
    
    except Exception as e:
        logger.error(f"Error loading ground truth: {str(e)}")
        return {}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Reranker Evaluation for RAG")
    
    parser.add_argument("--answers_dir", type=str, required=True,
                       help="Directory containing answer JSON files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to store evaluation results")
    parser.add_argument("--ground_truth", type=str, required=True,
                       help="Path to ground truth JSON file")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.answers_dir, "reranker_eval")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    ground_truth = load_ground_truth(args.ground_truth)
    answers_by_retriever = load_answers(args.answers_dir)
    
    # Check if we have data
    if not ground_truth:
        logger.error("No ground truth data loaded")
        return
    
    if not answers_by_retriever:
        logger.error("No answer data loaded")
        return
    
    # Initialize reranker evaluator
    reranker = RerankerEvaluator()
    
    # Run reranker evaluation
    logger.info("Running reranker-based comparison")
    results = reranker.evaluate(answers_by_retriever, ground_truth)
    
    # Save results
    results_path = os.path.join(args.output_dir, "reranker_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved reranker results to {results_path}")
    
    # Create comparison tables
    create_comparison_tables(results, args.output_dir)
    
    # Print summary
    logger.info("Reranker evaluation summary:")
    sorted_retrievers = sorted(
        results["aggregated"]["mean_score"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for retriever, score in sorted_retrievers:
        wins = results["aggregated"]["wins"].get(retriever, 0)
        top3 = results["aggregated"]["top3"].get(retriever, 0)
        rank = results["aggregated"]["mean_rank"].get(retriever, 0)
        logger.info(f"  {retriever}: Score={score:.4f}, Wins={wins}, Top3={top3}, Avg Rank={rank:.2f}")
    
    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main()