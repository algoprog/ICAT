import os
import json
import torch
import time
import hashlib
import argparse
import logging
import re

import numpy as np
import jsonlines
import requests

from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from retriever import Retriever
from llm_eval import LLMEvaluator

os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN'
os.environ['BRAVE_API_KEY'] = 'YOUR_BRAVE_API_KEY'

CACHE_PATH = 'hf_cache'
os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH
os.environ['HF_HOME'] = CACHE_PATH
os.environ['HF_DATASETS_CACHE'] = CACHE_PATH
os.environ['TORCH_HOME'] = CACHE_PATH

class ICAT:
    def __init__(self, 
                 nli_model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                 corpus_path: Optional[str] = None,
                 qrels_path: Optional[str] = None,
                 queries_path: str = "queries.jsonl",
                 nli_batch_size: int = 8,
                 llm_batch_size: int = 4,
                 api_base_llm: str = "meta-llama/Llama-3.3-70B-Instruct",
                 api_facts_llm: str = None,
                 use_web_search: bool = False):
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initializing CoverageScore...")
        self.use_web_search = use_web_search
        self.topk = 10

        if not use_web_search:
            if corpus_path is None:
                raise ValueError("corpus_path must be provided when not using web search")
            self.retriever = Retriever()
            self.retriever.process_corpus(corpus_path)
        
        self.llm_evaluator = LLMEvaluator(api_base_llm=api_base_llm, api_facts_llm=api_facts_llm)
        self.qrels_lookup = {}
        
        # Make qrels loading optional
        self.qrels = None
        if qrels_path:
            self.logger.info(f"Loading qrels from {qrels_path}")
            with open(qrels_path, "r") as f:
                self.qrels = [json.loads(line) for line in f]
                for qrel in self.qrels:
                    if qrel["relevance"] == 1:
                        if qrel["doc_id"] not in self.qrels_lookup:
                            self.qrels_lookup[qrel["doc_id"]] = []
                        self.qrels_lookup[qrel["doc_id"]].append(qrel["subtopic_id"])

        # Initialize NLI model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
        self.nli_model.eval()

        self.nli_batch_size = nli_batch_size
        self.llm_batch_size = llm_batch_size

        # Load queries and create query ID mapping
        self.logger.info(f"Loading queries from {queries_path}")
        self.queries_data = []
        self.query_id_map = {}  # New mapping dictionary
        with jsonlines.open(queries_path) as reader:
            for obj in reader:
                query_data = {
                    'query_id': obj['query_id'],
                    'query': obj['query'],
                    'subtopics': obj['subtopics']
                }
                self.queries_data.append(query_data)
                self.query_id_map[obj['query_id']] = query_data

    def _retrieve_documents(self, query: str) -> List[Tuple[str, str, float]]:
        """
        Unified retrieval method that either uses regular retrieval or web search
        
        Args:
            query (str): The query to search for
            
        Returns:
            List[Tuple[str, str, float]]: List of (doc_id/title, content/snippet, score) tuples
        """
        if self.use_web_search:
            results = self._brave_search(query)
            # Convert web results to same format as regular retrieval
            # Use title as doc_id, combine title and snippet as content, use 1.0 as placeholder score
            return [(title, f"{title} {snippet}", 1.0) for title, snippet in results]
        else:
            return self.retriever.retrieve(query, top_k=self.topk)

    def _brave_search(self, query: str, max_results: int = 10, max_retries: int = 5, retry_delay: int = 1) -> List[Tuple[str, str]]:
        """
        Perform a search using Brave Search API and return top 10 results.
        
        Args:
            query (str): The search query
            
        Returns:
            List[Tuple[str, str]]: List of (title, snippet) tuples
        """
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": os.environ['BRAVE_API_KEY']
        }
        
        params = {
            "q": query,
            "count": max_results  # Request 10 results
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()  # Raise exception for bad status codes
                
                results = response.json()
                
                # Extract titles and snippets from results
                search_results = []
                for web_page in results.get("web", {}).get("results", []):
                    title = web_page.get("title", "")
                    snippet = web_page.get("description", "")
                    
                    # Clean HTML tags from title and snippet using a simple regex
                    title = re.sub(r'<[^>]+>', '', title)
                    snippet = re.sub(r'<[^>]+>', '', snippet)
                    
                    search_results.append((title, snippet))
                
                return search_results
            
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:  # Don't log on last attempt
                    self.logger.warning(f"Brave Search API request attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Error making Brave Search API request after {max_retries} attempts: {e}")
            except (KeyError, ValueError) as e:
                self.logger.error(f"Error parsing Brave Search API response: {e}")
                break  # Don't retry on parsing errors
        
        return []

    def _check_entailment(self, premise: str, hypothesis: str) -> bool:
        inputs = self.nli_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.nli_model(**inputs)
        
        prediction = torch.softmax(output.logits[0], -1).cpu().numpy()
        return bool(prediction[0] > 0.5)  # Index 0 corresponds to entailment

    def _check_entailment_batch(self, premises: List[str], hypotheses: List[str]) -> List[bool]:
        # Process multiple premise-hypothesis pairs at once
        inputs = self.nli_tokenizer(
            premises, 
            hypotheses, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.nli_model(**inputs)
        
        predictions = torch.softmax(output.logits, -1).cpu().numpy()
        return [bool(pred[0] > 0.5) for pred in predictions]  # Index 0 corresponds to entailment

    def icat_score_m(self, model_responses: List[str], query_ids: Optional[List[str]] = None, queries: Optional[List[str]] = None) -> Tuple[List[Dict], Dict]:
        # Check if web search is being used
        if self.use_web_search:
            raise ValueError("icat_score_m cannot be used with web search. This method requires a local corpus.")
            
        # If query_ids and queries not provided, use all from loaded files
        if query_ids is None or queries is None:
            query_ids = [q['query_id'] for q in self.queries_data]
            queries = [q['query'] for q in self.queries_data]
        
        self.logger.info(f"Processing {len(queries)} queries")
        
        # Map query_ids to their data from self.queries_data
        query_data = [self.query_id_map[qid] for qid in query_ids]
        
        results = []
        
        # Process queries in batches
        for i in range(0, len(queries), self.llm_batch_size):
            batch_queries = queries[i:i + self.llm_batch_size]
            batch_responses = model_responses[i:i + self.llm_batch_size]
            batch_query_data = query_data[i:i + self.llm_batch_size]
            
            # Generate atomic facts for batch
            self.logger.info(f"Generating facts for batch {i//self.llm_batch_size + 1}")
            batch_facts = self.llm_evaluator.generate_facts(batch_responses)
            
            # Process each query in the batch
            for query, response, facts, current_query_data in zip(batch_queries, batch_responses, batch_facts, batch_query_data):
                # Track entailed facts and retrieved docs
                entailed_facts_count = 0
                retrieved_docs = []
                fact_results = []
                covered_subtopics = []
                
                # Process facts in batches
                for j in range(0, len(facts), self.nli_batch_size):
                    batch_facts_subset = facts[j:j + self.nli_batch_size]
                    
                    for fact in batch_facts_subset:
                        top_docs = self._retrieve_documents(fact)
                        fact_doc_results = []
                        is_fact_entailed = False
                        
                        # Batch process entailment checks
                        premises = [doc[1] for doc in top_docs]
                        hypotheses = [fact] * len(premises)
                        entailment_results = self._check_entailment_batch(premises, hypotheses)
                        
                        for (doc_id, snippet, score), is_entailed in zip(top_docs, entailment_results):
                            fact_doc_results.append({
                                "doc_id": doc_id,
                                "snippet": snippet,
                                "score": score,
                                "is_entailed": str(is_entailed)
                            })
                            if is_entailed:
                                is_fact_entailed = True
                                retrieved_docs.append(doc_id)
                                # Use lookup dictionary instead of filtering
                                if doc_id in self.qrels_lookup:
                                    for subtopic_id in self.qrels_lookup[doc_id]:
                                        covered_subtopics.append({
                                            "subtopic_id": subtopic_id,
                                            "supporting_fact": fact
                                        })
                                break
                        
                        if is_fact_entailed:
                            entailed_facts_count += 1
                        elif not is_fact_entailed:
                            retrieved_docs.append(top_docs[0][0])
                        
                        fact_results.append({
                            "fact": fact,
                            "retrieved_docs": fact_doc_results
                        })
                
                # Calculate metrics using covered subtopics
                total_subtopics = len(current_query_data['subtopics'])
                unique_covered_subtopics = len(set(s["subtopic_id"] for s in covered_subtopics))
                coverage = unique_covered_subtopics / total_subtopics if total_subtopics > 0 else 0
                factuality = entailed_facts_count / len(facts) if facts else 0
                f1 = 2 * (factuality * coverage) / (factuality + coverage) if (factuality + coverage) > 0 else 0
                
                results.append({
                    "query_id": current_query_data['query_id'],
                    "query": query,
                    "facts": fact_results,
                    "metrics": {
                        "coverage": coverage,
                        "factuality": factuality,
                        "f1": f1
                    }
                })
        
        # Calculate aggregate metrics
        avg_coverage = sum(r["metrics"]["coverage"] for r in results) / len(results)
        avg_factuality = sum(r["metrics"]["factuality"] for r in results) / len(results)
        avg_f1 = sum(r["metrics"]["f1"] for r in results) / len(results)
        
        aggregate_metrics = {
            "avg_coverage": avg_coverage,
            "avg_factuality": avg_factuality,
            "avg_f1": avg_f1
        }
        
        return results, aggregate_metrics

    def icat_score_s(self, model_responses: List[str], query_ids: Optional[List[str]] = None, queries: Optional[List[str]] = None) -> Tuple[List[Dict], Dict]:
        assert len(model_responses) == len(query_ids) if query_ids is not None else True
        assert len(model_responses) == len(queries) if queries is not None else True
        
        # If query_ids and queries not provided, use all from loaded files
        if query_ids is None or queries is None:
            query_ids = [q['query_id'] for q in self.queries_data]
            queries = [q['query'] for q in self.queries_data]
        
        self.logger.info(f"Processing {len(queries)} queries")
        
        # Map query_ids to their data - FIXED: Use query_id_map instead of index
        query_data = [self.query_id_map[qid] for qid in query_ids]
        
        # Get atomic facts for all responses
        self.logger.info("Generating atomic facts from responses...")
        all_facts = self.llm_evaluator.generate_facts(model_responses)
        
        # Process entailment for each query's facts in batches
        all_entailed_facts = []
        for query_idx, (query, response_facts) in enumerate(zip(query_data, all_facts)):
            self.logger.info(f"Processing entailment for query {query_idx + 1}")
            entailed_facts = []
            
            # Batch process facts for entailment
            for i in range(0, len(response_facts), self.nli_batch_size):
                batch_facts = response_facts[i:i + self.nli_batch_size]
                batch_results = []
                
                for fact in batch_facts:
                    top_docs = self._retrieve_documents(fact)
                    premises = [doc[1] for doc in top_docs]
                    hypotheses = [fact] * len(premises)
                    
                    entailment_results = self._check_entailment_batch(premises, hypotheses)
                    
                    if any(entailment_results):
                        for is_entailed, doc in zip(entailment_results, top_docs):
                            if is_entailed:
                                batch_results.append((doc[0], len(entailed_facts), fact))
                                break
                
                entailed_facts.extend(batch_results)
            
            all_entailed_facts.append(entailed_facts)

        # Collect all coverage prompts using ground truth subtopics
        coverage_prompts = []
        for query_idx, (query, response_facts) in enumerate(zip(query_data, all_entailed_facts)):
            entailed_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate([f[2] for f in response_facts])])
            # FIXED: Use query_data instead of queries_data
            subtopics = query['subtopics']
            
            coverage_prompt = (
                f'given this query "{query}", the following list of subtopics:\n\n' +
                "\n".join([f"{j+1} : {topic}" for j, topic in enumerate(subtopics)]) + "\n\n" +
                f'return the subtopics that are covered in the given text below with a list of facts, '
                f'mention each subtopic only once with a list of fact numbers for each subtopic, '
                f'the fact numbers should reference the most relevant facts that support the subtopic, '
                f'they should be explicitly mentioned in the given text, if they are not explicitly mentioned '
                f'don\'t include them in your response, if some subtopic is not covered without any evidence '
                f'don\'t include it in your response, use this jsonl format '
                f'{{"topic_id": ..., "evidence": [fact_number, ...]}}, one json object per line, '
                f'here is the text with enumerated facts:\n\n{entailed_text}'
            )
            coverage_prompts.append(coverage_prompt)

        # Process coverage prompts in batches
        covered_topics_responses = []
        for i in range(0, len(coverage_prompts), self.llm_batch_size):
            batch_prompts = coverage_prompts[i:i + self.llm_batch_size]
            batch_responses = self.llm_evaluator.generate(batch_prompts)
            covered_topics_responses.extend(batch_responses)

        results = []
        for query_idx, (response_facts, covered_topics_raw) in enumerate(zip(all_entailed_facts, covered_topics_responses)):
            total_topics = len(query_data[query_idx]['subtopics'])
            covered_data = []
            seen_topic_ids = set()
            
            query_atomic_facts = all_facts[query_idx]
            total_facts = len(query_atomic_facts)
            
            # Parse coverage response
            for line in covered_topics_raw.split('\n'):
                if line.strip().startswith('{"topic_id":'):
                    try:
                        data = json.loads(line.strip())
                        # Add validation for topic_id
                        if data.get("topic_id") is None:
                            continue
                            
                        # Try to convert topic_id to int, skip if invalid
                        try:
                            topic_id = int(data["topic_id"]) - 1
                        except (ValueError, TypeError):
                            continue
                        
                        # Skip if evidence list is empty or missing
                        if not data.get("evidence"):
                            continue
                        
                        if (0 <= topic_id < total_topics) and (topic_id not in seen_topic_ids):
                            valid_evidence = []
                            for fact_num in data["evidence"]:
                                try:
                                    fact_idx = int(fact_num) - 1
                                    if 0 <= fact_idx < total_facts:
                                        valid_evidence.append(fact_idx)
                                except (ValueError, TypeError):
                                    continue
                            
                            if valid_evidence:
                                seen_topic_ids.add(topic_id)
                                covered_data.append({
                                    "topic_id": topic_id + 1,
                                    "evidence": valid_evidence
                                })
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue

            coverage = len(covered_data) / total_topics if total_topics > 0 else 0
            factuality = len(response_facts) / total_facts if total_facts > 0 else 0
            f1 = 2 * (factuality * coverage) / (factuality + coverage) if (factuality + coverage) > 0 else 0

            self.logger.info(f"Query {query_idx + 1} metrics - Coverage: {coverage:.2f}, Factuality: {factuality:.2f}, F1: {f1:.2f}")

            results.append({
                "query_id": query_data[query_idx]['query_id'],
                "query": query_data[query_idx]['query'],
                "subtopics": query_data[query_idx]['subtopics'],
                "atomic_facts": query_atomic_facts,
                "entailed_facts": [f[2] for f in response_facts],
                "covered_topics": covered_data,
                "metrics": {
                    "coverage": coverage,
                    "factuality": factuality,
                    "f1": f1
                }
            })
        
        # Calculate aggregate metrics
        avg_coverage = sum(r["metrics"]["coverage"] for r in results) / len(results)
        avg_factuality = sum(r["metrics"]["factuality"] for r in results) / len(results)
        avg_f1 = sum(r["metrics"]["f1"] for r in results) / len(results)
        
        aggregate_metrics = {
            "avg_coverage": avg_coverage,
            "avg_factuality": avg_factuality,
            "avg_f1": avg_f1
        }
        
        return results, aggregate_metrics

    def icat_score_a(self, model_responses: List[str], query_ids: Optional[List[str]] = None, queries: Optional[List[str]] = None) -> Tuple[List[Dict], Dict]:
        assert len(model_responses) == len(query_ids) if query_ids is not None else True
        assert len(model_responses) == len(queries) if queries is not None else True
        
        # If query_ids and queries not provided, use all from loaded files
        if query_ids is None or queries is None:
            query_ids = [q['query_id'] for q in self.queries_data]
            queries = [q['query'] for q in self.queries_data]
        
        self.logger.info(f"Processing {len(queries)} queries")

        # Map query_ids to their data - FIXED: Use query_id_map instead of index
        query_data = [self.query_id_map[qid] for qid in query_ids]
        
        # Generate topics using LLM for provided queries (no need to extract from self.queries_data)
        self.logger.info("Generating topics for queries...")
        topics_prompts = [
            f'given this query "{query}" generate all the possible subtopics or related queries from most important to least, up to 10, one in each line with this jsonl format {{"topic": ...}}, nothing else in your response'
            for query in queries
        ]
        generated_topics_raw = self.llm_evaluator.generate(topics_prompts)
        
        # Process generated topics for each query
        self.logger.info("Parsing generated topics...")
        all_generated_topics = []
        for query_idx, topics_raw in enumerate(generated_topics_raw):
            generated_topics = []
            for line in topics_raw.split('\n'):
                if line.strip().startswith('{"topic":'):
                    try:
                        topic = json.loads(line.strip())["topic"]
                        generated_topics.append(topic)
                    except json.JSONDecodeError:
                        continue
            all_generated_topics.append(generated_topics)
            self.logger.info(f"Query {query_idx + 1}: Generated {len(generated_topics)} topics")

        # Get atomic facts for all responses
        self.logger.info("Generating atomic facts from responses...")
        all_facts = self.llm_evaluator.generate_facts(model_responses)
        
        # Process entailment for each query's facts in batches
        all_entailed_facts = []
        for query_idx, (query, response_facts) in enumerate(zip(query_data, all_facts)):
            self.logger.info(f"Processing entailment for query {query_idx + 1}")
            entailed_facts = []
            
            # Batch process facts for entailment
            for i in range(0, len(response_facts), self.nli_batch_size):
                batch_facts = response_facts[i:i + self.nli_batch_size]
                batch_results = []
                
                for fact in batch_facts:
                    top_docs = self._retrieve_documents(fact)
                    premises = [doc[1] for doc in top_docs]  # Get snippets
                    hypotheses = [fact] * len(premises)  # Repeat fact for each premise
                    
                    # Check entailment for current fact against all its premises
                    entailment_results = self._check_entailment_batch(premises, hypotheses)
                    
                    if any(entailment_results):
                        # Store the first document that entails this fact
                        for is_entailed, doc in zip(entailment_results, top_docs):
                            if is_entailed:
                                batch_results.append((doc[0], len(entailed_facts), fact))
                                break
                
                entailed_facts.extend(batch_results)
            
            all_entailed_facts.append(entailed_facts)

        # Collect all coverage prompts
        coverage_prompts = []
        for query_idx, (query, response_facts) in enumerate(zip(query_data, all_entailed_facts)):
            # Format entailed facts with numbers for this query
            entailed_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate([f[2] for f in response_facts])])
            
            coverage_prompt = (
                f'given this query "{query}", the following list of subtopics:\n\n' +
                "\n".join([f"{j+1} : {topic}" for j, topic in enumerate(all_generated_topics[query_idx])]) + "\n\n" +
                f'return the subtopics that are covered in the given text below with a list of facts, '
                f'mention each subtopic only once with a list of fact numbers for each subtopic, '
                f'the fact numbers should reference the most relevant facts that support the subtopic, '
                f'they should be explicitly mentioned in the given text, if they are not explicitly mentioned '
                f'don\'t include them in your response, if some subtopic is not covered without any evidence '
                f'don\'t include it in your response, use this jsonl format '
                f'{{"topic_id": ..., "evidence": [fact_number, ...]}}, one json object per line, '
                f'here is the text with enumerated facts:\n\n{entailed_text}'
            )
            coverage_prompts.append(coverage_prompt)

        # Process coverage prompts in batches
        covered_topics_responses = []
        for i in range(0, len(coverage_prompts), self.llm_batch_size):
            batch_prompts = coverage_prompts[i:i + self.llm_batch_size]
            batch_responses = self.llm_evaluator.generate(batch_prompts)
            covered_topics_responses.extend(batch_responses)

        results = []
        # Process each query-response pair
        for query_idx, (query, response_facts, covered_topics_raw) in enumerate(zip(query_data, all_entailed_facts, covered_topics_responses)):
            total_topics = len(all_generated_topics[query_idx])
            covered_data = []
            seen_topic_ids = set()
            
            # Get the correct atomic facts for this query
            query_atomic_facts = all_facts[query_idx]
            total_facts = len(query_atomic_facts)
            
            # Parse coverage response
            for line in covered_topics_raw.split('\n'):
                if line.strip().startswith('{"topic_id":'):
                    try:
                        data = json.loads(line.strip())
                        # Add validation for topic_id
                        if data.get("topic_id") is None:
                            continue
                            
                        # Try to convert topic_id to int, skip if invalid
                        try:
                            topic_id = int(data["topic_id"]) - 1
                        except (ValueError, TypeError):
                            continue
                        
                        # Skip if evidence list is empty or missing
                        if not data.get("evidence"):
                            continue
                        
                        if (0 <= topic_id < total_topics) and (topic_id not in seen_topic_ids):
                            valid_evidence = []
                            for fact_num in data["evidence"]:
                                try:
                                    fact_idx = int(fact_num) - 1
                                    if 0 <= fact_idx < total_facts:
                                        valid_evidence.append(fact_idx)
                                except (ValueError, TypeError):
                                    continue
                            
                            if valid_evidence:
                                seen_topic_ids.add(topic_id)
                                covered_data.append({
                                    "topic_id": topic_id + 1,
                                    "evidence": valid_evidence
                                })
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue

            coverage = len(covered_data) / total_topics if total_topics > 0 else 0
            factuality = len(response_facts) / total_facts if total_facts > 0 else 0
            f1 = 2 * (factuality * coverage) / (factuality + coverage) if (factuality + coverage) > 0 else 0

            self.logger.info(f"Query {query_idx + 1} metrics - Coverage: {coverage:.2f}, Factuality: {factuality:.2f}, F1: {f1:.2f}")

            results.append({
                "query_id": query['query_id'],
                "query": query['query'],
                "generated_topics": all_generated_topics[query_idx],
                "atomic_facts": query_atomic_facts,
                "entailed_facts": [f[2] for f in response_facts],
                "covered_topics": covered_data,
                "metrics": {
                    "coverage": coverage,
                    "factuality": factuality,
                    "f1": f1
                }
            })
        
        # Calculate aggregate metrics
        avg_coverage = sum(r["metrics"]["coverage"] for r in results) / len(results)
        avg_factuality = sum(r["metrics"]["factuality"] for r in results) / len(results)
        avg_f1 = sum(r["metrics"]["f1"] for r in results) / len(results)
        
        aggregate_metrics = {
            "avg_coverage": avg_coverage,
            "avg_factuality": avg_factuality,
            "avg_f1": avg_f1
        }
        
        return results, aggregate_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate coverage score')
    parser.add_argument('--corpus-path', default=None, help='Path to corpus file')
    parser.add_argument('--queries-path', default='queries.jsonl', help='Path to queries file')
    parser.add_argument('--qrels-path', default='qrel_cleaned.jsonl', help='Optional path to qrels file')
    parser.add_argument('--use-web-search', action='store_true', help='Use web search instead of corpus')
    parser.add_argument('--results-file', default='results_70b.txt', help='Path to results file')
    args = parser.parse_args()

    # Open results file
    with open(args.results_file, "w+") as results_file:
        results_file.write("Initializing components...\n")
        scorer = ICAT(
            corpus_path=args.corpus_path,
            queries_path=args.queries_path,
            qrels_path=args.qrels_path,
            use_web_search=args.use_web_search
        )

        # Load human judgements from model files
        results_file.write("Loading human judgements...\n")
        human_judgements = []
        model_files = [
            "annotations/gpt-4.jsonl",
            "annotations/llama3-70b.jsonl",
            "annotations/mixtral.jsonl",
            "annotations/openchat.jsonl"
        ]

        # Track judgements by model
        model_judgements_dict = {}
        
        for file_path in model_files:
            model_name = file_path.split('/')[-1].replace('.jsonl', '')
            try:
                with open(file_path, "r") as f:
                    model_judgements = [json.loads(line) for line in f][:5]
                    # Add model name to each judgment
                    for judgment in model_judgements:
                        judgment["model_name"] = model_name
                    model_judgements_dict[model_name] = model_judgements
                    human_judgements.extend(model_judgements)
            except Exception as e:
                results_file.write(f"Error loading {file_path}: {str(e)}\n")
                continue

        # Extract queries and responses
        queries = [j["query_description"] for j in human_judgements]
        query_ids = [j["query_id"] for j in human_judgements]
        responses = [j["model_response"] for j in human_judgements]

        # Calculate human scores
        human_scores = [len(j["covered"]) / len(j["subtopics"]) for j in human_judgements]

        # Run evaluations and calculate correlations
        results_file.write("\nRunning evaluations and calculating correlations...\n")
        results_file.write(f"Number of human judgements: {len(human_scores)}\n")

        # Semi-automatic evaluation
        results_s, metrics_s = scorer.icat_score_s(query_ids=query_ids, queries=queries, model_responses=responses)
        s_scores = [r["metrics"]["coverage"] for r in results_s]
        results_file.write(f"Number of semi-automatic scores: {len(s_scores)}\n")

        # Verify lengths match before calculating correlations
        if len(human_scores) != len(s_scores):
            results_file.write(f"WARNING: Score length mismatch - human: {len(human_scores)}, semi-auto: {len(s_scores)}\n")
            # Trim to shorter length to avoid errors
            min_len = min(len(human_scores), len(s_scores))
            human_scores = human_scores[:min_len]
            s_scores = s_scores[:min_len]

        s_pearson, _ = pearsonr(human_scores, s_scores)
        s_spearman, _ = spearmanr(human_scores, s_scores)
        s_kendall, _ = kendalltau(human_scores, s_scores)

        # Automatic evaluation
        results_a, metrics_a = scorer.icat_score_a(query_ids=query_ids, queries=queries, model_responses=responses)
        a_scores = [r["metrics"]["coverage"] for r in results_a]
        results_file.write(f"Number of automatic scores: {len(a_scores)}\n")

        # Verify lengths match before calculating correlations
        if len(human_scores) != len(a_scores):
            results_file.write(f"WARNING: Score length mismatch - human: {len(human_scores)}, auto: {len(a_scores)}\n")
            # Trim to shorter length to avoid errors
            min_len = min(len(human_scores), len(a_scores))
            human_scores = human_scores[:min_len]
            a_scores = a_scores[:min_len]

        a_pearson, _ = pearsonr(human_scores, a_scores)
        a_spearman, _ = spearmanr(human_scores, a_scores)
        a_kendall, _ = kendalltau(human_scores, a_scores)

        # Write results
        results_file.write("\nSemi-Automatic Evaluation Results:\n")
        results_file.write(f"Average Coverage: {metrics_s['avg_coverage']:.3f}\n")
        results_file.write(f"Average Factuality: {metrics_s['avg_factuality']:.3f}\n")
        results_file.write(f"Average F1: {metrics_s['avg_f1']:.3f}\n")
        results_file.write(f"Pearson correlation: {s_pearson:.3f}\n")
        results_file.write(f"Spearman correlation: {s_spearman:.3f}\n")
        results_file.write(f"Kendall correlation: {s_kendall:.3f}\n")

        results_file.write("\nAutomatic Evaluation Results:\n")
        results_file.write(f"Average Coverage: {metrics_a['avg_coverage']:.3f}\n")
        results_file.write(f"Average Factuality: {metrics_a['avg_factuality']:.3f}\n")
        results_file.write(f"Average F1: {metrics_a['avg_f1']:.3f}\n")
        results_file.write(f"Pearson correlation: {a_pearson:.3f}\n")
        results_file.write(f"Spearman correlation: {a_spearman:.3f}\n")
        results_file.write(f"Kendall correlation: {a_kendall:.3f}\n")

        # Calculate per-model metrics
        results_file.write("\nMetrics per Model:\n")
        for model_name, judgements in model_judgements_dict.items():
            # Get indices for this model from the original human_judgements list
            model_indices = [i for i, j in enumerate(human_judgements) if j["model_name"] == model_name]
            
            # Filter indices that are within bounds of our trimmed scores lists
            valid_indices = [i for i in model_indices if i < len(human_scores)]
            
            if not valid_indices:
                results_file.write(f"\n{model_name}: Insufficient data for correlation analysis\n")
                continue
                
            model_human_scores = [human_scores[i] for i in valid_indices]
            model_s_scores = [s_scores[i] for i in valid_indices]
            model_a_scores = [a_scores[i] for i in valid_indices]

            # Only calculate correlations if we have valid data
            if len(set(model_s_scores)) <= 1 or len(set(model_human_scores)) <= 1:
                results_file.write(f"\n{model_name}: Constant scores detected - correlations undefined\n")
                continue

            # Calculate correlations for this model
            try:
                model_s_pearson, _ = pearsonr(model_human_scores, model_s_scores)
                model_s_spearman, _ = spearmanr(model_human_scores, model_s_scores)
                model_s_kendall, _ = kendalltau(model_human_scores, model_s_scores)

                model_a_pearson, _ = pearsonr(model_human_scores, model_a_scores)
                model_a_spearman, _ = spearmanr(model_human_scores, model_a_scores)
                model_a_kendall, _ = kendalltau(model_human_scores, model_a_scores)

                results_file.write(f"\n{model_name}:\n")
                results_file.write("  Semi-Automatic:\n")
                results_file.write(f"    Pearson correlation: {model_s_pearson:.3f}\n")
                results_file.write(f"    Spearman correlation: {model_s_spearman:.3f}\n")
                results_file.write(f"    Kendall correlation: {model_s_kendall:.3f}\n")
                results_file.write("  Automatic:\n")
                results_file.write(f"    Pearson correlation: {model_a_pearson:.3f}\n")
                results_file.write(f"    Spearman correlation: {model_a_spearman:.3f}\n")
                results_file.write(f"    Kendall correlation: {model_a_kendall:.3f}\n")
            except Exception as e:
                results_file.write(f"\n{model_name}: Error calculating correlations: {str(e)}\n")

        # Save all results to JSON files
        with open("semi_auto_output.jsonl", "w+") as f:
            json.dump(results_s, f)
        with open("auto_output.jsonl", "w+") as f:
            json.dump(results_a, f)
