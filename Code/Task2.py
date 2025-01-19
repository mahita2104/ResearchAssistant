import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import requests
from semanticscholar import SemanticScholar
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import os
import re
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import csv


class ConferenceVectorDB:
    def __init__(self, persist_dir="./conference_db"):
        # Initialize ChromaDB
        self.client = chromadb.Client()
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Use SPECTER model specifically designed for academic papers
        self.embedder = SentenceTransformer('allenai/specter')

        # Initialize Semantic Scholar client
        self.sch = SemanticScholar()
        
        try:
            self.collection = self.client.get_collection(
                name="conference_papers",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="allenai/specter"
                )
            )
            print(f"Loaded existing collection with {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection(
                name="conference_papers",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="allenai/specter"
                )
            )
            print("Created new collection")

        # Conference metadata
        self.conferences = {
            'EMNLP': {
                'field': 'Natural Language Processing',
                'themes': ["Computational Social Science and Cultural Analytics",
                          "Dialogue and Interactive Systems",
                          "Discourse and Pragmatics",
                          "Low-resource Methods for NLP",
                          "Ethics, Bias, and Fairness",
                          "Generation",
                          "Information Extraction",
                          "Information Retrieval and Text Mining",
                          "Interpretability and Analysis of Models for NLP",
                          "Linguistic theories, Cognitive Modeling and Psycholinguistics",
                          "Machine Learning for NLP",
                          "Machine Translation",
                          "Multilinguality and Language Diversity",
                          "Multimodality and Language Grounding to Vision, Robotics and Beyond"],
                'focus': ['empirical methods', 'statistical approaches', 'deep learning for NLP']
            },
            'CVPR': {
                'field': 'Computer Vision',
                'themes': ["3D from multi-view and sensors", "3D from single images", "Adversarial attack and defense",
                           "Autonomous driving",  "Biometrics", "Computational imaging", "Computer vision for social good", "Computer vision theory",
                           "Datasets and evaluation",  "Deep learning architectures and techniques", "Document analysis and understanding", "Efficient and scalable vision",
                           "Embodied vision: Active agents, simulation", "Explainable computer vision", "Humans: Face, body, pose, gesture, movement", "Image and video synthesis and generation",
                           "Low-level vision", "Machine learning (other than deep learning)", "Medical and biological vision, cell microscopy", "Multimodal learning", "Optimization methods (other than deep learning)",
                           "Photogrammetry and remote sensing",  "Physics-based vision and shape-from-X", "Recognition: Categorization, detection, retrieval", "Representation learning",  "Robotics",
                           "Scene analysis and understanding", "Segmentation, grouping and shape analysis", "Self-& semi-& meta-& unsupervised learning", "Transfer/ low-shot/ continual/ long-tail learning", "Transparency, fairness, accountability, privacy and ethics in vision",
                           "Video: Action and event understanding", "Video: Low-level analysis, motion, and tracking", "Vision + graphics", "Vision, language, and reasoning", "Vision applications and systems"],
                'focus': ['deep learning', 'computer vision algorithms', 'visual understanding']
            },
            'KDD': {
                'field': 'Data Mining',
                'themes': ["Data Science: Methods for analyzing social networks, time series, sequences, streams, text, web, graphs, rules, patterns, logs, IoT data, spatio-temporal data, biological data, scientific and business data; recommender systems, computational advertising, multimedia, finance, bioinformatics.",
                           "Big Data: Large-scale systems for data analysis, machine learning, optimization, sampling, summarization; parallel and distributed data science (cloud, map-reduce, federated learning); novel algorithmic and statistical techniques for big data; algorithmically efficient data transformation and integration.",
                           "Foundations: Models and algorithms, asymptotic analysis; model selection, dimensionality reduction, relational/structured learning, matrix and tensor methods, probabilistic and statistical methods; deep learning, transfer learning, representation learning, meta learning, reinforcement learning; classification, clustering, regression, semi-supervised learning, self-supervised learning, few shot learning and unsupervised learning; personalization, security and privacy, visualization; fairness, interpretability, ethics and robustness."
                           ],
                'focus': ['large-scale data', 'predictive modeling', 'real-world applications']
            },
            'NeurIPS': {
                'field': 'Machine Learning and Computational Neuroscience',
                'themes': ["Applications (e.g., vision, language, speech and audio, Creative AI)",
                           "Deep learning (e.g., architectures, generative models, optimization for deep networks, foundation models, LLMs)",
                           "Evaluation (e.g., methodology, meta studies, replicability and validity, human-in-the-loop)",
                           "General machine learning (supervised, unsupervised, online, active, etc.)",
                           "Infrastructure (e.g., libraries, improved implementation and scalability, distributed solutions)",
                           "Machine learning for sciences (e.g., climate, health, life sciences, physics, social sciences)",
                           "Neuroscience and cognitive science (e.g., neural coding, brain-computer interfaces)",
                           "Optimization (e.g., convex and non-convex, stochastic, robust)",
                           "Probabilistic methods (e.g., variational inference, causal inference, Gaussian processes)",
                           "Reinforcement learning (e.g., decision and control, planning, hierarchical RL, robotics)",
                           "Social and economic aspects of machine learning (e.g., fairness, interpretability, human-AI interaction, privacy, safety, strategic behavior)",
                           "Theory (e.g., control theory, learning theory, algorithmic game theory)"],
                'focus': ['deep learning', 'probabilistic methods', 'theoretical advances']
            },
            'TMLR': {
                'field': 'Machine Learning Research',
                'themes': ['new algorithms with sound empirical validation, optionally with justification of theoretical, psychological, or biological nature;', 'experimental and/or theoretical studies yielding new insight into the design and behavior of learning in intelligent systems;', 'methodology', 'formalization of new learning tasks (e.g., in the context of new applications) and of methods for assessing performance on those tasks',
                           'computational models of natural learning systems at the behavioral or neural level', 'new approaches for analysis, visualization, and understanding of artificial or biological learning systems', 'surveys that draw new connections, highlight trends, and suggest new problems in an area'],
                'focus': ['theoretical foundations', 'novel approaches', 'empirical studies']
            }
        }
    def fetch_conference_papers(self, conference: str, year: int = 2023, limit: int = 100) -> List[Dict]:
        """Fetch papers from a specific conference"""
        papers = []
        query = f"venue:{conference} year:{year}"

        try:
            results = self.sch.search_paper(query, limit=limit)
            for paper in results:
                if paper.abstract:  # Only include papers with abstracts
                    papers.append({
                        'id': f"{conference}_{year}_{hash(paper.title)}",
                        'title': paper.title,
                        'abstract': paper.abstract,
                        'conference': conference,
                        'year': year
                    })
        except Exception as e:
            print(f"Error fetching papers for {conference}: {str(e)}")

        return papers

    def build_database(self, years: List[int] = [2022, 2023]):
        """Build the vector database for all conferences"""
        for conference in self.conferences.keys():
            print(f"Processing {conference}...")
            for year in years:
                papers = self.fetch_conference_papers(conference, year)

                # Add papers to vector database
                for paper in papers:
                    text = f"{paper['title']} {paper['abstract']}"

                    self.collection.add(
                        documents=[text],
                        metadatas=[{
                            'conference': paper['conference'],
                            'year': paper['year'],
                            'title': paper['title']
                        }],
                        ids=[paper['id']]
                    )

                print(f"Added {len(papers)} papers from {conference} {year}")

    def compare_paper(self, title: str, abstract: str, top_k: int = 5) -> List[Dict]:
        """Compare input paper with conference papers"""
        query_text = f"{title}\n{abstract}"
        
        # Query the vector database
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )

        # Process results
        recommendations = self._process_results(results, title, abstract)
        return recommendations

    def _process_results(self, results, title: str, abstract: str) -> List[Dict]:
        """Process results with enhanced analysis for more accurate conference recommendations"""
        conference_scores = {}
        query_text = f"{title}\n{abstract}"

        # Create embeddings for different components separately
        title_embedding = self.embedder.encode([title])[0]
        abstract_embedding = self.embedder.encode([abstract])[0]

        # Define section weights for scoring
        weights = {
            'theme_match': 0.3,
            'methodology_match': 0.25,
            'content_similarity': 0.25,
            'technical_depth': 0.20
        }

        def calculate_theme_score(paper_text: str, themes: List[str]) -> float:
            """Calculate how well the paper matches conference themes"""
            theme_scores = []
            for theme in themes:
                theme_words = set(theme.lower().split())
                # Count matching significant words (excluding common words)
                matches = sum(1 for word in theme_words 
                            if len(word) > 3 and word in paper_text.lower())
                theme_scores.append(matches / len(theme_words))
            return max(theme_scores) if theme_scores else 0

        def analyze_methodology(text: str) -> Dict[str, float]:
            """Analyze the research methodology used in the paper"""
            methodology_patterns = {
                'empirical': [
                    r'\b(experiment|study|survey|analysis|dataset|evaluation|results)\b',
                    r'\b(measure|metric|statistical|correlation|regression)\b'
                ],
                'theoretical': [
                    r'\b(theorem|proof|framework|model|theory|formal|analytical)\b',
                    r'\b(mathematical|proposition|lemma|algorithm|complexity)\b'
                ],
                'applied': [
                    r'\b(implementation|system|tool|application|platform|software)\b',
                    r'\b(deployment|user|interface|performance|optimization)\b'
                ]
            }
            
            scores = {}
            for method, patterns in methodology_patterns.items():
                score = sum(len(re.findall(pattern, text.lower())) 
                           for pattern in patterns)
                scores[method] = score
            
            # Normalize scores
            total = sum(scores.values()) + 1e-10
            return {k: v/total for k, v in scores.items()}

        def calculate_technical_depth(text: str) -> float:
            """Analyze the technical depth of the paper"""
            technical_indicators = {
                'high': [
                    r'\b(algorithm|complexity|optimization|theoretical|proof)\b',
                    r'\b(mathematical|analytical|formal|methodology|framework)\b',
                    r'\b(architecture|implementation|evaluation|analysis)\b'
                ],
                'medium': [
                    r'\b(results|performance|comparison|experiment|study)\b',
                    r'\b(approach|technique|method|system|model)\b'
                ],
                'basic': [
                    r'\b(use|using|based|simple|apply)\b',
                    r'\b(show|present|describe|introduce)\b'
                ]
            }
            
            scores = {
                'high': 3,
                'medium': 2,
                'basic': 1
            }
            
            total_score = 0
            total_matches = 0
            
            for level, patterns in technical_indicators.items():
                matches = sum(len(re.findall(pattern, text.lower())) 
                             for pattern in patterns)
                total_score += matches * scores[level]
                total_matches += matches
                
            return total_score / (total_matches + 1e-10)

        # Process each conference's papers
        for idx, metadata in enumerate(results['metadatas'][0]):
            conf = metadata['conference']
            paper_text = f"{metadata['title']}\n{metadata.get('abstract', '')}"
            
            if conf not in conference_scores:
                conference_scores[conf] = {
                    'total_score': 0,
                    'count': 0,
                    'papers': [],
                    'methodology_distribution': {'empirical': 0, 'theoretical': 0, 'applied': 0}
                }

            # Calculate various similarity scores
            paper_title_embedding = self.embedder.encode([metadata['title']])[0]
            paper_abstract = metadata.get('abstract', '')
            paper_abstract_embedding = self.embedder.encode([paper_abstract])[0] if paper_abstract else None

            # Content similarity score
            title_similarity = cosine_similarity([title_embedding], [paper_title_embedding])[0][0]
            abstract_similarity = (
                cosine_similarity([abstract_embedding], [paper_abstract_embedding])[0][0]
                if paper_abstract_embedding is not None else 0
            )
            content_score = 0.4 * title_similarity + 0.6 * abstract_similarity

            # Theme matching score
            theme_score = calculate_theme_score(query_text, self.conferences[conf]['themes'])

            # Methodology analysis
            paper_methodology = analyze_methodology(query_text)
            conf_methodology = analyze_methodology(' '.join(self.conferences[conf]['focus']))
            methodology_score = sum(min(paper_methodology[k], conf_methodology[k]) 
                                  for k in paper_methodology.keys())

            # Technical depth score
            technical_score = calculate_technical_depth(query_text)

            # Calculate weighted final score
            final_score = (
                weights['theme_match'] * theme_score +
                weights['methodology_match'] * methodology_score +
                weights['content_similarity'] * content_score +
                weights['technical_depth'] * technical_score
            )

            # Update conference scores
            conference_scores[conf]['total_score'] += final_score
            conference_scores[conf]['count'] += 1
            conference_scores[conf]['methodology_distribution'] = {
                k: v + paper_methodology[k] 
                for k, v in conference_scores[conf]['methodology_distribution'].items()
            }
            
            conference_scores[conf]['papers'].append({
                'title': metadata['title'],
                'year': metadata['year'],
                'similarity': final_score,
                'abstract': metadata.get('abstract', ''),
                'theme_match': theme_score,
                'methodology_match': methodology_score,
                'content_similarity': content_score,
                'technical_depth': technical_score
            })

        # Generate final recommendations
        recommendations = []
        for conf, data in conference_scores.items():
            avg_score = data['total_score'] / data['count']
            
            # Normalize methodology distribution
            total_methods = sum(data['methodology_distribution'].values()) + 1e-10
            normalized_distribution = {
                k: v/total_methods 
                for k, v in data['methodology_distribution'].items()
            }

            rationale = self._generate_rationale(
                conf,
                query_text,
                data['papers'],
                normalized_distribution
            )

            recommendations.append({
                'conference': conf,
                'score': avg_score,
                'field': self.conferences[conf]['field'],
                'methodology_distribution': normalized_distribution,
                'similar_papers': sorted(data['papers'], key=lambda x: x['similarity'], reverse=True),
                'rationale': rationale,
                'score_breakdown': {
                    'theme_match': sum(p['theme_match'] for p in data['papers']) / len(data['papers']),
                    'methodology_match': sum(p['methodology_match'] for p in data['papers']) / len(data['papers']),
                    'content_similarity': sum(p['content_similarity'] for p in data['papers']) / len(data['papers']),
                    'technical_depth': sum(p['technical_depth'] for p in data['papers']) / len(data['papers'])
                }
            })

        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return [recommendations[:2]]

    def _generate_rationale(self, conference: str, paper_text: str, similar_papers: List[Dict], methodology_distribution: Dict) -> str:
        """Generate enhanced rationale with detailed analysis"""
        conf_info = self.conferences[conference]
        split_query = paper_text.split('\n')
        title = split_query[0]
        abstract = split_query[1] if len(split_query) > 1 else ''

        # Find primary methodology
        primary_method = max(methodology_distribution.items(), key=lambda x: x[1])[0]
        
        # Analyze theme alignment more thoroughly
        theme_matches = []
        theme_relevance_scores = {}
        for theme in conf_info['themes']:
            theme_words = set(theme.lower().split())
            relevance = sum(word in paper_text.lower() for word in theme_words) / len(theme_words)
            if relevance > 0.3:  # Threshold for considering a theme relevant
                theme_matches.append(theme)
                theme_relevance_scores[theme] = relevance
        
        # Sort themes by relevance
        top_themes = sorted(theme_relevance_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        # Analyze similar papers
        recent_papers = [p for p in similar_papers if p.get('year', 0) >= 2022]
        avg_similarity = sum(p['similarity'] for p in recent_papers) / len(recent_papers) if recent_papers else 0
        
        # Generate comprehensive rationale
        rationale_components = [
            f"The paper '{title}' shows strong alignment with {conference}'s focus on {conf_info['field']}.",
            
            f"Using primarily {primary_method} methodologies ({methodology_distribution[primary_method]:.1%} of content), "
            f"this research aligns well with {conference}'s emphasis on {', '.join(conf_info['focus'][:2])}.",
            
            f"The work particularly resonates with {conference} themes of {', '.join(t[0] for t in top_themes)}" 
            if top_themes else None,
            
            f"Analysis of {len(recent_papers)} recent similar papers in {conference} (average similarity: {avg_similarity:.2f}) "
            f"suggests strong thematic alignment with current research trends."
            if recent_papers else None,
        
            f"The technical approach and depth of analysis align closely with {conference}'s standards "
            f"and {conf_info['focus'][-1]} focus."
           ]
    
        rationale = ' '.join(filter(None, rationale_components))
        return rationale

def main():
    # Initialize the system
    db = ConferenceVectorDB()
    # Build the database (only need to do this once)
   # db.build_database()

    # Directory containing conference subdirectories with PDFs
    directory_path = "/home/mahita/Desktop/conference_paper/Publishable"

    # Process the directory and extract papers
    def extract_title_and_abstract(pdf_path):
        """
        Extracts the title and abstract from a PDF.
        """
        title_abstract_dict = {"title": None, "abstract": None}

        try:
            reader = PdfReader(pdf_path)
            all_text = ""

            # Extract text from all pages
            for page in reader.pages:
                all_text += page.extract_text()

            # Normalize text
            all_text = re.sub(r'\s+', ' ', all_text).strip()

            # Extract the title
            title_match = re.search(
                r'^(.*?)(?:\n|Abstract|ABSTRACT)', all_text, re.IGNORECASE)
            if title_match:
                title_abstract_dict["title"] = title_match.group(1).strip()

            # Extract the abstract
            abstract_match = re.search(
                r'(?i)(Abstract|ABSTRACT)[:\n\s]*(.*?)(?:\n\s*\w+:|Introduction|INTRODUCTION)', all_text, re.DOTALL)
            if abstract_match:
                title_abstract_dict["abstract"] = abstract_match.group(
                    2).strip()
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

        return title_abstract_dict

    def process_directory(directory_path):
        """
        Iterates through all subdirectories and PDFs, and organizes extracted data in a nested dictionary.
        """
        conference_data = {}

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".pdf"):
                    # Get conference name (subdirectory name) and file path
                    conference_name = os.path.basename(root)
                    pdf_path = os.path.join(root, file)

                    # Extract title and abstract
                    extracted_data = extract_title_and_abstract(pdf_path)

                    # Add to dictionary
                    if conference_name not in conference_data:
                        conference_data[conference_name] = []
                    conference_data[conference_name].append({
                        "file_name": file,
                        "title": extracted_data["title"],
                        "abstract": extracted_data["abstract"],
                    })

        return conference_data

    # Extract conference data
    conference_data = process_directory(directory_path)

    # Prepare CSV file
    csv_file_path = "conference_recommendations.csv"
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(["Paper Title", "Recommended Conferences", "Rationales"])

        # Loop through all papers and print similarity scores
        print("\nProcessing all papers for similarity scores...\n")
        for conference, papers in conference_data.items():
            for paper in papers:
                title = paper["title"]
                abstract = paper["abstract"]
                if title and abstract:
                    recommendations = db.compare_paper(title, abstract)

                    # Concatenate recommendations
                    recommended_conferences = []
                    rationales = []
                    for rec_list in recommendations:
                        for rec in rec_list:
                            recommended_conferences.append(rec['conference'])
                            rationales.append(rec['rationale'])

                    # Join recommendations with commas
                    recommended_conferences_str = ", ".join(recommended_conferences[:2])
                    rationales_str = " | ".join(rationales[:2])

                    # Write results to CSV
                    csv_writer.writerow([title, recommended_conferences_str, rationales_str])
                    print(f"Stored recommendation for paper: {title}")

                else:
                    print(f"Skipping paper {paper['file_name']} due to missing title or abstract.")

    print(f"Results saved to {csv_file_path}")


if __name__ == "__main__":
    main()
