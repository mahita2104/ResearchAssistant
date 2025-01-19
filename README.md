# ResearchAssistant 

## Overview  

The exponential growth in research publications calls for innovative AI-driven solutions to streamline the evaluation and classification of academic papers. This project focuses on automating the processes of assessing research paper publishability and recommending suitable academic conferences for submission.  

The framework leverages advanced tools like Google Gemini API and Pathway's AI capabilities to deliver scalable, accurate, and efficient workflows, reducing the manual workload for researchers and reviewers.  

---

## Key Features  

1. **Research Paper Publishability Assessment**  
   - Evaluates papers for adherence to academic standards, logical coherence, and methodological rigor.  
   - Classifies papers as "Publishable" or "Non-Publishable" with high accuracy.  
   - Validates results using labeled reference papers and calculates robust metrics like F1 Score for benchmarking.  

2. **Conference Selection**  
   - Matches "Publishable" papers to leading conferences (e.g., CVPR, NeurIPS, EMNLP, KDD, TMLR) based on thematic and methodological alignment.  
   - Employs embedding-based similarity analysis using Pathway's Vector Store and generates rationale using the Pathway Q&A RAG App.  

3. **Automated Outputs**  
   - Consolidates results into a CSV file, detailing classification, conference recommendations, and justification for the decisions.  
   - Supports real-time updates and scalable processing for large volumes of papers.  

---

## Problem Statement  

### Task 1: Research Paper Publishability Assessment  
The challenge lies in evaluating research papers for logical coherence, methodological rigor, and adherence to academic norms, automating the classification into "Publishable" or "Non-Publishable."  

### Task 2: Conference Selection  
For "Publishable" papers, the task involves identifying the most suitable academic conference by aligning the paper’s content with the themes and focus areas of leading conferences.  

---

## Methodology  

### Task 1: Publishability Assessment  
1. **Feature Extraction**  
   - Logical structure (abstract, methodology, results, etc.).  
   - Clarity and coherence of arguments.  
   - Methodological rigor and relevance.  

2. **Preprocessing**  
   - Extracts structured content (headings, paragraphs).  
   - Cleanses textual data, removing irrelevant metadata or artifacts.  

3. **Modeling**  
   - Utilizes Google Gemini API for evaluating content based on academic quality.  
   - Generates a CSV file with "Publishable" (1) and "Non-Publishable" (0) labels.  

4. **Validation**  
   - Results are validated using labeled reference papers, with metrics like accuracy and F1 Score calculated.  

<p align="center">
  <img src="https://github.com/mahita2104/ResearchAssistant/blob/main/architecture.png" height="320" />
</p> 
---

### Task 2: Conference Selection  
1. **Embedding Generation**  
   - Generates vector embeddings for "Publishable" papers using Pathway Vector Store.  
   - Embeds reference papers from each conference for similarity comparison.  

2. **Similarity Analysis**  
   - Computes cosine similarity scores to determine the most aligned conference.  

3. **Justification Generation**  
   - Employs Pathway Q&A RAG App to generate concise rationale for each recommendation.  

4. **Output**  
   - Produces a CSV file with the paper ID, publishability status, recommended conference, and rationale.  

---

## Technologies Used  

- **Google Gemini API:** For publishability assessment.  
- **Pathway Tools:** Vector Store for embedding generation and Q&A RAG App for rationale generation.  
- **Programming Languages:** Python for data preprocessing and integration.  
- **Output Format:** CSV files for results consolidation.  

---

## Results  

- **Task 1:** Accurate classification of papers into "Publishable" or "Non-Publishable" categories, validated through performance metrics.
<p align="center">
  <img src="https://github.com/mahita2104/ResearchAssistant/blob/main/task1_results.jpg" height="320" />
</p> 
- **Task 2:** Precise recommendations for suitable conferences, with well-documented rationales.  
<p align="center">
  <img src="https://github.com/mahita2104/ResearchAssistant/blob/main/task2_results.png" height="320" />
</p> 
---

## Future Work  

- Extend conference database to include more venues.  
- Incorporate additional validation datasets to improve publishability classification.  
- Enhance scalability for broader academic use cases.  

---
