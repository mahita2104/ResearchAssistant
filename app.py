import logging
import pathway as pw
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, InstanceOf
import os
import json
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Set up logging configuration for debugging and tracking information
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Step 2: Load environment variables, including OpenAI API key for embedding generation
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure that OPENAI_API_KEY is in your .env file

# Step 3: Function to generate text embeddings using OpenAI's API
def generate_embedding(text):
    """
    Generates an embedding vector for the input text using OpenAI's embedding model.

    Args:
        text (str): The text for which the embedding is to be generated.

    Returns:
        np.array: The embedding vector as a numpy array.
    """
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # OpenAI's powerful text embedding model
        input=text
    )
    return np.array(response['data'][0]['embedding'])  # Extract the embedding vector from the response

# Step 4: Define the Application class to handle the configuration and initialization of the app
class App(BaseModel):
    """
    Application class that handles the configuration of the app and manages the Q&A server.
    """
    question_answerer: InstanceOf[SummaryQuestionAnswerer]
    host: str = "0.0.0.0"  # Default host
    port: int = 8000  # Default port
    with_cache: bool = True  # Enable or disable caching
    terminate_on_error: bool = False  # Option to terminate on error

    def run(self) -> None:
        """
        Starts the Q&A server using the configuration defined above.
        """
        server = QASummaryRestServer(self.host, self.port, self.question_answerer)
        server.run(
            with_cache=self.with_cache,
            terminate_on_error=self.terminate_on_error,
            cache_backend=pw.persistence.Backend.filesystem("Cache"),
        )

    model_config = ConfigDict(extra="forbid")  # Prevent adding any extra fields to the model

# Step 5: Custom JSON Parser to load and parse conference data
class ConferenceJsonParser:
    """
    Custom parser to read and extract conference data from a JSON file and return it in a structured format.
    """
    def __init__(self, metadata_keys, embedding_key, content_key):
        """
        Initializes the parser with keys for metadata, embedding, and content.

        Args:
            metadata_keys (dict): Keys to extract metadata such as conference name and title.
            embedding_key (str): Key to extract embeddings.
            content_key (str): Key to extract the content of each conference description.
        """
        self.metadata_keys = metadata_keys
        self.embedding_key = embedding_key
        self.content_key = content_key

    def parse(self, file_path):
        """
        Parses the JSON file and returns a list of structured documents with embeddings.

        Args:
            file_path (str): Path to the JSON file containing conference data.

        Returns:
            list: List of parsed conference documents with embeddings and metadata.
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        parsed_documents = []
        for entry in data:
            parsed_documents.append({
                "id": entry.get("id", ""),  # Document ID
                "content": entry.get(self.content_key, ""),  # Content of the conference description
                "embedding": entry.get(self.embedding_key, []),  # Embedding for the conference
                "metadata": {key: entry["metadata"].get(value, "") for key, value in self.metadata_keys.items()},
            })
        return parsed_documents

# Step 6: Function to calculate similarity between paper embeddings and conference embeddings
def calculate_similarity(query_embedding, conference_embeddings):
    """
    Compares the given query (paper) embedding with multiple conference embeddings and calculates similarity.

    Args:
        query_embedding (np.array): Embedding vector of the input paper.
        conference_embeddings (list): List of conferences with their embeddings.

    Returns:
        list: Sorted list of conferences with similarity scores.
    """
    similarities = []
    for conference in conference_embeddings:
        # Cosine similarity computation
        similarity_score = cosine_similarity([query_embedding], [conference['embedding']])[0][0]
        similarities.append({
            "conference_name": conference["conference_name"],  # Name of the conference
            "similarity_score": similarity_score  # Similarity score between paper and conference
        })

    # Sort the conferences by similarity score in descending order
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similarities

# Step 7: Generate Rationale Based on Similarity
def generate_rationale(paper_text, recommended_conference, similarity_score):
    """
    Generate a comprehensive rationale for recommending the conference to the paper.

    Args:
        paper_text (str): The text of the paper that is being compared.
        recommended_conference (str): The name of the recommended conference.
        similarity_score (float): The similarity score between the paper and the recommended conference.

    Returns:
        str: The rationale string explaining the recommendation.
    """
    rationale = (
        f"The paper titled '{paper_text[:30]}...' is highly recommended for {recommended_conference} "
        f"because of its strong alignment with the conference's themes and research focus. "
        f"The cosine similarity score between the paper and the conference is {similarity_score:.2f}, "
        "indicating a significant match. "
        "This paper's focus on deep learning and artificial intelligence aligns well with the conference's "
        "emphasis on cutting-edge computational methods and innovative research in these domains."
    )
    return rationale

if __name__ == "__main__":
    # Step 8: Load configuration for the app (from YAML file)
    with open("app.yaml") as f:
        config = pw.load_yaml(f)

    # Step 9: Initialize and parse conference data
    parser = ConferenceJsonParser(
        metadata_keys={"conference": "conference", "title": "title"},
        embedding_key="embedding",
        content_key="content"
    )

    # Parse the conference data from the specified JSON file
    json_file_path = os.getenv("JSON_FILE_PATH", "conference_data.json")  # You can set this path in .env
    parsed_data = parser.parse(json_file_path)

    # Log parsed conference data for debugging purposes
    logging.info("Parsed Data: %s", parsed_data)

    # Step 10: Generate embeddings for each conference using the conference content
    conference_embeddings = []
    for conference in parsed_data:
        embedding = generate_embedding(conference["content"])  # Generate embedding for each conference
        conference_embeddings.append({
            "conference_name": conference["metadata"]["conference"],
            "embedding": embedding  # Store the conference name and embedding
        })

    # Step 11: Input paper (query) to be compared against the conference embeddings
    paper_text = "This paper explores deep learning techniques for image recognition."  # Example query
    query_embedding = generate_embedding(paper_text)  # Generate embedding for the paper text

    # Step 12: Calculate similarity between the paper and the conferences
    similarity_results = calculate_similarity(query_embedding, conference_embeddings)

    # Log similarity results (for debugging)
    logging.info("Similarity Results: %s", similarity_results)

    # Step 13: Recommend the most similar conference based on the highest similarity score
    recommended_conference = similarity_results[0] if similarity_results else None
    if recommended_conference:
        logging.info(f"Recommended Conference: {recommended_conference['conference_name']} with similarity score: {recommended_conference['similarity_score']:.4f}")

        # Step 14: Generate rationale for the recommendation
        rationale = generate_rationale(paper_text, recommended_conference['conference_name'], recommended_conference['similarity_score'])
        logging.info(f"Rationale for recommendation: {rationale}")
    else:
        logging.info("No matching conferences found.")

    # Step 15: Initialize and run the app (this is optional if you need an API)
    app = App(**config)
    app.run()
