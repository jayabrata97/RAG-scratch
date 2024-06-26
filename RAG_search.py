import random
import torch
import numpy as np
import pandas as pd
from sentence_transformers import util, SentenceTransformer
from time import perf_counter as timer
import textwrap

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(model_name_or_path = "all-mpnet-base-v2", device = device)
# Import embedding df
text_chunks_and_embedding_df = pd.read_csv("/home/user/RAG_from_scratch/utils/text_chunks_and_embeddings_df.csv")
# Convert embedding column back to numpy array
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
# Convert texts and embedding df to list of dictionaries
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
# Convert embeddings to torch tensor and send to device
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
print("Embeddings shape: ", embeddings.shape)

# Define helper function to print wrapped text
def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def retrieve_relevant_resources(query: str, embeddings: torch.tensor, model: SentenceTransformer=embedding_model, n_resources_to_return: int=5, print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """
    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time: .5f} seconds.")
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    
    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict]=pages_and_chunks,
                                 n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """
    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings, n_resources_to_return=n_resources_to_return)
    print(f"Query: {query}\n")
    print("Results:")
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print_wrapped(pages_and_chunks[index]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        print("\n")

# query = "macronutrients functions"
# print(f"Query: {query}")
# query_embedding = embedding_model.encode(query, convert_to_tensor=True)
# start_time = timer()
# dot_scores = util.dot_score(a = query_embedding, b = embeddings)[0]
# end_time = timer()
# print(f"Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")
# top_results_dot_product = torch.topk(dot_scores, k=5)
# print(top_results_dot_product)

# for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
#     print(f"Score: {score:.4f}")
#     print("Text:")
#     print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
#     print(f"Page number: {pages_and_chunks[idx]['page_number']}")
#     print("\n")

query = "symptoms of pellagra"
scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings)
print("scores: ", scores)
print("indices: ", indices)

print_top_results_and_scores(query=query, embeddings=embeddings)

