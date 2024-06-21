import random
import torch
import numpy as np
import pandas as pd
from sentence_transformers import util, SentenceTransformer
from time import perf_counter as timer
import textwrap

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define helper function to print wrapped text
def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

# Import embedding df
text_chunks_and_embedding_df = pd.read_csv("/home/user/RAG_from_scratch/utils/text_chunks_and_embeddings_df.csv")
# Convert embedding column back to numpy array
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
# Convert texts and embedding df to list of dictionaries
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
# Convert embeddings to torch tensor and send to device
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
print("Embeddings shape: ", embeddings.shape)

embedding_model = SentenceTransformer(model_name_or_path = "all-mpnet-base-v2", device = device)

query = "macronutrients functions"
print(f"Query: {query}")
query_embedding = embedding_model.encode(query, convert_to_tensor=True)
start_time = timer()
dot_scores = util.dot_score(a = query_embedding, b = embeddings)[0]
end_time = timer()
print(f"Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")
top_results_dot_product = torch.topk(dot_scores, k=5)
print(top_results_dot_product)

for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
    print(f"Score: {score:.4f}")
    print("Text:")
    print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
    print(f"Page number: {pages_and_chunks[idx]['page_number']}")
    print("\n")

