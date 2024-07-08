import random
import torch
import numpy as np
import pandas as pd
from sentence_transformers import util, SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
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

def get_model_num_params(model: torch.nn.Module):
    return sum([param.numel() for param in model.parameters()])

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
print("Search part done.")
print("\n")

gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
gpu_memory_gb = round(gpu_memory_bytes / (2**30))
print(f"Available GPU memory: {gpu_memory_gb} GB")

if gpu_memory_gb < 5.1:
    print(f"Available GPU memory is {gpu_memory_gb} GB. It may not be possible to run a Gemma LLM locally without quantization.")
elif gpu_memory_gb < 8.1:
    print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
    use_quantization_config = True
    model_id = "google/gemma-2b-it"
elif gpu_memory_gb < 19.0:
    print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in a 4-bit precision.")
    use_quantization_config = False
    model_id = "google/gemma-7b-it"
elif gpu_memory_gb > 19.0:
    print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 7B in 4-bit or float16 precision.")
    use_quantization_config = False
    model_id= "google/gemma-7b-it"

print(f"use quantization set to: {use_quantization_config}")
print(f"model_id set to: {model_id}")

if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"
print(f"[INFO] Using attention implementation: {attn_implementation}")

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_id)
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = model_id,
                                                torch_dtype = torch.float16,
                                                quantization_config = quantization_config if use_quantization_config else None,
                                                low_cpu_mem_usage = False,
                                                attn_implementation = attn_implementation)
if not use_quantization_config:
    llm_model.to("cuda")                             
print("llm_model: ", llm_model)
print("Total model parameters: ", get_model_num_params(llm_model))


