import fitz
from tqdm.auto import tqdm
import random
import pandas as pd
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer

pdf_path = "../source_pdf/human-nutrition-text.pdf"
num_sentence_chink_size = 10

def text_formatter(text: str) -> str:
    """Performs minor formatting on text"""
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text() # get plain text encoded as UTF-8
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number - 41,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,
                                "text": text})
    return pages_and_texts

def split_list(input_list : list, slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]

pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
# print(pages_and_texts[:2])
# print()
# print(random.sample(pages_and_texts, k=3))
# print()
# df = pd.DataFrame(pages_and_texts)
# print(df.head())
# print()
# print(df.describe().round(2))

nlp = English()
nlp.add_pipe("sentencizer")
doc = nlp("I am Jayabrata. I am a student interested in Artificial Intelligence.")
assert len(list(doc.sents)) == 2
print(list(doc.sents))

for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    item["page_sentence_count_spacy"] = len(item["sentences"])

# print(random.sample(pages_and_texts, k=1))

# df2 = pd.DataFrame(pages_and_texts)
# print(df2.describe().round(2))

for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=num_sentence_chink_size)
    item["num_chunks"] = len(item["sentence_chunks"])
print(random.sample(pages_and_texts,k=1))
# df3 = pd.DataFrame(pages_and_texts)
# print(df3.describe().round(2))
pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]
        # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
        chunk_dict["sentence_chunk"] = joined_sentence_chunk
        # Get stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters
        pages_and_chunks.append(chunk_dict)
print("Length of pages_and_chunks: ",len(pages_and_chunks))
# print(random.sample(pages_and_chunks, k=1))
# Get stats about our chunks
df = pd.DataFrame(pages_and_chunks)
# print(df.describe().round(2))

min_token_length = 30
for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
    print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')

# Remove chunks with less than 30 tokens in length
pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda")
# sentences = [
#     "The Sentences Transformers library provides an easy and open-source way to create embeddings.",
#     "Sentences can be embedded one by one or as a list of strings.",
#     "Embeddings are one of the most powerful concepts in machine learning!",
#     "Learn to use embeddings well and you'll be well on your way to being an AI engineer."
# ]
# embeddings = embedding_model.encode(sentences)
# embeddings_dict = dict(zip(sentences, embeddings))
# for sentence, embedding in embeddings_dict.items():
#     print("Sentence: ", sentence)
#     print("Embedding: ", embedding)
#     print("")

# single_sentence = "Yo! How cool are embeddings?"
# single_embedding = embedding_model.encode(single_sentence)
# print(f"Sentence: {single_sentence}")
# print(f"Embedding:\n{single_embedding}")
# print(f"Embedding size: {single_embedding.shape}")

for item in tqdm(pages_and_chunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])

# Turn text chunks into a signle list
text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]
text_chunk_embeddings = embedding_model.encode(text_chunks, batch_size=32, convert_to_tensor=True)


# Save embeddings to file
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

text_chunks_and_embeddings_df_load = pd.read_csv(embeddings_df_save_path)
print(text_chunks_and_embeddings_df_load.head())

