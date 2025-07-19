# %% [markdown]
# # Lab | LangChain Med
# 
# ## Objectives
# 
# - continue on with lesson 2' example, use different datasets to test what we did in class. Some datasets are suggested in the notebook but feel free to scout other datasets on HuggingFace or Kaggle.
# - Find another model on Hugging Face and compare it.
# - Modify the prompt to fit your selected dataset.

# %%
import numpy as np 
import pandas as pd

# %% [markdown]
# ## Load the Dataset
# As you can see the notebook is ready to work with three different Datasets. Just uncomment the lines of the Dataset you want to use. 
# 
# I selected Datasets with News. Two of them have just a brief decription of the news, but the other contains the full text. 
# 
# As we are working in a free and limited space, I limited the number of news to use with the variable MAX_NEWS. Feel free to pull more if you have memory available. 
# 
# The name of the field containing the text of the new is stored in the variable *DOCUMENT* and the metadata in *TOPIC*

# %%
# news = pd.read_csv('/kaggle/input/topic-labeled-news-dataset/labelled_newscatcher_dataset.csv', sep=';')
# MAX_NEWS = 1000
# DOCUMENT="title"
# TOPIC="topic"

#news = pd.read_csv('/kaggle/input/bbc-news/bbc_news.csv')
#MAX_NEWS = 1000
#DOCUMENT="description"
#TOPIC="title"

#news = pd.read_csv('/kaggle/input/mit-ai-news-published-till-2023/articles.csv')
#MAX_NEWS = 100
#DOCUMENT="Article Body"
#TOPIC="Article Header"

news = pd.read_csv(r'C:\Users\happy\Documents\ironhack\Week18\lab-langchain-med\datasets\Receipes from around the world.csv', encoding='latin1')
MAX_NEWS = 100
DOCUMENT="recipe_name"
TOPIC="cuisine"
# news = "PICK A DATASET" #Ideally pick one from the commented ones above

# %%
news.shape

# %%
news

# %% [markdown]
# ChromaDB requires that the data has a unique identifier. We can make it with this statement, which will create a new column called **Id**.
# 

# %%
news["id"] = news.index
news.head()

# %%
#Because it is just a course we select a small portion of News.
subset_news = news.head(MAX_NEWS)

# %% [markdown]
# ## Import and configure the Vector Database
# I'm going to use ChromaDB, the most popular OpenSource embedding Database. 
# 
# First we need to import ChromaDB, and after that import the **Settings** class from **chromadb.config** module. This class allows us to change the setting for the ChromaDB system, and customize its behavior. 

# %%
!pip install chromadb

# %%
import chromadb
from chromadb.config import Settings

# %% [markdown]
# Now we need to create the seetings object calling the **Settings** function imported previously. We store the object in the variable **settings_chroma**.
# 
# Is necessary to inform two parameters 
# * chroma_db_impl. Here we specify the database implementation and the format how store the data. I choose ***duckdb***, because his high-performace. It operate primarly in memory. And is fully compatible with SQL. The store format ***parquet*** is good for tabular data. With good compression rates and performance. 
# 
# * persist_directory: It just contains the directory where the data will be stored. Is possible work without a directory and the data will be stored in memory without persistece, but Kaggle dosn't support that. 

# %%
chroma_client = chromadb.PersistentClient(path="/path/to/persist/directory")

# %% [markdown]
# ## Filling and Querying the ChromaDB Database
# The Data in ChromaDB is stored in collections. If the collection exist we need to delete it. 
# 
# In the next lines, we are creating the collection by calling the ***create_collection*** function in the ***chroma_client*** created above.

# %%
collection_name = "news_collection"
if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
        chroma_client.delete_collection(name=collection_name)

collection = chroma_client.create_collection(name=collection_name)
    

# %% [markdown]
# It's time to add the data to the collection. Using the function ***add*** we need to inform, at least ***documents***, ***metadatas*** and ***ids***. 
# * In the **document** we store the big text, it's a different column in each Dataset. 
# * In **metadatas**, we can informa a list of topics. 
# * In **id** we need to inform an unique identificator for each row. It MUST be unique! I'm creating the ID using the range of MAX_NEWS. 
# 

# %%

collection.add(
    documents=subset_news[DOCUMENT].tolist(),
    metadatas=[{TOPIC: topic} for topic in subset_news[TOPIC].tolist()],
    ids=[f"id{x}" for x in range(MAX_NEWS)],
)

# %%
results = collection.query(query_texts=["laptop"], n_results=10 )

print(results)

# %% [markdown]
# ## Vector MAP

# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# %%

getado = collection.get(ids="id141", 
                       include=["documents", "embeddings"])


# %%
word_vectors = getado["embeddings"]
word_list = getado["documents"]
word_vectors

# %% [markdown]
# Once we have our information inside the Database we can query It, and ask for data that matches our needs. The search is done inside the content of the document, and it dosn't look for the exact word, or phrase. The results will be based on the similarity between the search terms and the content of documents. 
# 
# The metadata is not used in the search, but they can be utilized for filtering or refining the results after the initial search. 
# 

# %% [markdown]
# ## Loading the model and creating the prompt
# TRANSFORMERS!!
# Time to use the library **transformers**, the most famous library from [hugging face](https://huggingface.co/) for working with language models. 
# 
# We are importing: 
# * **Autotokenizer**: It is a utility class for tokenizing text inputs that are compatible with various pre-trained language models.
# * **AutoModelForCasualLLM**: it provides an interface to pre-trained language models specifically designed for language generation tasks using causal language modeling (e.g., GPT models), or the model used in this notebook ***databricks/dolly-v2-3b***.
# * **pipeline**: provides a simple interface for performing various natural language processing (NLP) tasks, such as text generation (our case) or text classification. 
# 
# The model selected is [dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b), the smallest Dolly model. It have 3billion paramaters, more than enough for our sample, and works much better than GPT2. 
# 
# Please, feel free to test [different Models](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending), you need to search for NLP models trained for text-generation. My recomendation is choose "small" models, or we will run out of memory in kaggle.  
# 

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "databricks/dolly-v2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
lm_model = AutoModelForCausalLM.from_pretrained(model_id)



# %% [markdown]
# The next step is to initialize the pipeline using the objects created above. 
# 
# The model's response is limited to 256 tokens, for this project I'm not interested in a longer response, but it can easily be extended to whatever length you want.
# 
# Setting ***device_map*** to ***auto*** we are instructing the model to automaticaly select the most appropiate device: CPU or GPU for processing the text generation.  

# %%
pipe = pipeline(
    "text-generation",
    model=lm_model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    device_map="auto",
)

# %% [markdown]
# ## Creating the extended prompt
# To create the prompt we use the result from query the Vector Database  and the sentence introduced by the user. 
# 
# The prompt have two parts, the **relevant context** that is the information recovered from the database and the **user's question**. 
# 
# We only need to join the two parts together to create the prompt that we are going to send to the model. 
# 
# You can limit the lenght of the context passed to the model, because we can get some Memory problems with one of the datasets that contains a realy large text in the document part. 

# %%
question = "Can I buy a Toshiba laptop?"
context = " ".join([f"#{str(i)}" for i in results["documents"][0]])
#context = context[0:5120]
prompt_template = f"Relevant context: {context}\n\n The user's question: {question}"
prompt_template

# %% [markdown]
# Now all that remains is to send the prompt to the model and wait for its response!
# 

# %%
lm_response = pipe(prompt_template)
print(lm_response[0]["generated_text"])

# %%
# LangChain setup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

# Enable cache
# set_llm_cache(SQLiteCache(database_path=".langchain.db"))
set_llm_cache(SQLiteCache(database_path="cache.db"))
# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Prepare the recipe documents
docs = [
    Document(page_content=row['recipe_name'], metadata={"cuisine": row["cuisine"]})
    for _, row in news.head(MAX_NEWS).iterrows()
]

# Build FAISS vector store
db = FAISS.from_documents(docs, embedding_model)

# Example query 
query = "I want something spicy"
results = db.similarity_search(query, k=5)

# Print top results
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content} (Cuisine: {doc.metadata['cuisine']})")


# %%
# Search for top matching recipes
query = "I want something spicy"
results = db.similarity_search(query, k=1)  # Just pick the top 1 for now

# Use result as input to FLAN-T5 for generation
recipe_name = results[0].page_content

from transformers import pipeline

# Load FLAN-T5 (choose small or base)
flan_pipe = pipeline("text2text-generation", model="google/flan-t5-base")

prompt = f"Describe how to cook {recipe_name} in a traditional way."

# Generate cooking instructions
flan_output = flan_pipe(prompt, max_length=100)[0]['generated_text']
print("FLAN-T5 Recipe Instructions:")
print(flan_output)



