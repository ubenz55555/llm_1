# !pip install -U -q langchain openai ragas
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()
## getting data from HuggingFace Hub 
from datasets import load_dataset

fiqa_eval = load_dataset("explodinggradients/fiqa", "ragas_eval")
fiqa_eval
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision
)
from ragas import evaluate

result = evaluate(
    fiqa_eval["baseline"].select(range(1)),
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall
    ],
)

result
df = result.to_pandas()
df.head()
# !pip install -U -q langchain openai ragas arxiv pymupdf chromadb wandb tiktoken tqdm datasets
import os
import openai
from getpass import getpass

openai.api_key = getpass("Please provide your OpenAI Key: ")
os.environ["OPENAI_API_KEY"] = openai.api_key

from langchain.document_loaders import ArxivLoader

base_docs = ArxivLoader(query="Retrieval Augmented Generation", load_max_docs=5).load()
len(base_docs)
for doc in base_docs:
  print(doc.metadata)
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(chunk_size=250)
docs = text_splitter.split_documents(base_docs)
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
base_retriever = vectorstore.as_retriever(search_kwargs={"k" : 5})
relevant_docs = base_retriever.get_relevant_documents("What is Retrieval Augmented Generation?")

for rel_doc in relevant_docs:
  print(rel_doc)

from langchain.prompts import ChatPromptTemplate

template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

### CONTEXT
{context}

### QUESTION
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

retrieval_augmented_qa_chain = (
    {"context": itemgetter("question")
    | base_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
)
question = "What is RAG?"

result = retrieval_augmented_qa_chain.invoke({"question" : question})

print(result)
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

question_schema = ResponseSchema(
    name="question",
    description="a question about the context."
)

question_response_schemas = [
    question_schema,
]
question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)
format_instructions = question_output_parser.get_format_instructions()

format_instructions
question_generation_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")

bare_prompt_template = "{content}"
bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)
from langchain.prompts import ChatPromptTemplate

qa_template = """\
You are a University Professor creating a test for advanced students. For each context, create a question that is specific to the context. Avoid creating generic or general questions.

question: a question about the context.

Format the output as JSON with the following keys:
question

context: {context}
"""

prompt_template = ChatPromptTemplate.from_template(template=qa_template)

messages = prompt_template.format_messages(
    context=docs[0],
    format_instructions=format_instructions
)

question_generation_chain = bare_template | question_generation_llm

response = question_generation_chain.invoke({"content" : messages})
output_dict = question_output_parser.parse(response.content)
for k, v in output_dict.items():
  print(k)
  print(v)
from tqdm import tqdm

qac_triples = []

for text in tqdm(docs[:10]):
  messages = prompt_template.format_messages(
      context=text,
      format_instructions=format_instructions
  )
  response = question_generation_chain.invoke({"content" : messages})
  try:
    output_dict = question_output_parser.parse(response.content)
  except Exception as e:
    continue
  output_dict["context"] = text
  qac_triples.append(output_dict)

qac_triples[2]
answer_generation_llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

answer_schema = ResponseSchema(
    name="answer",
    description="an answer to the question"
)

answer_response_schemas = [
    answer_schema,
]

answer_output_parser = StructuredOutputParser.from_response_schemas(answer_response_schemas)
format_instructions = answer_output_parser.get_format_instructions()

qa_template = """\
You are a University Professor creating a test for advanced students. For each question and context, create an answer.

answer: a answer about the context.

Format the output as JSON with the following keys:
answer

question: {question}
context: {context}
"""

prompt_template = ChatPromptTemplate.from_template(template=qa_template)

messages = prompt_template.format_messages(
    context=qac_triples[0]["context"],
    question=qac_triples[0]["question"],
    format_instructions=format_instructions
)

answer_generation_chain = bare_template | answer_generation_llm

response = answer_generation_chain.invoke({"content" : messages})
output_dict = answer_output_parser.parse(response.content)

for k, v in output_dict.items():
  print(k)
  print(v)
for triple in tqdm(qac_triples):
  messages = prompt_template.format_messages(
      context=triple["context"],
      question=triple["question"],
      format_instructions=format_instructions
  )
  response = answer_generation_chain.invoke({"content" : messages})
  try:
    output_dict = answer_output_parser.parse(response.content)
  except Exception as e:
    continue
  triple["answer"] = output_dict["answer"]

import pandas as pd
from datasets import Dataset

ground_truth_qac_set = pd.DataFrame(qac_triples)
ground_truth_qac_set["context"] = ground_truth_qac_set["context"].map(lambda x: str(x.page_content))
ground_truth_qac_set = ground_truth_qac_set.rename(columns={"answer" : "ground_truth"})


eval_dataset = Dataset.from_pandas(ground_truth_qac_set)
eval_dataset
eval_dataset.to_csv("groundtruth_eval_dataset.csv")
import requests

url = 'groundtruth_eval_dataset.csv'  
r = requests.get(url)
with open('groundtruth_eval_dataset.csv', 'wb') as file:
    file.write(r.content)

from datasets import Dataset
eval_dataset = Dataset.from_csv("groundtruth_eval_dataset.csv")
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity
)

from ragas.metrics.critique import harmfulness
from ragas import evaluate
def create_ragas_dataset(rag_pipeline, eval_dataset):
  rag_dataset = []
  for row in tqdm(eval_dataset):
    answer = rag_pipeline.invoke({"question" : row["question"]})  
    rag_dataset.append(
        {"question" : row["question"],
         "answer" : answer["response"].content,
         "contexts" : [context.page_content for context in answer["context"]],
         "ground_truths" : [row["ground_truth"]]
         }
    )
  rag_df = pd.DataFrame(rag_dataset)
  rag_eval_dataset = Dataset.from_pandas(rag_df)
  return rag_eval_dataset
def evaluate_ragas_dataset(ragas_dataset):
  result = evaluate(
    ragas_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    ],
  )
  return result

from tqdm import tqdm
import pandas as pd

basic_qa_ragas_dataset = create_ragas_dataset(retrieval_augmented_qa_chain, eval_dataset)
basic_qa_ragas_dataset

basic_qa_ragas_dataset[0]
basic_qa_ragas_dataset.to_csv("basic_qa_ragas_dataset.csv")
basic_qa_result = evaluate_ragas_dataset(basic_qa_ragas_dataset)
df = basic_qa_result.to_pandas()
df.head()

def create_qa_chain(retriever):
  primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
  created_qa_chain = (
    {"context": itemgetter("question") | retriever,
     "question": itemgetter("question")
    }
    | RunnablePassthrough.assign(
        context=itemgetter("context")
      )
    | {
         "response": prompt | primary_qa_llm,
         "context": itemgetter("context"),
      }
  )

  return created_qa_chain

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

vectorstore = Chroma(collection_name="split_parents", embedding_function=OpenAIEmbeddings())
store = InMemoryStore()

parent_document_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

parent_document_retriever.add_documents(base_docs)
parent_document_retriever_qa_chain = create_qa_chain(parent_document_retriever)
parent_document_retriever_qa_chain.invoke({"question" : "What is RAG?"})["response"].content
pdr_qa_ragas_dataset = create_ragas_dataset(parent_document_retriever_qa_chain, eval_dataset)
pdr_qa_ragas_dataset.to_csv("pdr_qa_ragas_dataset.csv")
pdr_qa_result = evaluate_ragas_dataset(pdr_qa_ragas_dataset)
pdr_qa_df = pdr_qa_result.to_pandas()
pdr_qa_df.head()
