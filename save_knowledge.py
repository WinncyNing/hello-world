from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_milvus import Milvus
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


URI = "/Users/winncyning/Desktop/毕设/hello-world/vector_base/test.db"
embedding_model_name = "/Users/winncyning/Desktop/毕设/model/bge-large-zh-v1.5"
model_kwargs = {"device": "mps"}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
vector_store = Milvus(
    embedding_function=embedding_model,
    connection_args={"uri": URI}        
)
def load_file_and_split(file_path):
    loader = TextLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=10)
    documents = loader.load_and_split(text_splitter=text_splitter)
    return documents


def store_in_milvus(embedded_chunks):
    ids = [str(i) for i in range(len(embedded_chunks))]
    vector_store.add_documents(documents=embedded_chunks, ids=ids)


if __name__ == '__main__':
    file_path = "/Users/winncyning/Desktop/毕设/hello-world/documents/云商术语.txt"  # 替换为你的文件路径
    documents = load_file_and_split(file_path)
    # store_in_milvus(documents)