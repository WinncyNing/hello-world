from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_milvus import Milvus

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

if __name__ == "__main__":
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3},
    )
    print(retriever.invoke("财票：财务公司承兑汇票是指企业集团财务公司承兑的商业汇票。"))