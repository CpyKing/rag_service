HUGGINGFACEHUB_API_TOKEN = ""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

class RAGDatabase():
    def __init__(self):
        data_paths = [
            "/home/zzz/rag_service/data/Fashion_Shop_Consultant/train.csv",
            "/home/zzz/rag_service/data/Fashion_Shop_Consultant/test.csv",
            "/home/zzz/rag_service/data/clothes_shop_consultant/train.csv",
            "/home/zzz/rag_service/data/clothes_shop_consultant/test.csv"
        ]
        docs = []
        doc_set = set()
        for p in data_paths:
            loader = CSVLoader(
                file_path=p,
            )
            for doc in loader.load():
                doc.page_content = doc.page_content.replace('"', '')
                doc.page_content = doc.page_content.split('\n')[1] + ' ' + doc.page_content.split('\n')[2] + '\n'
                if doc.page_content not in doc_set:
                    docs.append(doc)
                    doc_set.add(doc.page_content)
        del doc_set
        print(f"len docs {len(docs)}")
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=0,
            length_function=len,
        )
        docs = text_spliter.split_documents(docs)
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
        
        self.db = FAISS.from_documents(docs, embeddings)
    
    def similarity_search(self, query, top_k):
        return self.db.similarity_search(query , k = top_k)

if __name__ == '__main__':
    data_paths = [
        "/home/zzz/rag_service/data/Fashion_Shop_Consultant/train.csv",
        "/home/zzz/rag_service/data/Fashion_Shop_Consultant/test.csv",
        "/home/zzz/rag_service/data/clothes_shop_consultant/train.csv",
        "/home/zzz/rag_service/data/clothes_shop_consultant/test.csv"
    ]
    docs = []
    doc_set = set()
    for p in data_paths:
        loader = CSVLoader(
            file_path="/home/zzz/rag_service/data/Fashion_Shop_Consultant/train.csv",
        )
        for doc in loader.load():
            doc.page_content = doc.page_content.replace('"', '')
            doc.page_content = doc.page_content.split('\n')[1] + ' ' + doc.page_content.split('\n')[2] + '\n'
            if doc.page_content not in doc_set:
                docs.append(doc)
                doc_set.add(doc.page_content)
    del doc_set
    print(f"len docs {len(docs)}")
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=0,
        length_function=len,
    )
    docs = text_spliter.split_documents(docs)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    
    db = FAISS.from_documents(docs, embeddings)
    import pdb;pdb.set_trace()
