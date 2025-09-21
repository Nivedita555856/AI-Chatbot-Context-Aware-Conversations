import os
import re
import hashlib, langid
from flask import Flask, request, jsonify, send_from_directory, send_file
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from db import log_document_upload, authenticate_user, verify_document_access, get_connection
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_to_dict
import uuid
from datetime import datetime
from config import DOCUMENT_STORAGE
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
from langchain.schema import Document
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "multilingualrag"
TEMP_DIR = "./temp"

# DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
# os.makedirs(DOCUMENTS_DIR, exist_ok=True)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DOCUMENT_STORAGE = os.path.join(BASE_DIR, "documents")
# os.makedirs(DOCUMENT_STORAGE, exist_ok=True)

app = Flask(__name__)
pc = Pinecone(api_key=PINECONE_API_KEY)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', max_length=512)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize Pinecone index with AWS free-tier configuration
if INDEX_NAME in pc.list_indexes().names():
    pc.delete_index(INDEX_NAME)

pc.create_index(
    name=INDEX_NAME,
    dimension=384,  # This matches all-mpnet-base-v2 embedding size
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


def get_splitter(language_code):
    separators = {
        "en": ["\n\n", "\n", r"(?<!\d)\.(?!\d)", " ", ""],
        "hi": ["\n\n", "\n", "।", "॥", " ", ""],  # Hindi sentence boundaries
        "mr": ["\n\n", "\n", "।", "॥", " ", ""]  # Marathi boundaries
    }
    return RecursiveCharacterTextSplitter(
        chunk_size=750,  # Reduced for precision
        chunk_overlap=150,
        separators=separators.get(language_code, ["\n\n"]),
        keep_separator=True
    )


def detect_language(text):
    if not text.strip():
        return "en"

    # Hindi/Marathi specific checks (improved)
    devanagari_chars = re.search(r'[\u0900-\u097F]', text)
    if devanagari_chars:
        # Marathi-specific words
        marathi_indicators = ['आहे', 'म्हणून', 'आणि', 'मी', 'तू', 'आम्ही']
        # Hindi-specific words
        hindi_indicators = ['है', 'और', 'मैं', 'तुम', 'हम']

        marathi_count = sum(text.count(word) for word in marathi_indicators)
        hindi_count = sum(text.count(word) for word in hindi_indicators)

        if marathi_count > hindi_count:
            return "mr"
        return "hi"

    # Fallback to langid with confidence threshold
    lang, confidence = langid.classify(text)
    if confidence > 0.7:  # Only trust high-confidence predictions
        return lang if lang in ['hi', 'mr'] else 'en'
    return "en"


memory_store = {}  # Stores conversation memories by session ID


class MultilingualGroqLLM:
    """Wrapper to make Groq API compatible with LangChain memory"""

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.language_instructions = {
            "hi": "हिंदी में संक्षिप्त और सटीक उत्तर दें।",
            "mr": "मराठीमध्ये संक्षिप्त आणि अचूक उत्तर द्या.",
            "en": "Provide a concise answer in English."
        }

    def __call__(self, prompt: str, language: str = "en") -> str:
        lang_instruction = self.language_instructions.get(language, "")
        enhanced_prompt = f"{prompt}\n\n{lang_instruction}"
        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": enhanced_prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content


def extract_references_from_answer(answer, all_docs):
    """Extract referenced documents from the answer based on citations mentioned"""
    referenced_docs = []

    # Look for patterns like document names and page numbers in the answer
    for doc in all_docs:
        doc_name = doc.metadata.get("document_name", "unknown document")
        # page_num = doc.metadata.get("page_number", 1)

        # Check if the document is mentioned in the answer
        if doc_name.lower() in answer.lower():
            referenced_docs.append({
                "document": doc_name,

                "content": re.sub(r"\s+", " ", doc.page_content).strip()[:200] + "...",
                "relevance_score": 1.0  # High relevance since it's cited in answer
            })

    # If no specific references found, return most relevant docs (top 2)
    if not referenced_docs and all_docs:
        for doc in all_docs[:2]:  # Only top 2 most relevant
            doc_name = doc.metadata.get("document_name", "unknown document")
            # page_num = doc.metadata.get("page_number", 1)

            referenced_docs.append({
                "document": doc_name,

                "content": re.sub(r"\s+", " ", doc.page_content).strip()[:200] + "...",
                "relevance_score": getattr(doc, 'score', 0.0)
            })

    return referenced_docs


def rerank_documents(query: str,
                     documents: List[Dict[str, Any]],
                     query_lang: str,
                     top_k: int = 3) -> List[Dict[str, Any]]:
    # Prepare pairs with language metadata
    pairs = []
    for doc in documents:
        if isinstance(doc, Document):
            content = doc.page_content
            metadata = doc.metadata
        else:  # Dictionary format from Pinecone
            content = doc.get('page_content', '') or doc.get('text', '')
            metadata = doc.get('metadata', {})
        doc_lang = doc.metadata.get('language', 'en')

        # Apply language priority boost (higher score for matching language)
        lang_boost = 1.5 if doc_lang == query_lang else 1.0

        # Normalize content length to avoid bias
        normalized_content = content[:2000]  # Cap very long documents

        pairs.append((query, normalized_content, lang_boost))

    # Get raw scores
    text_pairs = [(q, c) for q, c, _ in pairs]
    scores = reranker.predict(text_pairs)

    # Apply language boosts
    boosted_scores = [score * boost for score, (_, _, boost) in zip(scores, pairs)]

    # Rank documents
    ranked = sorted(zip(documents, boosted_scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked[:top_k]]


@app.route("/start-conversation", methods=["POST"])
def start_conversation():
    data = request.json
    username = data.get("username")
    language = data.get("language", "en")
    if not username:
        return jsonify({"error": "Username required"}), 400

    session_id = str(uuid.uuid4())
    llm = MultilingualGroqLLM()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Always replace any previous session for the user
    memory_store[session_id] = {
        "memory": memory,
        "username": username,
        "language": language,
        "created_at": datetime.now().isoformat()
    }

    return jsonify({"session_id": session_id})


# Configuration
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)


@app.route("/download/<filename>", methods=["GET"])
def download_document(filename):
    """Serve files directly from the documents directory"""
    try:
        safe_filename = secure_filename(filename)
        return send_from_directory(
            directory=DOCUMENTS_DIR,
            path=safe_filename,
            as_attachment=True
        )
    except FileNotFoundError:
        return jsonify({"error": "Document not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    role = authenticate_user(username, password)
    if role:
        return jsonify({"role": role})
    else:
        return jsonify({"error": "Invalid credentials"}), 401


@app.route("/upload", methods=["POST"])
def upload():
    try:
        username = request.form.get("username")
        file = request.files.get("file")

        if not file:
            return jsonify({"error": "No file provided"}), 400

        # Read and hash file content
        file_content = file.read()
        doc_hash = hashlib.md5(file_content).hexdigest()
        file.seek(0)  # Reset for processing

        # Secure filename and set paths
        filename = secure_filename(file.filename)
        storage_path = os.path.join(DOCUMENTS_DIR, filename)  # Now using DOCUMENTS_DIR

        # Save original file to documents folder
        with open(storage_path, 'wb') as f:
            f.write(file_content)

        # Process document for embeddings (unchanged)
        ext = filename.lower().split(".")[-1]
        if ext == "pdf":
            loader = PyPDFLoader(storage_path)
        elif ext == "docx":
            loader = Docx2txtLoader(storage_path)
        elif ext == "csv":
            loader = CSVLoader(storage_path)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        documents = loader.load()

        # Simplified metadata - keeps what's essential for RAG
        for doc in documents:
            doc.metadata.update({
                "document_name": filename,  # Key for downloads
                # "page_number": doc.metadata.get("page", 0) + 1 if ext == "pdf" else 1,
                # Removed storage_path and document_hash as they're not needed for the simplified approach
            })

        # Store in vector database (unchanged)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        vector_store = PineconeVectorStore(embedding=embedding_model, index_name=INDEX_NAME)
        vector_store.add_documents(docs)

        # # Minimal database logging (just filename tracking)
        # conn = get_connection()
        # cur = conn.cursor()
        # # cur.execute(
        # #     """INSERT INTO stored_documents
        # #            (filename, uploaded_by)
        # #        VALUES (%s, %s) ON CONFLICT (filename) DO NOTHING""",
        # #     (filename, username)
        # # )
        # conn.commit()
        # conn.close()

        return jsonify({
            "message": "Document processed successfully",
            "filename": filename  # Return filename instead of hash
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/bulk-upload", methods=["POST"])
def bulk_upload():
    """Handle multiple file uploads with simplified storage"""
    username = request.form.get("username")
    if not username:
        return jsonify({"error": "Authentication required"}), 401

    files = request.files.getlist("files[]")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    success_files = []
    failed_files = []
    conn = get_connection()
    cur = conn.cursor()

    for file in files:
        try:
            # Process each file
            filename = secure_filename(file.filename)
            storage_path = os.path.join(DOCUMENTS_DIR, filename)

            # Save original file directly to documents folder
            file.save(storage_path)

            # Process document for embeddings (unchanged core functionality)
            ext = filename.lower().split(".")[-1]
            if ext == "pdf":
                loader = PyPDFLoader(storage_path)
            elif ext == "docx":
                loader = Docx2txtLoader(storage_path)
            elif ext == "csv":
                loader = CSVLoader(storage_path)
            else:
                failed_files.append(filename)
                continue

            documents = loader.load()

            # Essential metadata only
            for doc in documents:
                doc.metadata.update({
                    "document_name": filename,
                    "page_number": doc.metadata.get("page", 0) + 1 if ext == "pdf" else 1
                })

            # Store in vector database (unchanged)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.split_documents(documents)
            vector_store = PineconeVectorStore(embedding=embedding_model, index_name=INDEX_NAME)
            vector_store.add_documents(docs)

            # Minimal database logging
            cur.execute(
                """INSERT INTO stored_documents
                       (filename, uploaded_by)
                   VALUES (%s, %s) ON CONFLICT (filename) DO NOTHING""",
                (filename, username)
            )

            success_files.append(filename)
        except Exception as e:
            failed_files.append(filename)
            app.logger.error(f"Error processing {filename}: {str(e)}")
            # Remove failed file if it was partially saved
            if os.path.exists(storage_path):
                os.remove(storage_path)

    # Commit all successful uploads at once
    conn.commit()
    conn.close()

    return jsonify({
        "message": "Bulk upload completed",
        "success": success_files,
        "failed": failed_files,
        "total_processed": len(success_files) + len(failed_files)
    })


@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question")
    username = data.get("username")
    session_id = data.get("session_id")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if not session_id or session_id not in memory_store:
        # Auto-generate new session
        detected_lang = detect_language(question)
        session_id = str(uuid.uuid4())
        llm = MultilingualGroqLLM()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory_store[session_id] = {
            "memory": memory,
            "username": username,
            "language": detected_lang,
            "created_at": datetime.now().isoformat()
        }
    try:
        memory = memory_store[session_id]["memory"]
        preferred_lang = memory_store[session_id].get("language", "en")
        print(preferred_lang)
        # Get relevant context from vector store
        vector_store = PineconeVectorStore(embedding=embedding_model, index_name=INDEX_NAME)
        raw_results = vector_store.similarity_search(question, k=10)

        def rerank_documents(query, docs, top_k=3):
            pairs = [(query, doc.page_content) for doc in docs]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in ranked[:top_k]]

        reranked_results = rerank_documents(question, raw_results)

        # 3. Prepare context with language markers
        context_parts = []
        for doc in reranked_results:
            lang = doc.metadata.get('language', detect_language(doc.page_content))
            context_parts.append(f"[{lang.upper()}] {doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)
        # Get conversation history
        history = memory.load_memory_variables({})["chat_history"]

        prompt = f"""Role: You are an official RCF (Rashtriya Chemicals & Fertilizers) AI assistant.
         Your ONLY task is to answer user queries in the exact language requested by the user (`{preferred_lang}`). Never deviate.  

### Strict Rules:  
1. Language Lock:  
   - If `{preferred_lang}` = Marathi, respond **ONLY in Marathi**. Never use Hindi/English words or phrases.  
   - If the user asks in Hindi/English, respond in the **same language** as the query.  
   - Example: User asks in Marathi → Marathi reply. User asks in Hindi → Hindi reply.  

2. Terminology:  
   - Use only official RCF terms** from documents.  
   - Never guess acronyms. Example:  
     - Correct (if in document): "ALF म्हणजे Accommodation Licence Fee."  
     - Incorrect (if not in document): "ALF चा अर्थ शोधून घ्या."  

3. Context Handling:  
   - **Ignore past conversations** if the query is new.  
   - For follow-ups, use **only Marathi** (or selected language) + document context.  

4. Errors & Unknowns:  
   - If the answer isn’t in documents, say:  
     - Marathi: "ही माहिती RCF च्या दस्तऐवजात उपलब्ध नाही. कृपया अधिक स्पष्टीकरण द्या."  
     - Hindi/English: Not allowed if `{preferred_lang}` = Marathi.  

 
### Response Template (Marathi):  
1. Direct Answer: 1-2 lines.  
2. Document Proof: "RCF दस्तऐवजानुसार: [विवरण]."  
3. No Mixed Languages**: Even proper nouns (e.g., "HR Policy") must be translated/explained in Marathi.  

**Note**: Any language switch = Critical failure. Terminate and restart in `{preferred_lang}`.  

Inputs:
Document Context (may be in multiple languages):
{context}
Conversation History:
{history}
Question:{question}

Begin crafting a clear and helpful answer below:
Answer:"""

        # Get response from LLM
        llm = MultilingualGroqLLM()
        answer = llm(prompt, language=preferred_lang)
        # FAQ generation prompt - now in preferred language
        faq_instruction = {
            "en": """Generate 4 distinct FAQ questions in English that are:""",
            "hi": """हिंदी में 4 विशिष्ट प्रश्न उत्पन्न करें जो:""",
            "mr": """मराठीमध्ये 4 वेगळे प्रश्न तयार करा जे:"""
        }.get(preferred_lang, "en")

        faq_prompt = f"""
        You are a helpful assistant generating follow-up questions in {preferred_lang}.
        {faq_instruction}
        1. Relevant to the context but different from the main question
        2. Cover different aspects of the topic
        3. Are practical and actionable
        4. Use varied question types (how, what, why, etc.)
        5. Please don't switch languages.

        Document Context:
        {context}

        User Query:
        {question}

        Return only the 4 questions numbered in {preferred_lang}:
        1. Always prioritize document search to find the most accurate answer.
        2. Use conversation history only when the context shows continuity.
        3. Do not mention technical terms like "document context" or "conversation history" in your response.
        4. Avoid repeating unnecessary background or disclaimers—be concise and direct.
        5. Use a professional, respectful, and supportive tone.
        6. Do NOT repeat, rephrase or paraphrase the User Query.
        7. Each question MUST be grammatically correct, natural sounding, and highly relevant to the Document Context.
        8. Questions MUST originate only from real information in the 'Document Context' — no assumptions.
        9. Your questions should guide the user to other helpful and related parts of the document.
        10. NEVER generate vague or incomplete questions.
        11. Give exactly 4 distinct, high-quality FAQ-style questions.
        12. Output ONLY the numbered list.
        13. Make sure the questions are well framed and left aligned.
        14. Do not give self made questions whose answers are not there in the document provided.
        15. Please provide relevant questions which are more of the document and related to the query.
        16. Only give the questions in the that preferred language, do not provide it in english extra.
        17. If the questions are not present then it can reduce the number of FAQs to 0/1/2/3
        18. Please if possible then take the questions from 'DOCUMENT_QUEST'. 
        19. Please generate questions from the document when you are taking the answer from.
        """

        faq_response = llm(faq_prompt, language=preferred_lang)
        faqs = re.findall(r"\d\.\s*(.*)", faq_response)

        # Save to memory
        memory.save_context({"input": question}, {"output": answer})
        used_references = []
        for doc in reranked_results:
            # Check if the document content appears in the answer
            if doc.page_content.strip() and doc.page_content.strip() in answer:
                used_references.append({
                    "document": doc.metadata.get("document_name", "unknown document"),
                    # "page": doc.metadata.get("page_number", 1),
                    "content": re.sub(r"\s+", " ", doc.page_content).strip()[:200] + "..."
                })
            # Alternatively, check for key phrases from the document in the answer
            else:
                # Take first few meaningful words from document
                key_phrase = " ".join(doc.page_content.strip().split()[:10])
                if key_phrase and key_phrase in answer:
                    used_references.append({
                        "document": doc.metadata.get("document_name", "unknown document"),
                        # "page": doc.metadata.get("page_number", 1),
                        "content": re.sub(r"\s+", " ", doc.page_content).strip()[:200] + "..."
                    })

        # If no direct matches found, use the most relevant document
        if not used_references and reranked_results:
            used_references.append({
                "document": reranked_results[0].metadata.get("document_name", "unknown document"),
                # "page": reranked_results[0].metadata.get("page_number", 1),
                "content": re.sub(r"\s+", " ", reranked_results[0].page_content).strip()[:200] + "..."
            })
        return jsonify({
            "answer": answer,
            "suggestions": faqs[:4],
            "session_id": session_id,
            "detected_language": detect_language(question),
            "response_language": preferred_lang,
            "history": messages_to_dict(history),
            "references": used_references[:5]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend is running"})


@app.route("/debug-index", methods=["GET"])
def debug_index():
    index_stats = pc.describe_index(INDEX_NAME)
    vector_count = pc.Index(INDEX_NAME).describe_index_stats()["total_vector_count"]
    return jsonify({
        "index_status": index_stats,
        "vector_count": vector_count
    })


@app.route("/logout", methods=["POST"])
def logout():
    data = request.json
    session_id = data.get("session_id")
    if session_id in memory_store:
        del memory_store[session_id]
    return jsonify({"message": "Logged out successfully"})


if __name__ == "__main__":
    app.run(debug=True, port=8000)