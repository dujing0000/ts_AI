import os
import glob
import numpy as np
import google.generativeai as genai
from pypdf import PdfReader
from dotenv import load_dotenv
# .envファイルの読み込み
load_dotenv()

# ==========================================
# 設定・初期化
# ==========================================

# TODO: ここにGeminiのAPIキーを入れてください
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 埋め込みモデルと生成モデルの設定
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.0-flash"

# PDFフォルダ
PDF_DIR = os.path.join("data", "output_PDF")

# RAGを使用する類似度の閾値 (0.0〜1.0)
# この値を調整することで「関連性が高い」の判定基準を変えられます
SIMILARITY_THRESHOLD = 0.6

# キャラクター設定 (TOON形式でプロンプトに挿入されます)
CHARACTER_SETTINGS = """
you are a chatbot of zonewatch(name)
character{name, personality, tone}:
GeminiBot, Helpful and intelligent assistant, Polite and concise
"""

# ==========================================
# 関数定義: PDF処理 & RAG関連
# ==========================================

def load_pdfs(directory):
    """指定ディレクトリのPDFからテキストを抽出してチャンク化する"""
    documents = []
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Folder created: {directory}. Please put PDF files here.")
        return []

    files = glob.glob(os.path.join(directory, "*.pdf"))
    if not files:
        return []

    print("Loading PDFs...")
    for file_path in files:
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # 簡易的なチャンク分割 (500文字区切り)
            chunk_size = 500
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                if len(chunk) > 50: # 短すぎるノイズは除外
                    documents.append(chunk)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return documents

def get_embedding(text):
    """テキストをベクトル化する"""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']

def find_relevant_docs(query, documents, doc_embeddings):
    """クエリに関連するドキュメントを検索する"""
    if not documents:
        return None, 0.0

    query_embedding = get_embedding(query)
    
    # コサイン類似度の計算
    scores = []
    for doc_emb in doc_embeddings:
        dot_product = np.dot(query_embedding, doc_emb)
        norm_a = np.linalg.norm(query_embedding)
        norm_b = np.linalg.norm(doc_emb)
        score = dot_product / (norm_a * norm_b)
        scores.append(score)
    
    max_score = max(scores)
    best_idx = scores.index(max_score)
    
    return documents[best_idx], max_score

# ==========================================
# 関数定義: TOONフォーマット変換
# ==========================================

def format_history_toon(history_list):
    """
    会話履歴をTOON形式(JSONよりトークン削減)に変換する
    Format:
    history[N]{role, content}:
    user, ...
    model, ...
    """
    if not history_list:
        return ""
    
    count = len(history_list)
    toon_str = f"\nhistory[{count}]{{role, content}}:\n"
    
    for msg in history_list:
        # コンマが含まれるとCSVとして崩れる可能性があるため、最低限のエスケープまたは置換を行う
        # ここではシンプルに改行をスペースにし、コンマを全角にする等で対処
        clean_content = msg['content'].replace("\n", " ").replace(",", "，")
        toon_str += f"{msg['role']}, {clean_content}\n"
        
    return toon_str

# ==========================================
# メインループ
# ==========================================

def main():
    # 1. PDF読み込みとベクトル化（起動時に一度だけ実行）
    docs = load_pdfs(PDF_DIR)
    doc_embeddings = []
    if docs:
        print(f"Embedding {len(docs)} document chunks... (Please wait)")
        # バッチ処理ではなくシンプルにループで処理
        for doc in docs:
            # embeddingのtask_typeはdocumentにする
            emb = genai.embed_content(model=EMBEDDING_MODEL, content=doc, task_type="retrieval_document")['embedding']
            doc_embeddings.append(emb)
        print("Ready!")
    else:
        print("No PDFs found or processed. Running in standard mode.")

    # 会話履歴の保持
    history = []
    model = genai.GenerativeModel(GENERATION_MODEL)

    print("\n--- Chatbot Started (type 'exit' to quit) ---")

    while True:
        user_input = input("\nYou > ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # --- RAG 判定ロジック ---
        rag_context = ""
        is_rag_used = False
        
        if docs:
            best_doc, score = find_relevant_docs(user_input, docs, doc_embeddings)
            
            # 閾値を超えた場合のみRAGを使用
            if score >= SIMILARITY_THRESHOLD:
                is_rag_used = True
                # TOON形式で参照資料を渡す
                rag_context = f"\nreference_material{{content}}:\n{best_doc}\n"
            else:
                # 関連性が低い場合はRAGを使わない
                is_rag_used = False

        # --- プロンプト構築 (TOON形式) ---
        # 1. キャラクター設定
        # 2. RAG資料 (あれば)
        # 3. 会話履歴
        # 4. 現在の質問
        
        history_toon = format_history_toon(history)
        
        prompt = f"""
{CHARACTER_SETTINGS}

{rag_context}

{history_toon}

current_input{{role, content}}:
user, {user_input}

instruction:
Respond to the user input based on the character settings and history.
If 'reference_material' is provided, use it to answer.
Output only the response text.
"""

        # --- 生成実行 ---
        try:
            response = model.generate_content(prompt)
            reply_text = response.text.strip()

            # --- 出力整形 ---
            if is_rag_used:
                print(f"chat : (RAG) {reply_text}")
            else:
                print(f"chat : {reply_text}")

            # 履歴に追加
            history.append({"role": "user", "content": user_input})
            history.append({"role": "model", "content": reply_text})

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()