import os
import fitz  # PyMuPDF
import base64
import re
import shutil
from datetime import datetime
from pathlib import Path

# --- LangChain ---
from langchain.chains.summarize import load_summarize_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

# --- PDF生成 ---
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage, PageBreak, ListFlowable, ListItem
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
from reportlab.lib import colors

# .envファイルの読み込み
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# 設定
# ==========================================
API_KEY = "AIzaSyA0IZW7yEAdZ20eJ5urvVxHYGotzl1qO5c" # ご自身のキーを使用してください
MODEL_NAME = "gemini-2.5-flash" # 画像認識性能が高いモデル推奨
TOKEN_LIMIT = 700000
MAX_IMAGES = 30

DIRS = {
    'input': 'data/input_PDF',
    'output': 'data/output_PDF',
    'processed': 'data/PDF_document',
    'temp_img': 'data/temp_images' # 画像一時保存用
}

for d in DIRS.values():
    Path(d).mkdir(parents=True, exist_ok=True)

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0,
    google_api_key=API_KEY,
    convert_system_message_to_human=True
)

SYSTEM_PROMPT = """
あなたはプロのドキュメント作成者です。
提供されたテキストと画像を基に、視覚的で分かりやすい要約レポートを作成してください。
- **太字**や箇条書きを適切に使用して可読性を高めてください。
- 画像から得られた具体的な数値や傾向は必ず記述してください。
- 重要な画像については「(図1参照)」のように本文中で言及してください。
"""

# ==========================================
# 処理関数
# ==========================================
def extract_content_from_pdf(pdf_path):
    """PDFからテキスト抽出＋画像を保存し解説文を生成"""
    doc = fitz.open(pdf_path)
    full_text = ""
    image_data = [] # (path, description) のリスト
    
    print(f"🔍 解析開始: {Path(pdf_path).name}")

    # 1. テキスト抽出
    for page in doc:
        full_text += page.get_text()

    # 2. 画像抽出と保存
    print("🖼️ 画像解析中...")
    
    # 一時フォルダのクリア
    shutil.rmtree(DIRS['temp_img'], ignore_errors=True)
    Path(DIRS['temp_img']).mkdir(parents=True, exist_ok=True)

    img_count = 0
    for page_index, page in enumerate(doc):
        if img_count >= MAX_IMAGES: break
        
        image_list = page.get_images(full=True)
        for img in image_list:
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                
                if len(image_bytes) < 5000: continue # 小さい画像はスキップ

                # ファイル保存 (PDF生成用)
                img_filename = f"img_{page_index+1}_{img_count}.{ext}"
                img_path = os.path.join(DIRS['temp_img'], img_filename)
                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                # Gemini解析用 (Base64)
                img_b64 = base64.b64encode(image_bytes).decode('utf-8')
                
                msg = HumanMessage(content=[
                    {"type": "text", "text": f"この画像（図{img_count+1}）は何を表していますか？詳細に説明してください。"},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
                ])
                
                res = llm.invoke([msg])
                desc = f"【図{img_count+1}】(Page {page_index+1}): {res.content}"
                
                image_data.append({
                    "path": img_path,
                    "desc": desc,
                    "label": f"図{img_count+1}"
                })
                
                img_count += 1
                print(f"  - 画像 {img_count} を保存・解析しました")
                
            except Exception as e:
                print(f"  - 画像エラー: {e}")

    doc.close()
    
    # 画像の解説テキストをまとめる
    img_text_summary = "\n".join([d["desc"] for d in image_data])
    
    return full_text, img_text_summary, image_data

def generate_summary(text_content, image_text_summary):
    """要約の生成"""
    combined_content = f"""
    {text_content}
    
    === 以下はドキュメントに含まれる画像の分析結果です ===
    これらの情報を統合し、本文と画像を関連付けたレポートを作成してください。
    {image_text_summary}
    """
    
    num_tokens = llm.get_num_tokens(combined_content)
    print(f"📊 推定トークン数: {num_tokens}")

    prompt_text = f"{SYSTEM_PROMPT}\n\n以下の情報を統合し、Markdown形式（#見出し, **太字**, - 箇条書き）を使ってレポートを作成してください。\n\n{combined_content}"

    if num_tokens < TOKEN_LIMIT:
        response = llm.invoke(prompt_text)
        return response.content
    else:
        # 分割処理（簡略化のためMap-Reduceの定義は省略し、単純分割とします）
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=30000, chunk_overlap=1000)
        docs = text_splitter.create_documents([prompt_text])
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        result = chain.invoke(docs)
        return result['output_text']

# ==========================================
# PDF保存ロジック（Markdown対応）
# ==========================================
def format_line(text):
    """Markdownの太字等をReportLabタグに変換"""
    # **Bold** -> <b>Bold</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    return text

def save_to_pdf(text, image_data, original_filename):
    output_filename = f"{Path(original_filename).stem}_summary.pdf"
    output_path = os.path.join(DIRS['output'], output_filename)
    
    # フォント設定
    font_path = "C:/Windows/Fonts/msgothic.ttc" # Win
    if not os.path.exists(font_path):
        font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc" # Mac
    try:
        pdfmetrics.registerFont(TTFont('Japanese', font_path))
        font_name = 'Japanese'
    except:
        font_name = 'Helvetica'

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    
    styles = getSampleStyleSheet()
    # カスタムスタイル
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName=font_name, fontSize=18, spaceAfter=20)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontName=font_name, fontSize=14, spaceBefore=15, spaceAfter=10, textColor=colors.darkblue)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName=font_name, fontSize=10.5, leading=16, spaceAfter=8)
    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], fontName=font_name, fontSize=10.5, leading=16, leftIndent=20)
    caption_style = ParagraphStyle('Caption', parent=styles['Normal'], fontName=font_name, fontSize=9, textColor=colors.grey, alignment=1)

    story = []
    
    # タイトル
    story.append(Paragraph(f"要約レポート: {original_filename}", title_style))
    story.append(Paragraph(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Spacer(1, 1*cm))

    # 本文処理
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue

        # 見出し (#)
        if line.startswith('#'):
            clean_line = line.replace('#', '').strip()
            story.append(Paragraph(clean_line, h2_style))
        
        # 箇条書き (*, -)
        elif line.startswith('* ') or line.startswith('- '):
            clean_line = format_line(line[2:])
            # 中黒(・)を行頭につける
            story.append(Paragraph(f"• {clean_line}", bullet_style))
            
        # 通常テキスト
        else:
            formatted_line = format_line(line)
            story.append(Paragraph(formatted_line, body_style))

    # --- 画像セクションの追加 ---
    if image_data:
        story.append(PageBreak())
        story.append(Paragraph("添付資料・画像分析", h2_style))
        story.append(Spacer(1, 0.5*cm))
        
        for img_info in image_data:
            try:
                # 画像描画 (横幅をページ幅に合わせて調整)
                img_path = img_info['path']
                im = PDFImage(img_path)
                
                # サイズ調整 (アスペクト比保持、最大幅15cm)
                img_width = 15 * cm
                orig_w, orig_h = im.imageWidth, im.imageHeight
                ratio = img_width / orig_w
                im.drawHeight = orig_h * ratio
                im.drawWidth = img_width
                
                story.append(im)
                story.append(Spacer(1, 0.2*cm))
                
                # 画像説明
                desc_text = format_line(f"<b>{img_info['label']}</b>: {img_info['desc']}")
                story.append(Paragraph(desc_text, caption_style))
                story.append(Spacer(1, 1*cm))
                
            except Exception as e:
                print(f"画像埋め込みエラー: {e}")

    try:
        doc.build(story)
        print(f"💾 PDF保存完了: {output_path}")
        return True
    except Exception as e:
        print(f"❌ PDF保存エラー: {e}")
        return False

# ==========================================
# メイン処理
# ==========================================
def main():
    input_files = list(Path(DIRS['input']).glob('*.pdf'))
    if not input_files:
        print("ファイルがありません。")
        return

    for pdf_file in input_files:
        try:
            # 1. 抽出（画像ファイル保存含む）
            text, img_text, img_data = extract_content_from_pdf(str(pdf_file))
            
            # 2. 要約
            summary = generate_summary(text, img_text)
            
            # 3. 画像付きPDF作成
            if save_to_pdf(summary, img_data, pdf_file.name):
                # 完了移動
                new_path = Path(DIRS['processed']) / pdf_file.name
                pdf_file.rename(new_path)
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()