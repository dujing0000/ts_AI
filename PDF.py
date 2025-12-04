import os
import re
import fitz
import base64
import shutil
from pathlib import Path
from datetime import datetime

# --- PDF ---
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage, 
    Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Line

# 環境変数読み込み
load_dotenv()

# ==========================================
# 設定
# ==========================================
API_KEY = os.getenv('GOOGLE_API_KEY') 
MODEL_NAME = "gemini-2.5-flash" 
MAX_IMAGES = 50

DIRS = {
    'input': 'data/input_PDF',
    'output': 'data/output_PDF',
    'processed': 'data/PDF_document',
    'temp_img': 'data/temp_images'
}

for d in DIRS.values():
    Path(d).mkdir(parents=True, exist_ok=True)

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0,
    google_api_key=API_KEY,
    convert_system_message_to_human=True
)

user_instruction = input("質問または指示を入力してください:")

SYSTEM_PROMPT = f"""
あなたは優秀なビジネスアナリスト兼ドキュメント作成のスペシャリストです。
提供されたPDFの内容（テキストおよび画像リスト）を分析し、以下の**「ユーザーの指示」**に最も適した形式でレポートを作成してください。

=========================================
ユーザーの指示:
「{user_instruction}」
=========================================

## 作成ルール（思考プロセス）
まず、ユーザーの指示の「意図」を判断し、以下の**モードA**または**モードB**のいずれかのアプローチを採用してください。

### 【モードA：特定の質問・テーマが指定された場合】
ユーザーが「〜の対策は？」「〜を数字や表で示して」「〜のリスクについて」など、**特定のトピック**に焦点を当てている場合。
- **方針**: 全体の要約は最小限にし、**質問への回答を最優先**してください。
- **構成**:
  1. **結論（ダイレクトアンサー）**: 質問に対する答えをズバリ記述。
  2. **根拠となるデータ・詳細**: 本文中の該当箇所を深掘りし、表や箇条書きで詳述。
  3. **関連情報**: 質問に関連する周辺情報。
  
### 【モードB：一般的な要約・解説の場合】
ユーザーが「要約して」「わかりやすく解説して」「重要ポイントをまとめて」など、**全体把握**を求めている場合。
- **方針**: 全体を網羅し、重要度順に構造化してください。
- **構成**: 下記の「標準テンプレート」に従ってください。

---

## 共通の出力ガイドライン（Markdown形式）
どのような指示であっても、以下の技術的制約を必ず守ってください。

1. **画像と表の活用**
   - 画像タグ（[[IMG: 図X]]）は、必ず前後のテキストとは別の行（独立した行）に配置してください。
   - 文脈に合った画像があれば、その位置に `[[IMG: 画像ラベル]]` というタグをそのまま記述してください（例: `[[IMG: 図1]]`）。
   - 比較や数値データは必ず Markdown の表で整理してください。
   - **重要**: Markdownの表の中で改行コードを使わないでください（崩れる原因になります）。

2. **文章のトーン**
   - 日本語で記述。
   - 専門用語には簡潔な補足を付記。
   - 推測は行わず、PDFに記載されている事実のみをベースにする。

3. **出力フォーマット**
   PythonのスクリプトでPDF化するため、以下の見出し記法（#）を使用してください。

---
(以下、モードBの場合の標準テンプレート。モードAの場合は、見出しタイトルを質問に合わせて適宜変更して構いません)

## 1. ハイライト / 結論
（ここに、ユーザーの指示に対する最も核心的な回答、または全体の要約を記述）

## 2. 重要ポイント解説
- **ポイント1**: （詳細）
- **ポイント2**: （詳細）
- **ポイント3**: （詳細）

## 3. 詳細分析とデータ
（必要に応じて図を挿入）
[[IMG: 図1]]
### [主要トピックA]
- 内容

（必要に応じてデータ比較表）
| 項目 | 内容A | 内容B |
|------|------|------|
### [主要トピックB]
- 内容

### [関連するサブトピック]
- （解説）

## 4. まとめ・考察
- （全体を通しての示唆、または今後の課題など）
"""

# ==========================================
# ユーティリティ
# ==========================================
def get_jp_font_name():
    """日本語フォントの自動検出"""
    font_paths = [
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont('Japanese', path))
                return 'Japanese'
            except:
                continue
    return 'Helvetica'

JP_FONT = get_jp_font_name()

# ==========================================
# コンテンツ抽出・解析
# ==========================================
def extract_content_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    image_data = {} 
    
    print(f"🔍 解析開始: {Path(pdf_path).name}")

    for page in doc:
        full_text += page.get_text()

    print("🖼️ 画像解析中...")
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
                
                # 小さい画像は除外
                if len(image_bytes) < 5000: continue 

                img_filename = f"img_{img_count+1}.{base_image['ext']}"
                img_path = os.path.join(DIRS['temp_img'], img_filename)
                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                # Geminiによる画像キャプション生成
                img_b64 = base64.b64encode(image_bytes).decode('utf-8')
                msg = HumanMessage(content=[
                    {"type": "text", "text": f"この画像（図{img_count+1}）は何の画像ですか？15文字以内で簡潔に答えてください。"},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
                ])
                res = llm.invoke([msg])
                
                label = f"図{img_count+1}"
                image_data[label] = {
                    "path": img_path,
                    "caption": res.content.strip(),
                    "label": label
                }
                img_count += 1
                print(f"  - {label} 検出: {res.content.strip()}")
            except Exception as e:
                print(f"  - 画像スキップ: {e}")

    doc.close()
    img_list_text = "\n".join([f"{k}: {v['caption']}" for k, v in image_data.items()])
    return full_text, img_list_text, image_data

def generate_summary(text_content, image_list_text):
    combined_content = f"""
    === ドキュメント全文 ===
    {text_content}
    
    === 利用可能な画像リスト ===
    {image_list_text}
    """
    
    prompt_text = f"{SYSTEM_PROMPT}\n\n以下の情報を基に、省略せずに完全なレポートを作成してください。\n\n{combined_content}"
    
    print(f"🚀 AI生成開始...")
    response = llm.invoke(prompt_text)
    return response.content

# ==========================================
# PDF生成 (ReportLab)
# ==========================================
def parse_markdown_table(markdown_lines):
    data = []
    for line in markdown_lines:
        row = [cell.strip() for cell in line.strip('|').split('|')]
        if len(row) > 0:
            data.append(row)
    return data

def format_inline_bold(text):
    if not text: return ""
    # ReportLab用タグ修正
    text = re.sub(r'<br\s*/?>', '<br/>', text, flags=re.IGNORECASE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    return text

def create_paragraph_table(raw_data_lines, styles, available_width):
    if not raw_data_lines: return None

    parsed_data = parse_markdown_table(raw_data_lines)
    
    # セパレータ行除去
    clean_data = [row for row in parsed_data if not (len(row) > 0 and set("".join(row)).issubset({'-', ':', ' '}))]
    if not clean_data: return None

    # 列数正規化
    max_cols = max(len(row) for row in clean_data)
    normalized_data = [row + [''] * (max_cols - len(row)) for row in clean_data]

    col_width = available_width / max_cols
    col_widths = [col_width] * max_cols

    style_cell_center = ParagraphStyle('TableCellC', parent=styles['Normal'], fontName=JP_FONT, fontSize=9, alignment=1, leading=11)
    style_cell_left = ParagraphStyle('TableCellL', parent=styles['Normal'], fontName=JP_FONT, fontSize=9, alignment=0, leading=11)
    
    table_data = []
    for i, row in enumerate(normalized_data):
        converted_row = []
        for cell_text in row:
            s = style_cell_center if i == 0 else style_cell_left
            p = Paragraph(format_inline_bold(cell_text), s)
            converted_row.append(p)
        table_data.append(converted_row)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('FONT', (0,0), (-1,-1), JP_FONT, 9),
        ('BACKGROUND', (0,0), (-1,0), colors.aliceblue),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BOX', (0,0), (-1,-1), 1.0, colors.black),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    
    # 行数が少ない場合は分割禁止
    return KeepTogether([t]) if len(clean_data) < 30 else t

def save_to_pdf(markdown_text, image_data_dict, original_filename):
    output_filename = f"{Path(original_filename).stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    output_path = os.path.join(DIRS['output'], output_filename)
    
    margin = 2*cm
    doc = SimpleDocTemplate(output_path, pagesize=A4, margin=[margin]*4)
    available_width = A4[0] - (2 * margin)
    
    styles = getSampleStyleSheet()
    
    # スタイル定義 (keepWithNextでヘッダー分離防止)
    style_title = ParagraphStyle('MainTitle', parent=styles['Heading1'], fontName=JP_FONT, fontSize=24, leading=30, alignment=1, spaceAfter=20)
    style_h1 = ParagraphStyle('WikiH1', parent=styles['Heading2'], fontName=JP_FONT, fontSize=18, leading=22, spaceBefore=5, spaceAfter=10, textColor=colors.black, keepWithNext=True)
    style_h2 = ParagraphStyle('WikiH2', parent=styles['Heading3'], fontName=JP_FONT, fontSize=14, leading=18, spaceBefore=15, spaceAfter=5, textColor=colors.darkblue, keepWithNext=True)
    style_h3 = ParagraphStyle('WikiH3', parent=styles['Normal'], fontName=JP_FONT, fontSize=12, leading=16, spaceBefore=10, spaceAfter=2, textColor=colors.black, keepWithNext=True)
    style_body = ParagraphStyle('WikiBody', parent=styles['Normal'], fontName=JP_FONT, fontSize=10.5, leading=16, spaceAfter=6)
    style_bullet = ParagraphStyle('WikiBullet', parent=styles['Normal'], fontName=JP_FONT, fontSize=10.5, leading=16, leftIndent=15, spaceAfter=2)
    style_caption = ParagraphStyle('Caption', parent=styles['Normal'], fontName=JP_FONT, fontSize=9, textColor=colors.dimgrey, alignment=1)

    story = []
    
    # タイトル
    story.append(Paragraph(f"{Path(original_filename).stem} 要約・解説 レポート", style_title))
    d_line = Drawing(available_width, 5*mm)
    d_line.add(Line(0, 2*mm, available_width, 2*mm, strokeColor=colors.black, strokeWidth=2))
    story.append(d_line)
    story.append(Spacer(1, 1*cm))

    lines = markdown_text.split('\n')
    table_buffer = []
    in_table = False

    for line in lines:
        line = line.strip()
        
        # テーブル処理
        if line.startswith('|'):
            in_table = True
            table_buffer.append(line)
            continue
        else:
            if in_table:
                t = create_paragraph_table(table_buffer, styles, available_width)
                if t:
                    story.append(Spacer(1, 0.2*cm))
                    story.append(t)
                    story.append(Spacer(1, 0.5*cm))
                table_buffer = []
                in_table = False

        if not line: continue

        # 画像タグ処理
        img_match = re.match(r'\[\[IMG:\s*(.*?)\]\]', line)
        if img_match:
            label_key = img_match.group(1).strip().replace(" ", "")
            if label_key in image_data_dict:
                info = image_data_dict[label_key]
                try:
                    im = PDFImage(info['path'])
                    # サイズ調整
                    max_w, max_h = available_width, 10*cm 
                    img_w, img_h = im.imageWidth, im.imageHeight
                    aspect = img_h / float(img_w)
                    
                    if img_w > max_w:
                        img_w = max_w
                        img_h = img_w * aspect
                    if img_h > max_h:
                        img_h = max_h
                        img_w = img_h / aspect
                        
                    im.drawWidth = img_w
                    im.drawHeight = img_h
                    
                    story.append(KeepTogether([
                        Spacer(1, 0.2*cm),
                        im,
                        Spacer(1, 0.1*cm),
                        Paragraph(f"▲ {info['caption']}", style_caption),
                        Spacer(1, 0.5*cm)
                    ]))
                except Exception as e:
                    print(f"画像描画エラー: {e}")
            continue

        # テキスト処理
        if line.startswith('# '):
            if len(story) > 5: story.append(PageBreak())
            text = line.replace('# ', '').strip()
            d_h1 = Drawing(available_width, 1)
            d_h1.add(Line(0, 0, available_width, 0, strokeColor=colors.grey, strokeWidth=1))
            story.append(KeepTogether([
                Spacer(1, 0.5*cm),
                Paragraph(text, style_h1),
                d_h1,
                Spacer(1, 0.3*cm)
            ]))
        elif line.startswith('## '):
            story.append(Paragraph(line.replace('## ', '').strip(), style_h2))
        elif line.startswith('### '):
            story.append(Paragraph(line.replace('### ', '').strip(), style_h3))
        elif line.startswith('#### '):
            story.append(Paragraph(f"<b>{line.replace('#### ', '').strip()}</b>", style_body))
        elif line.startswith('- ') or line.startswith('* '):
            story.append(Paragraph(f"• {format_inline_bold(line[2:])}", style_bullet))
        else:
            story.append(Paragraph(format_inline_bold(line), style_body))

    if in_table and table_buffer:
        t = create_paragraph_table(table_buffer, styles, available_width)
        if t: story.append(t)

    try:
        doc.build(story)
        print(f"💾 PDF保存完了: {output_path}")
        return True
    except Exception as e:
        print(f"❌ PDF保存エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==========================================
# メイン処理
# ==========================================
def main():
    input_files = list(Path(DIRS['input']).glob('*.pdf'))
    if not input_files:
        print("ファイルが見つかりません。data/input_PDF を確認してください。")
        return

    for pdf_file in input_files:
        try:
            text, img_list, img_data = extract_content_from_pdf(str(pdf_file))
            summary = generate_summary(text, img_list)
            
            if save_to_pdf(summary, img_data, pdf_file.name):
                new_path = Path(DIRS['processed']) / pdf_file.name
                if new_path.exists(): os.remove(new_path)
                shutil.move(str(pdf_file), str(new_path))
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()