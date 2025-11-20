# ts_AI
PDFファイルを要約出来るchat AI<br>

依頼をダウンロード
```
pip install -r requirements.txt
```
<br>

.envで自分のGemini API を入れて
```
GOOGLE_API_KEY="<YOUR_GOOGLE_API_KEY>"
```
PDF.pyはPDFを要約出来るAIです、使う方は以下の通りです。<br>
・ input_PDF = あなたは要約したいPDFを入れて<br>
・ output_PDF = 要約PDF<br>
・ PDF_document = input_PDF => output_PDF 終わったら元々のPDFを保存する所<br>

