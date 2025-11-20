# ts_AI
PDFファイルを要約出来るchat AI

依頼をダウンロード
```
pip install -r requirements.txt
```

.envで自分のGemini API を入れて
```
GOOGLE_API_KEY="<YOUR_GOOGLE_API_KEY>"
```
PDF.pyはPDFを要約出来るAIです、使う方は以下の通りです。
・ input_PDF = あなたは要約したいPDFを入れて
・ output_PDF = 要約PDF
・ PDF_document = input_PDF => output_PDF 終わったら元々のPDFを保存する所

