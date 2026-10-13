# Authentic RAG: ต้นฉบับและรุ่นปรับปรุง

โครงการสำหรับโจทย์ข้อ (1)–(4) โดยพัฒนาจาก [2025-authenticRAG ของ aekanun2020](https://github.com/aekanun2020/2025-authenticRAG) ไม่เปลี่ยน LLM สำหรับคำตอบสุดท้าย: **qwen/qwen-2.5-72b-instruct** ผ่าน OpenRouter ทั้งสองรุ่นใช้ prompt, temperature 0.6 และ max_tokens 1000 เดียวกันใน `generate_response` ที่สืบทอดจากต้นฉบับ

## ไฟล์ตามโจทย์

- (1) `upstream/` เก็บสามไฟล์จากลิงก์เดิมโดยไม่แก้ไข และ `baseline.py` เป็นรุ่นที่แก้ compatibility เพื่อเรียกใช้งานผ่าน `run.py`
- (2) `improved.py` และ `retrieval.py` พัฒนากระบวนการ retrieval โดยสืบทอดตัวสร้างคำตอบจาก baseline
- (3) [COMPARISON.md](COMPARISON.md) อธิบายความแตกต่าง เหตุผล ข้อแลกเปลี่ยน และวิธีตรวจคุณภาพ
- (4) โฟลเดอร์นี้พร้อมนำขึ้น GitHub มี `.gitignore` กัน API key และ cache พร้อมชุดข้อมูลตัวอย่างและการทดสอบ

## ติดตั้งและรัน (PowerShell, Python 3.10+)

ต้องมี Docker Desktop, Python และ OpenRouter API key โมเดล embedding/reranker ดาวน์โหลดจาก Hugging Face ในการรันครั้งแรกและใช้หน่วยความจำหลาย GB ไม่ต้องติดตั้ง Ollama เพราะโค้ดเรียก SentenceTransformer โดยตรง

```powershell
git clone https://github.com/pophengg/RAG_improved.git
cd RAG_improved
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d
# รอจนคำสั่งนี้ตอบ cluster status ก่อนทำ index
Invoke-RestMethod http://localhost:9200/_cluster/health
$env:OPENROUTER_API_KEY = 'your-key'
.\.venv\Scripts\python.exe run.py index --mode baseline
.\.venv\Scripts\python.exe run.py index --mode improved
.\.venv\Scripts\python.exe run.py search --mode improved --question 'RRF รวมผลค้นหาอย่างไร'
.\.venv\Scripts\python.exe run.py ask --mode baseline --question 'RRF รวมผลค้นหาอย่างไร'
.\.venv\Scripts\python.exe run.py ask --mode improved --question 'RRF รวมผลค้นหาอย่างไร'
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe -m unittest -v
```

`index` สร้างดัชนีใหม่เฉพาะรุ่นที่เลือกและลบข้อมูลเดิมในดัชนีนั้น ใช้ `baseline-*` และ `improved-*` แยกกัน ไม่แตะดัชนี `anthropic-*` ของต้นฉบับ ใส่เอกสาร Markdown UTF-8 ใน `corpus_input/` ก่อน index และหลีกเลี่ยงเอกสารเต็มที่เกินบริบทของ Qwen ถ้า API ล้มเหลวรุ่นปรับปรุงจะหยุด indexing ให้แก้เหตุและรันใหม่

`search` และ `evaluate.py` ไม่เรียก LLM สร้างคำตอบ แต่ constructor ที่สืบทอดยังต้องมีตัวแปร key ส่วน `index` ใช้ API เพื่อสร้าง context ต่อ chunk ซึ่งมีค่าใช้จ่าย รุ่นปรับปรุงมี cache ลดการเรียกซ้ำ และไฟล์ cache อาจมีข้อมูลเอกสาร จึงไม่ส่งขึ้น GitHub

## สถานะการตรวจสอบ

ดู [VALIDATION.md](VALIDATION.md) สำหรับสิ่งที่ตรวจแล้ว ผลทดสอบไม่ใช่หลักฐานว่ารุ่นปรับปรุงมี retrieval ดีกว่า ต้องรัน evaluation กับบริการจริง ผลจะอยู่ใน `results/retrieval.json` โดยยังไม่รวมใน repository อัตโนมัติ

ชุด `corpus_input/` เป็นข้อความตัวอย่างที่เขียนสำหรับโครงการนี้ ไม่ใช่ corpus โรคตามตัวอย่างต้นฉบับ ชุดคำถาม 8 ข้อใช้ตรวจการทำงานเบื้องต้นเท่านั้น ก่อนรายงานผลจริงให้เพิ่มเอกสารและ relevance labels ที่ผู้ประเมินตรวจแล้ว และแบ่งชุดปรับพารามิเตอร์กับชุดทดสอบ

## การส่งขึ้น GitHub

Repository สำหรับส่งงาน: https://github.com/pophengg/RAG_improved

หากต้องการส่งโครงการนี้ไป repository อื่น ให้สร้าง repository ปลายทางที่มีสิทธิ์เขียน แล้วรันจากโฟลเดอร์สำเนาโครงการที่ยังไม่มี Git:

```powershell
git init
git add .
git commit -m "Add baseline and improved Authentic RAG with comparison and evaluation"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

เก็บ attribution ของผู้เขียนต้นฉบับไว้ใน `upstream/` และ [SOURCE.md](SOURCE.md) โครงการนี้ไม่ได้ประกาศ license ใหม่ครอบโค้ดต้นฉบับ ตรวจเงื่อนไขของต้นฉบับก่อนเผยแพร่หรือใช้ต่อ
