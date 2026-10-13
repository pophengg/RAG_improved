# เปรียบเทียบข้อ (1) และ (2)

## ส่วนที่คงเดิมเพื่อให้เปรียบเทียบอย่างเป็นธรรม

ทั้งสองระบบใช้ OpenSearch, dense embedding **BAAI/bge-m3**, BM25 และ Qwen **qwen/qwen-2.5-72b-instruct** ตัวสร้างคำตอบและตัวเรียก API เป็นโค้ดเดียวกับต้นฉบับ รุ่นปรับปรุงไม่ได้ override สอง method นี้ การทดสอบตรวจความเท่ากันของโครงสร้างโค้ดด้วย AST

Baseline แก้เพียง import LangChain ให้ใช้ `langchain_core.documents`, อ่าน UTF-8 ชัดเจน และเอา `chunk_size=256` ออกจาก MarkdownNodeParser เพราะ parser แบบ Markdown แบ่งตามหัวข้อ ไม่ใช่ตัวควบคุมความยาว token ขนาด 256 ไม่ได้เป็นข้อรับประกันของโค้ดเดิม แยกชื่อดัชนีผ่าน wrapper ใน `run.py` ไม่เปลี่ยนกลไก retrieval ของ baseline

## ความแตกต่างและเหตุผล

| ส่วน | (1) baseline | (2) improved | ผลที่คาดหวังและเหตุผล |
|---|---|---|---|
| Chunking | MarkdownNodeParser แบ่งตามหัวข้อ | แบ่งตามหัวข้อ แล้วจำกัด 384 token พร้อม overlap 64 โดยใช้ tokenizer ของ BGE-M3 | จำกัด chunk ยาวและรักษาข้อมูลตรงรอยต่อ ใช้ token แทนช่องว่างสำหรับภาษาไทย เก็บข้อความจาก character offsets เพื่อไม่ให้ decode เปลี่ยนข้อความ |
| Contextualization | `full_doc_content = doc.page_content` จึงส่ง chunk เป็นเอกสารเต็มด้วย | ส่งเอกสารเต็มจริง เก็บ source และ heading | Context มีโอกาสระบุหัวข้อและความหมายที่อยู่นอก chunk ได้ถูกต้องขึ้น แต่เอกสารเต็มยาวเพิ่มค่า API |
| Embedding | encode ทีละ chunk | batch 16 และ normalize พร้อมตรวจความยาว contextual chunk | ลด overhead โดยยังใช้ BGE-M3 เหมือนเดิม normalization ไม่ได้แปลว่าจะเพิ่ม accuracy เมื่อใช้ cosine อยู่แล้ว |
| BM25 | standard analyzer และสอง field เท่ากัน | thai analyzer; content และ heading boost 2 | Thai analyzer แบ่งคำภาษาไทย ทำให้ keyword matching มีโอกาสดีขึ้น การ boost ต้องปรับจาก validation set |
| Candidate search | แต่ละช่องค้นหา `2*k` | แต่ละช่องค้นหา `max(30,6*k)` | ให้ reranker มี candidate มากขึ้น อาจเพิ่ม recall ก่อน rerank แต่ต้นทุนสูงขึ้น |
| Scoring | RRF แบบเริ่มอันดับ 0 | weighted RRF เริ่มอันดับ 1 และรวม metadata จากสองช่อง | ใช้สูตร sum(w/(60+rank)) ตามอันดับเริ่ม 1 ไม่เทียบ raw BM25 กับ cosine และไม่สูญเสีย context เมื่อตัวแรกมาจาก dense |
| Reranking | ไม่มี | BGE-reranker-v2-m3 cross-encoder หลัง fusion และตัดข้อความซ้ำ | ประเมินคำถามกับข้อความร่วมกัน อาจแยก candidate ที่คำคล้ายแต่ตอบไม่ตรงได้ดีขึ้น |
| Context budget | จำนวนคำคูณ 1.3 | จำกัด UTF-8 bytes แบบ conservative | ภาษาไทยไม่มีช่องว่างทุกคำ baseline อาจประเมินต่ำเกินจริง รุ่นใหม่จำกัดได้แน่นอนเป็น bytes แต่ไม่ใช่ tokenizer Qwen ที่แม่นยำและอาจตัดข้อมูลมากเกินจำเป็น |
| Index integrity | ID `doc_i`, bulk error แค่พิมพ์แจ้ง | stable ID จาก source+ตำแหน่ง+ข้อความ, bulk error ทำให้หยุด, cache context | ตรวจ failure ได้ชัด รันซ้ำได้สม่ำเสมอ CLI rebuild เพื่อไม่เหลือ chunk เก่าหลังแก้เอกสาร |

## ดีขึ้นอย่างไร: แยกสิ่งที่ตรวจได้กับสมมติฐาน

สิ่งที่ตรวจจากโค้ดและ unit tests ได้คือการจำกัด chunk, overlap, stable ID, การรวม metadata, สูตร RRF, metrics และการคงตัวสร้างคำตอบเดิม ส่วนคุณภาพ retrieval ที่ดีขึ้นเป็น **สมมติฐานที่ต้องวัด** ไม่สามารถสรุปว่าเพิ่มขึ้นเป็นเปอร์เซ็นต์จากการออกแบบเพียงอย่างเดียว

รันทั้งสองรุ่นกับ corpus เดียวกันและคำถามเดียวกัน โดยไม่เรียก generation แล้ววัด document-level Recall@k, MRR@k, nDCG@k และเวลาเฉลี่ย สคริปต์ deduplicate source ในรายการ chunk ที่ได้ก่อนคำนวณ แต่ไม่ได้ค้นเพิ่มเพื่อเติมให้ครบ k เอกสาร จึงสะท้อนว่าการได้ chunk ซ้ำจากเอกสารเดียวลดความครอบคลุม ทั้ง labels เป็น binary relevant/not relevant

ใช้ `python evaluate.py --k 3` หลัง index ทั้งสองรุ่น ใช้ `python evaluate.py --k 3 --no-rerank --output results/retrieval-no-rerank.json` เพื่อตรวจผลของ reranker เทียบกับ improved แบบไม่ rerank การรันนี้ไม่แยกผลทุกการปรับปรุง หากต้องการพิสูจน์แต่ละส่วนต้องทำ ablation เพิ่มทีละส่วน

ข้อแลกเปลี่ยนคือโมเดล reranker เพิ่ม RAM และ latency, candidate pool ใหญ่ขึ้นเพิ่มเวลาค้นหา, overlap เพิ่มจำนวน chunk และค่า contextualization, context ที่ LLM สร้างอาจผิดได้ และทั้งสองรุ่นยังสืบทอดพฤติกรรมตอบโดยไม่มีหลักฐานเมื่อค้นไม่พบ ควรประเมินความถูกต้องและการอ้างอิงของคำตอบแยกจาก retrieval ก่อนใช้งานจริง

## เอกสารอ้างอิง

- [โค้ดต้นฉบับ](https://github.com/aekanun2020/2025-authenticRAG/blob/main/authenticRAG.py)
- [OpenSearch Thai analyzer](https://docs.opensearch.org/latest/analyzers/language-analyzers/thai/)
- [BAAI bge-reranker-v2-m3: คู่ query/passage และ multilingual reranking](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [SentenceTransformer encode: batching และ normalize_embeddings](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
