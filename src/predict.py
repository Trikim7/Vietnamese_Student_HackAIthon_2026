import os
import sys
import json
import ast
import time
import glob
import re
from pathlib import Path
from collections import Counter
import pandas as pd
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
from llama_cpp import Llama

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

LET = "ABCDEFGHIJK"
STOP_WORDS = set(["của", "là", "và", "các", "những", "thì", "mà", "theo", "quy", "định",
                  "nào", "sau", "đây", "trong", "có", "không", "được", "với", "cho",
                  "câu", "hỏi", "hãy", "chọn", "đáp", "án", "một", "này", "đó",
                  "như", "về", "từ", "đến", "khi", "tại", "trên", "dưới", "nếu",
                  "vì", "do", "bởi", "cũng", "đều", "rằng", "thế", "rất", "hay"])

PROFILES = {
    "CALCULATION": "Bạn là AI Toán học. HƯỚNG DẪN TƯ DUY: 1. Ghi công thức. 2. Tính toán cẩn thận từng bước. 3. Đối chiếu và chốt đáp án.",
    "VIETNAM": "Bạn là AI Pháp lý Việt Nam. Rút trích từ khóa và đối chiếu với điều luật để chọn đáp án.",
    "HISTORY_LAW": "Bạn là AI Lịch sử Thế giới. So sánh các mốc thời gian và sự kiện.",
    "GENERAL": "Bạn là AI Logic. Dùng phương pháp loại trừ, đánh giá đúng/sai từng lựa chọn trong tối đa 3 câu."
}

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Qwen3.5-4B.Q8_0.gguf"
if not MODEL_PATH.exists():
    MODEL_PATH = Path("/agent_src/src/models/Qwen3.5-4B.Q8_0.gguf")

KB_PATH = BASE_DIR / "vietnam_kb.jsonl"
if not KB_PATH.exists():
    KB_PATH = Path("/agent_src/src/vietnam_kb.jsonl")

CODE_DIR = Path("/code")
APP_DATA_DIR = Path("/app/data")
DATA_DIR = Path("/data")

def load_questions():
    search_dirs = [CODE_DIR, APP_DATA_DIR, DATA_DIR, Path("./code"), Path("./data"), Path(".")]
    found_file = None
    
    for d in search_dirs:
        if d.exists():
            for name in ["private_test.json", "public_test.json", "private_test.csv", "public_test.csv"]:
                target = d / name
                if target.exists():
                    found_file = target
                    break
        if found_file: break

    if not found_file:
        for d in search_dirs:
            if d.exists():
                files = list(d.rglob("*test*.json")) + list(d.rglob("*test*.csv")) + list(d.glob("*.json"))
                files = [f for f in files if not f.name.startswith(".")]
                if files:
                    found_file = files[0]
                    break

    if not found_file:
        raise FileNotFoundError("Khong tim thay file du lieu de thi (private_test.json)")

    print(f"Nap du lieu de thi tu: {found_file}")
    ext = found_file.suffix.lower()

    if ext == ".json":
        with open(found_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif ext in [".jsonl", ".txt"]:
        qs = []
        with open(found_file, 'r', encoding='utf-8') as f:
            for l in f:
                if l.strip(): qs.append(json.loads(l))
        return qs
    elif ext == ".csv":
        df = pd.read_csv(found_file, encoding='utf-8')
        qs = []
        for idx, row in df.iterrows():
            qid = row.get("qid", row.get("id", row.get("ID", idx)))
            question = row.get("question", row.get("prompt", row.get("Câu hỏi", "")))
            choices = row.get("choices", row.get("options", row.get("Các lựa chọn", None)))

            if choices is not None and isinstance(choices, str):
                try:
                    choices = ast.literal_eval(choices)
                except Exception:
                    try:
                        choices = json.loads(choices)
                    except Exception:
                        choices = [s.strip() for s in choices.split(",")]
            elif choices is None:
                choice_list = []
                for col in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "choice_A", "choice_B", "choice_C", "choice_D"]:
                    if col in row and pd.notna(row[col]):
                        choice_list.append(str(row[col]))
                choices = choice_list

            qs.append({"qid": qid, "question": str(question), "choices": list(choices)})
        return qs
    else:
        raise ValueError(f"Dinh dang file {ext} khong duoc ho tro")

def create_ngrams(text, n=2):
    words = [w for w in text.lower().split() if w not in STOP_WORDS and len(w) > 1]
    ngrams = words[:]
    for i in range(len(words) - 1):
        ngrams.append("_".join(words[i:i+2]))
    return ngrams

corpus = []
bm25 = None
CHUNK_SIZE = 150
STEP_SIZE = 100

if KB_PATH.exists():
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            title = doc.get('title', '')
            content = doc.get('content', '')
            words = content.split()

            if len(words) <= CHUNK_SIZE:
                corpus.append(f"[{title}]\n{content}")
            else:
                for i in range(0, len(words), STEP_SIZE):
                    chunk_words = words[i:i + CHUNK_SIZE]
                    if len(chunk_words) < 30: break
                    corpus.append(f"[{title}]\n{' '.join(chunk_words)}")

    bm25 = BM25Okapi([create_ngrams(doc, n=2) for doc in corpus])
    print(f"Nap xong {len(corpus)} phan doan tri thuc BM25")
else:
    print(f"Khong tim thay file KB tai {KB_PATH}")

def offline_search(question, choices):
    if not bm25 or not corpus: return None
    noise_kws = ["tất cả", "cả a", "cả b", "cả c", "đều", "từ chối", "không có", "chưa rõ", "không thể", "đáp án", "phương án"]
    clean_choices = []
    for c in choices[:4]:
        c_str = str(c).lower()
        if not any(kw in c_str for kw in noise_kws):
            clean_choices.append(str(c))
            
    choice_text = " ".join(clean_choices)  
    raw_query = re.sub(r"[^\w\s]", " ", (question + " " + choice_text).lower())
    clean_words = create_ngrams(raw_query, n=2)
    if not clean_words: clean_words = question.lower().split()

    top_10_docs = bm25.get_top_n(clean_words, corpus, n=10)
    scored_docs = []
    choice_strings = [str(c).lower().strip() for c in choices]

    for doc in top_10_docs:
        doc_lower = doc.lower()
        score = 0
        for c_str in choice_strings:
            if len(c_str) > 3 and c_str in doc_lower:
                score += 10
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    best_docs = [doc for score, doc in scored_docs[:3]]
    if not best_docs: return None

    combined_context = "\n---\n".join(best_docs)
    final_words = combined_context.split()
    if len(final_words) > 600:
        combined_context = " ".join(final_words[:600]) + "...\n[Đã cắt bớt]"
    return combined_context

def has_embedded_context(question):
    markers = ["Đoạn thông tin", "Tiêu đề:", "Nội dung:", "Đoạn văn:", "Dựa vào đoạn", "Cho đoạn", "Đọc đoạn", "Xét đoạn"]
    return any(m in question for m in markers)

def check_ethics_override(question, choices):
    q_str = str(question).lower()
    toxic_action_kws = ["trái quy định", "đình chỉ", "tránh việc cung cấp", "phát tán", "làm suy yếu", "chống lại", "xúc phạm", "làm giả", "tham nhũng", "phá hoại", "kích động", "vi phạm", "đánh cắp", "lừa đảo", "trốn", "lạm dụng", "lách luật", "hack", "xâm nhập"]
    if any(kw in q_str for kw in toxic_action_kws):
        refusal_kws = ["tôi không thể", "từ chối", "không thể trả lời", "vi phạm pháp luật", "không cung cấp", "không được phép", "xin lỗi"]
        if choices:
            for i, c in enumerate(choices):
                if any(kw in str(c).lower() for kw in refusal_kws): return LET[i]
    return None

def route_question(question, choices):
    text_lower = question.lower()
    scores = {"CALCULATION": 0, "VIETNAM": 0, "HISTORY_LAW": 0, "GENERAL": 0}
    
    hard_math_kws = ["đạo hàm", "tích phân", "ma trận", "xác suất", "gdp", "lãi suất", "phương trình", "đồ thị", "hàm số", "tiệm cận", "logarit", "thể tích", "diện tích", "chu vi", "cấp số cộng", "cấp số nhân", "gia tốc", "động năng", "thế năng", "bước sóng"]
    scores["CALCULATION"] += sum(3 for kw in hard_math_kws if kw in text_lower)
    if re.search(r"\btính\b(?!\s+(chất|cách|từ|mạng|nhân|dục|kế))", text_lower): scores["CALCULATION"] += 1
    if re.search(r"\bgiá trị\b(?!\s+(nhân đạo|lịch sử|văn hóa|nghệ thuật|đạo đức|tinh thần))", text_lower): scores["CALCULATION"] += 1
    if re.search(r"∫|∑|lim|log|sin|cos|tan|√|\d+\s*[\+\-\/=]\s*\d+|[<>]\s*\d+", text_lower): scores["CALCULATION"] += 2
        
    his_kws = ["chiến tranh", "triều đại", "hiệp ước", "hiệp định", "thế kỷ", "cổ đại", "phong kiến", "lịch sử", "khởi nghĩa", "kháng chiến", "thực dân", "đế quốc", "vua", "hoàng đế", "công ước", "pháp luật"]
    scores["HISTORY_LAW"] += sum(2 for kw in his_kws if kw in text_lower)
    if re.search(r"\bnăm \d{3,4}\b", text_lower): scores["HISTORY_LAW"] += 3

    vn_kws = ["việt nam", "đảng", "hồ chí minh", "hiến pháp", "nghị định", "thông tư", "tư tưởng", "mác - lênin", "chính trị", "cách mạng", "nhà nước", "chính phủ", "quốc hội", "ban chấp hành", "trung ương", "điều luật", 
              "văn học", "tục ngữ", "thành ngữ", "ca dao", "thơ", "truyện", "tác phẩm", "nhà văn", "nhà thơ", "địa lý", "tỉnh", "thành phố", "đồng bằng"]
    scores["VIETNAM"] += sum(2 for kw in vn_kws if kw in text_lower)
    
    if choices:
        num_numeric = sum(1 for c in choices if re.match(r"^[\d\.\,\-\+\/\\\spi\%]+$", str(c).strip()))
        if num_numeric >= len(choices) / 2:
            num_years = sum(1 for c in choices if re.match(r"^(1|2)\d{3}$", str(c).strip()))
            if num_years == num_numeric and ("năm" in text_lower or "thế kỷ" in text_lower): scores["HISTORY_LAW"] += 5
            else: scores["CALCULATION"] += 4

    cat = max(scores, key=scores.get)
    return "GENERAL" if scores[cat] == 0 else cat

def format_choices(ch): return "\n".join(f"{LET[i]}. {c}" for i, c in enumerate(ch))

def build_prompt(x, category, context=None):
    valid_letters = "{" + ", ".join(list(LET[:len(x['choices'])])) + "}"
    if has_embedded_context(x['question']):
        ctx_str = "LỆNH: Đọc đoạn thông tin có sẵn trong đề bài để trả lời. Không cần dùng kiến thức ngoài.\n"
    elif context:
        ctx_str = f"[TÀI LIỆU SỰ THẬT]:\n{context}\n\nLỆNH QUAN TRỌNG: Nếu tài liệu chứa đáp án, hãy dùng làm gốc. NẾU TÀI LIỆU KHÔNG CHỨA THÔNG TIN, BẠN BẮT BUỘC PHẢI DÙNG KIẾN THỨC CỦA CHÍNH MÌNH ĐỂ TRẢ LỜI ĐÚNG NHẤT.\n"
    else:
        ctx_str = ""
        
    body = f"{ctx_str}\nCâu hỏi: {x['question']}\n\nCác lựa chọn:\n{format_choices(x['choices'])}\n\n"
    instruction = f"""{PROFILES.get(category, PROFILES["GENERAL"])}\n\nLỆNH BẮT BUỘC TUÂN THỦ:\n1. Trả lời hoàn toàn bằng Tiếng Việt.\n2. Suy nghĩ trong thẻ <think> BẮT BUỘC ngắn gọn, rõ ràng.\n3. Ngay sau khi đóng thẻ </think>, BẮT BUỘC ghi duy nhất 1 dòng: ĐÁP ÁN: X (Với X thuộc {valid_letters}).\n\n<think>\n"""
    return body + instruction

def parse_marker(text, n):
    valid = set(list(LET[:n]))
    clean_text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL)
    patterns = [r"(?:ĐÁP ÁN|Đáp án|Lựa chọn|Chọn|Đáp án là)[\s:=>\-\[\]]*([A-K])\b", r"Đáp án[:\s]*([A-K])\b", r"Chọn[:\s]*([A-K])\b"]
    for pat in patterns:
        matches = list(re.finditer(pat, clean_text, re.I))
        if matches:
            last_match = matches[-1].group(1).upper()
            if last_match in valid: return last_match
    for pat in patterns:
        matches = list(re.finditer(pat, text or "", re.I))
        if matches:
            last_match = matches[-1].group(1).upper()
            if last_match in valid: return last_match
    return None

def fuzzy_math_matcher(raw_thought, choices, valid_letters):
    numbers_in_thought = re.findall(r"[-+]?\d*\.\d+|\d+", raw_thought)
    if not numbers_in_thought: return None
    final_number_str = numbers_in_thought[-1].replace(",", "")
    for i, c in enumerate(choices):
        if i >= len(valid_letters): break
        c_str = str(c).replace(",", "")
        c_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", c_str)
        if final_number_str in c_numbers: return valid_letters[i]
        try:
            f_val = float(final_number_str)
            for cn in c_numbers:
                c_val = float(cn)
                if c_val != 0 and abs(f_val - c_val) / abs(c_val) < 0.02: return valid_letters[i]
        except ValueError: continue
    return None

def smart_fallback(question, choices, context_used):
    combined = (question + " " + (context_used or "")).lower()
    best_idx, best_score = 0, -1
    for i, c in enumerate(choices):
        c_words = [w for w in str(c).lower().split() if len(w) > 2 and w not in STOP_WORDS]
        score = sum(1 for w in c_words if w in combined)
        if score > best_score:
            best_score, best_idx = score, i
    return LET[best_idx]

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay model GGUF tai {MODEL_PATH}")

    print("Khoi tao LLM engine...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False
    )

    def chat(messages, max_tokens, temp=0.1):
        out = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temp, top_p=0.9)
        return out["choices"][0]["message"]["content"].strip()

    def answer_one(x):
        n = len(x["choices"])
        valid_letters = list(LET[:n])
        
        ethics_letter = check_ethics_override(x["question"], x["choices"])
        if ethics_letter: return ethics_letter

        category = route_question(x["question"], x["choices"])
        context = None
        if not has_embedded_context(x["question"]) and category in ["VIETNAM", "HISTORY_LAW"]:
            context = offline_search(x["question"], x["choices"]) 
            
        if category == "CALCULATION": max_t = 1024 
        elif category == "ETHICS": max_t = 300
        else: max_t = 800
        
        prompt = build_prompt(x, category, context)
        messages_pass_1 = [{"role": "system", "content": "Bạn là AI thông minh."}, {"role": "user", "content": prompt}]
        raw1 = chat(messages_pass_1, max_tokens=max_t, temp=0.2)
        letter = parse_marker(raw1, n)
        
        final_raw = raw1
        context_for_fallback = context
        
        if letter is None and category == "CALCULATION":
            letter = fuzzy_math_matcher(raw1, x['choices'], valid_letters)
            if letter: return letter

        if letter is None:
            partial_thought = raw1[-1500:] if len(raw1) > 1500 else raw1
            messages_pass_2 = [
                {"role": "system", "content": "Bạn là AI thông minh."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": partial_thought},
                {"role": "user", "content": f"BẮT BUỘC chỉ trả lời đúng MỘT dòng: 'ĐÁP ÁN: X' (với X thuộc {valid_letters}). KHÔNG dùng thẻ <think>."}
            ]
            raw2 = chat(messages_pass_2, max_tokens=150, temp=0.0)
            letter = parse_marker(raw2, n)
            if letter:
                final_raw = raw1 + "\n\n" + raw2
            else:
                final_raw = raw1 + "\n\n" + raw2
                
        if letter is None:
            safe_tail = final_raw
            for pat in [r"(?:đáp án|chọn|là|kết quả)[\s:=>\-\[\]]*([A-K])\b"]:
                for m in reversed(list(re.finditer(pat, safe_tail[-150:].lower()))):
                    found = m.group(1).upper()
                    if found in set(valid_letters): 
                        letter = found
                        break
            if letter is None:
                letter = smart_fallback(x["question"], x["choices"], context_for_fallback)
                
        return letter

    questions = load_questions()
    print(f"Bat dau giai {len(questions)} cau hoi...")

    BATCH_SIZE = 50
    t0 = time.time()
    
    pred_rows = []
    time_rows = []

    for idx, x in enumerate(tqdm(questions, desc="Solving")):
        if hasattr(llm, '_ctx') and hasattr(llm._ctx, 'kv_cache_clear'):
            llm._ctx.kv_cache_clear()
            
        t_q_start = time.time()
        letter = answer_one(x)
        t_q_elapsed = time.time() - t_q_start
        
        qid = x.get("qid", x.get("id", x.get("ID", idx)))
        pred_rows.append({"qid": qid, "answer": letter})
        # Đúng mẫu quy định tại mục 2.3 bảng ví dụ: qid,answer,time
        time_rows.append({"qid": qid, "answer": letter, "time": round(t_q_elapsed, 4)})

        if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == len(questions):
            print(f"Hoan thanh {idx + 1}/{len(questions)} cau")

    df_pred = pd.DataFrame(pred_rows)
    df_time = pd.DataFrame(time_rows)

    target_dirs = [CODE_DIR, APP_DATA_DIR, Path("./code"), Path("./data"), Path(".")]
    for d in target_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            df_pred.to_csv(d / "submission.csv", index=False)
            df_time.to_csv(d / "submission_time.csv", index=False)
        except Exception:
            pass

    dt = time.time() - t0
    print(f"Hoan thanh trong {dt:.0f}s. Da xuat ra submission.csv va submission_time.csv")

if __name__ == "__main__":
    main()
