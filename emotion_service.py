# emotion_service.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from konlpy.tag import Okt
import re, requests, uuid

# 감정 분석 모델 로딩
emotion_model = "dlckdfuf141/korean-emotion-kluebert-v2"
tokenizer = AutoTokenizer.from_pretrained(emotion_model)
model = AutoModelForSequenceClassification.from_pretrained(emotion_model)
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
okt = Okt()

# Clova API 설정
CLOVA_API_KEY = "nv-8bde2bcde8f2486c93f209120119a3728S16"
API_URL = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"
HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "Authorization": f"Bearer {CLOVA_API_KEY}",
    "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
    "Accept": "application/json"
}

chat_history = [
    {"role": "system", "content": "당신은 투자 전문가이자 감정 케어 상담사입니다. 감정을 분석하고 현실적인 재정/심리 조언을 따뜻하게 전달하세요."}
]

emotion_care = {
    "공포": "두려움을 느끼는 것은 자연스러운 일입니다.",
    "분노": "화가 나는 상황에서 감정을 표현하는 건 중요한 일이에요.",
    "슬픔": "마음이 무거운 시간을 보내고 있군요.",
    "중립": "지금은 감정의 파동이 크지 않은 상태로 보이네요.",
    "행복": "행복한 순간을 함께 나눠주셔서 감사합니다!",
    "혐오": "불쾌한 감정을 느낀 일이 있었군요.",
    "놀람": "놀라운 일이 있었군요!",
}

def preprocess_input(text):
    return re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", "", text).strip()

def extract_keywords(text):
    goal_candidates = ["내집", "집", "노후", "차", "결혼", "목표", "자녀", "양육", "유학"]
    goal = next((word for word in goal_candidates if word in text), None)
    purpose_phrases = re.findall(r"([가-힣\s]{1,15})(?:를|을)?\s*(?:위해|위해서|하려고)", text)
    if purpose_phrases:
        goal = goal or purpose_phrases[0].strip()
    period_match = re.search(r"(\d+\s*년|\d+\s*개월)", text)
    money_match = re.search(r"(\d+(\.\d+)?\s*(억|천만원|만원|천원|원))", text)
    return {
        "목표": goal,
        "기간": period_match.group(0).strip() if period_match else None,
        "금액": money_match.group(0).strip() if money_match else None
    }

def detect_emotion(text):
    result = emotion_classifier(text, top_k=1)[0]
    raw_label = result['label']

    # ✅ 문자열 또는 정수 모두 처리
    if isinstance(raw_label, str) and raw_label.startswith("LABEL_"):
        label_id = int(raw_label.replace("LABEL_", ""))
    elif isinstance(raw_label, int):
        label_id = raw_label
    else:
        raise ValueError(f"알 수 없는 라벨 형식: {raw_label}")

    label_map = {
        0: "공포", 1: "놀람", 2: "분노", 3: "슬픔", 4: "중립", 5: "행복", 6: "혐오"
    }
    return [label_map.get(label_id, "알 수 없음")]

def build_prompt(keywords, emotions):
    goal = keywords.get("목표")
    period = keywords.get("기간")
    money = keywords.get("금액")
    primary_emotion = emotions[0] if emotions else None
    emotion_intro = emotion_care.get(primary_emotion, "")
    emotion_str = ", ".join(emotions)

    return f"""
[사용자 감정 분석]
감정 상태: {emotion_str}
공감: {emotion_intro}

[투자 목표 분석]
- 목표: {goal or '미입력'}
- 기간: {period or '미입력'}
- 금액: {money or '미입력'}

[요청]
감정과 목표를 고려한 따뜻한 재정 조언과 전략을 제안해주세요.
"""

def call_clova_api(prompt):
    chat_history.append({"role": "user", "content": prompt})
    body = {
        "messages": chat_history,
        "topP": 0.8,
        "temperature": 0.7,
        "maxTokens": 500
    }
    res = requests.post(API_URL, headers=HEADERS, json=body)
    return res.json()['result']['message']['content'] if res.ok else f"❌ Clova 오류: {res.status_code}"

def clova_financial_chatbot(user_text):
    cleaned = preprocess_input(user_text)
    keywords = extract_keywords(cleaned)
    emotions = detect_emotion(cleaned)
    prompt = build_prompt(keywords, emotions)
    clova_response = call_clova_api(prompt)
    return {
        "감정": emotions,
        "키워드": keywords,
        "프롬프트": prompt,
        "클로바 응답": clova_response
    }
