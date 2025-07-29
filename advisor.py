# advisor.py
import re, uuid, json, requests
from konlpy.tag import Okt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# 감정 분석기 설정
emotion_model = "dlckdfuf141/korean-emotion-kluebert-v2"
tokenizer = AutoTokenizer.from_pretrained(emotion_model)
model = AutoModelForSequenceClassification.from_pretrained(emotion_model)
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
okt = Okt()

# 감정 공감 멘트
emotion_care = {
    "공포": "두려움을 느끼는 것은 자연스러운 일입니다.",
    "분노": "화가 나는 상황에서 감정을 표현하는 건 중요한 일이에요.",
    "슬픔": "마음이 무거운 시간을 보내고 있군요.",
    "중립": "지금은 감정의 파동이 크지 않은 상태로 보이네요.",
    "행복": "행복한 순간을 함께 나눠주셔서 감사합니다!",
    "혐오": "불쾌한 감정을 느낀 일이 있었군요.",
    "놀람": "놀라운 일이 있었군요!",
}

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
    label_id = int(result['label'].replace("LABEL_", ""))
    label_map = {0: "공포", 1: "놀람", 2: "분노", 3: "슬픔", 4: "중립", 5: "행복", 6: "혐오"}
    return [label_map.get(label_id, "알 수 없음")]

def build_prompt(keywords, emotions, user_text, investment_pref=None):
    goal = keywords.get("목표")
    period = keywords.get("기간")
    money = keywords.get("금액")
    primary_emotion = emotions[0] if emotions else None
    emotion_intro = emotion_care.get(primary_emotion, "")
    emotion_str = ", ".join(emotions) if emotions else "감정 표현 없음"

    if goal or period or money:
        return f"""[사용자 감정 분석]
감정 상태: {emotion_str}
공감: {emotion_intro}

[투자 목표 분석]
- 목표: {goal or '명시되지 않음'}
- 기간: {period or '미제공'}
- 금액: {money or '미제공'}
- 투자 성향: {investment_pref or '미제공'}

[요청]
위의 감정 상태와 목표를 고려하여 재정 전략을 제시하고 3줄 요약 포함해주세요.
"""
    else:
        return f"""감정 상태: {emotion_str}\n공감: {emotion_intro}\n→ 목표/기간/금액/성향을 물어보는 질문을 따뜻하게 던져주세요."""

def call_clova(prompt, history, clova_key):
    history.append({"role": "user", "content": prompt})
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {clova_key}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
        "Accept": "application/json"
    }
    body = {
        "messages": history,
        "topP": 0.8,
        "temperature": 0.7,
        "maxTokens": 800
    }
    response = requests.post(
        "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005",
        headers=headers,
        json=body
    )
    return response.json()['result']['message']['content'] if response.ok else f"❌ API 오류: {response.status_code}"

def clova_finance_chatbot(user_input, clova_key):
    history = [{"role": "system", "content": "감정을 분석하고 현실적인 조언을 따뜻하게 전달하세요."}]
    emotions = detect_emotion(user_input)
    keywords = extract_keywords(user_input)
    prompt = build_prompt(keywords, emotions, user_input)
    reply = call_clova(prompt, history, clova_key)
    return {
        "감정": emotions,
        "키워드": keywords,
        "클로바 응답": reply
    }
