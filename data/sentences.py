"""
설문용 30개 문장 목록.
다양한 무례함 수준을 포함하도록 설계되었다.
실제 사용 시 다양한 화자(성별/연령/방언)가 녹음해야 한다.
"""

SENTENCES = [
    # 매우 공손 (예상 점수: 1~2) — 존댓말/반말/반존대 혼합
    {"id": 1,  "text": "죄송한데요, 혹시 시간 괜찮으시면 잠깐 여쭤봐도 될까요?",               "expected": 1, "style": "존댓말"},
    {"id": 2,  "text": "고마워, 정말 많이 도움됐어.",                                          "expected": 1, "style": "반말"},
    {"id": 3,  "text": "제가 잘 이해를 못 했을 수도 있어요. 다시 한 번만 설명해줄 수 있어요?", "expected": 2, "style": "반존대"},
    {"id": 4,  "text": "먼저 연락드리지 못해서 죄송합니다.",                                   "expected": 1, "style": "존댓말"},
    {"id": 5,  "text": "바쁜데 시간 내줘서 진짜 고마워.",                                      "expected": 1, "style": "반말"},
    {"id": 6,  "text": "실례지만, 잠깐 여쭤봐도 될까요?",                                     "expected": 1, "style": "반존대"},

    # 보통 (예상 점수: 2~3)
    {"id": 7,  "text": "이거 언제까지 해줄 수 있어요?",                                        "expected": 2, "style": "반존대"},
    {"id": 8,  "text": "그 내용 나한테도 알려줘.",                                             "expected": 2, "style": "반말"},
    {"id": 9,  "text": "오늘 회의 몇 시야?",                                                   "expected": 2, "style": "반말"},
    {"id": 10, "text": "이거 다시 한번 확인해봐요.",                                           "expected": 3, "style": "반존대"},
    {"id": 11, "text": "그거 제대로 하신 거 맞죠?",                                            "expected": 3, "style": "존댓말"},
    {"id": 12, "text": "왜 이렇게 된 거야?",                                                   "expected": 3, "style": "반말"},

    # 다소 무례 (예상 점수: 3~4)
    {"id": 13, "text": "그냥 빨리 해. 그게 뭐가 어렵다고.",                                   "expected": 3, "style": "반말"},
    {"id": 14, "text": "이것도 모르면 어떻게 해요, 진짜.",                                     "expected": 4, "style": "반존대"},
    {"id": 15, "text": "내가 몇 번을 말해야 알아들어?",                                        "expected": 4, "style": "반말"},
    {"id": 16, "text": "그거 당연한 거 아닌가요? 왜 물어보세요?",                              "expected": 3, "style": "존댓말"},
    {"id": 17, "text": "그러니까 지금까지 뭘 하고 있었던 거야?",                               "expected": 4, "style": "반말"},
    {"id": 18, "text": "시간 낭비야, 진짜. 빨리 끝내요.",                                      "expected": 4, "style": "반존대"},

    # 매우 무례 (예상 점수: 4~5)
    {"id": 19, "text": "이게 뭐야? 대체 일을 한 거야, 논 거야?",                               "expected": 5, "style": "반말"},
    {"id": 20, "text": "당신이 잘못한 거잖아요. 왜 나한테 따져요?",                            "expected": 4, "style": "반존대"},
    {"id": 21, "text": "그러니까 맨날 이 모양이지.",                                           "expected": 5, "style": "반말"},
    {"id": 22, "text": "됐습니다. 그냥 나가세요.",                                             "expected": 4, "style": "존댓말"},
    {"id": 23, "text": "이것도 못 하면 그냥 꺼져.",                                            "expected": 5, "style": "반말"},
    {"id": 24, "text": "진짜 이해가 안 되네, 이 사람.",                                        "expected": 5, "style": "반존대"},

    # 억양/어조로 판단해야 하는 중립 문장 (오디오 모델 핵심 테스트)
    {"id": 25, "text": "알겠어.",    "expected": None, "style": "반말"},    # 순응/퉁명 구분
    {"id": 26, "text": "그렇군요.",  "expected": None, "style": "반존대"},  # 공감/냉소 구분
    {"id": 27, "text": "해봐.",      "expected": None, "style": "반말"},    # 격려/명령 구분
    {"id": 28, "text": "잘 하셨네요.", "expected": None, "style": "반존대"}, # 진심/비꼼 구분
    {"id": 29, "text": "그래.",      "expected": None, "style": "반말"},    # 수긍/무시 구분
    {"id": 30, "text": "알겠습니다.", "expected": None, "style": "존댓말"}, # 공손/건성 구분
]


def get_sentence_by_id(sentence_id: int) -> dict:
    for s in SENTENCES:
        if s["id"] == sentence_id:
            return s
    return None


def get_all_texts() -> list[str]:
    return [s["text"] for s in SENTENCES]
