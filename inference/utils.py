"""추론 유틸리티 — 외부 의존성 없음."""


def score_to_level(score: float) -> str:
    if score <= 1.5:
        return "매우 공손함"
    elif score <= 2.5:
        return "공손함"
    elif score <= 3.5:
        return "보통"
    elif score <= 4.5:
        return "무례함"
    else:
        return "매우 무례함 (싸가지 없음)"
