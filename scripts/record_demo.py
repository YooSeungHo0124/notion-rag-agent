"""Record the README demo clips with a headless browser (repeatable, no manual screen capture).

Each scene drives the running app (http://127.0.0.1:8000) in a fresh browser profile, so
the sidebar holds no personal chat history, and records a WebM per scene. Convert with
scripts/demo_to_gif.sh. Topics are chosen so the clips only show generic technical notes
and the Odoo demo database, never private note content.

    uv run --with playwright==1.55.0 python scripts/record_demo.py [scene ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

APP = "http://127.0.0.1:8000/"
OUT = Path("docs/assets/raw")
SIZE = {"width": 1280, "height": 800}
INPUT = "textarea"


def ask(p: Page, q: str) -> None:
    box = p.locator(INPUT).first
    box.click()
    box.type(q, delay=35)
    time.sleep(0.4)
    box.press("Enter")


def done(p: Page, timeout_s: int = 120) -> None:
    """An answer is finished when its footer (copy button) shows up."""
    p.get_by_text("복사").last.wait_for(timeout=timeout_s * 1000)
    time.sleep(1.0)


def scene_ask(p: Page) -> None:
    ask(p, "FlashAttention은 어떻게 동작하고, 왜 설치를 고려했어?")
    done(p)
    p.locator(".trust .pill").first.click()
    time.sleep(2.0)
    p.get_by_role("button", name="출처 보기").first.click()
    time.sleep(3.0)


def scene_abstain(p: Page) -> None:
    ask(p, "우리 팀 2027년 마케팅 예산은 얼마로 확정됐어?")
    time.sleep(25)


def scene_erp(p: Page) -> None:
    p.get_by_role("button", name="에이전트").first.click()
    time.sleep(0.6)
    ask(p, "Odoo에서 수주 확정된 판매 주문 중 금액이 가장 큰 건과 고객은?")
    done(p, 180)
    p.get_by_text("실행 과정").first.click()
    time.sleep(1.2)
    p.locator(".trust .pill").first.click()
    time.sleep(2.5)


def scene_quiz(p: Page) -> None:
    p.get_by_role("button", name="스튜디오").first.click()
    time.sleep(0.6)
    topic = p.get_by_placeholder("주제로 좁히기 (선택) 예: OCR 전처리")
    topic.type("Transformer attention", delay=35)
    time.sleep(0.4)
    p.locator(".sm-opt", has_text="퀴즈").click()
    p.locator("section.qz").first.wait_for(timeout=180_000)
    time.sleep(1.0)
    for q in p.locator("section.qz").all()[:3]:  # answer the first three questions
        q.scroll_into_view_if_needed()
        q.locator(".qz-opt").nth(1).click()
        time.sleep(1.6)
    p.locator(".qz-score").scroll_into_view_if_needed()
    time.sleep(2.5)


SCENES = {"ask": scene_ask, "abstain": scene_abstain, "erp": scene_erp, "quiz": scene_quiz}


def main(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for name in names or list(SCENES):
            ctx = browser.new_context(viewport=SIZE, record_video_dir=str(OUT / name),
                                      record_video_size=SIZE, locale="ko-KR")
            p = ctx.new_page()
            p.goto(APP)
            p.get_by_text("무엇을 찾고 있나요?").wait_for(timeout=60_000)
            time.sleep(1.2)
            try:
                SCENES[name](p)
            finally:
                video = p.video
                ctx.close()
                if video:
                    dst = OUT / f"{name}.webm"
                    Path(video.path()).replace(dst)
                    print("saved", dst)
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1:])
