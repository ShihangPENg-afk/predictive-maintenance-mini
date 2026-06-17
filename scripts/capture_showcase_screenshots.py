"""Capture README showcase screenshots (Swagger + rag-agentic-system health tab)."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "docs" / "images"
SWAGGER_URL = "http://127.0.0.1:8010/docs"
STREAMLIT_URL = "http://127.0.0.1:8501"

DEMO_FEATURES = {
    "temperature": "73.5",
    "pressure": "5.2",
    "vibration": "2.1",
    "speed": "118.0",
    "humidity": "48.0",
}


def capture_swagger(page, output_path: Path) -> None:
    page.goto(SWAGGER_URL, wait_until="networkidle", timeout=60000)
    page.wait_for_selector("text=Predictive Maintenance API", timeout=30000)
    page.screenshot(path=str(output_path), full_page=True)


def capture_rag_health_tab(page, output_path: Path) -> None:
    page.goto(STREAMLIT_URL, wait_until="networkidle", timeout=60000)
    page.get_by_role("tab", name="设备健康预测").click()
    page.get_by_role("button", name="获取模型信息").click()
    page.wait_for_selector("text=类别 classes", timeout=30000)

    for feature, value in DEMO_FEATURES.items():
        page.get_by_role("spinbutton", name=feature).fill(value)

    page.get_by_role("button", name="预测设备健康状态").click()
    page.wait_for_selector("text=预测结果", timeout=30000)
    page.wait_for_selector("text=recommendation", timeout=30000)
    page.evaluate(
        """() => {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) sidebar.style.display = 'none';
            const header = document.querySelector('[data-testid="stHeader"]');
            if (header) header.style.display = 'none';
        }"""
    )
    page.set_viewport_size({"width": 1440, "height": 2200})
    page.locator("text=预测结果").scroll_into_view_if_needed()
    page.wait_for_timeout(1000)
    page.screenshot(path=str(output_path), full_page=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    swagger_path = OUTPUT_DIR / "swagger_docs.png"
    rag_path = OUTPUT_DIR / "rag_agent_tab.png"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        capture_swagger(page, swagger_path)
        print(f"Saved {swagger_path}")

        capture_rag_health_tab(page, rag_path)
        print(f"Saved {rag_path}")

        browser.close()


if __name__ == "__main__":
    main()
