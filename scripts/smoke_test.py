"""
Quick Playwright smoke test for health-monitor Gradio app.
Tests: app loads, 5 tabs exist, 3D viz renders, demo inference works.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from playwright.async_api import async_playwright


async def smoke_test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        results = []

        # 1. Initial load
        try:
            await page.goto("http://localhost:7860/", wait_until="networkidle", timeout=15000)
            results.append(("App loads", "PASS"))
        except Exception as e:
            results.append(("App loads", "FAIL: " + str(e)))
            await browser.close()
            return results

        # 2. Count tabs
        tabs = await page.query_selector_all('[role="tab"]')
        tab_count = len(tabs)
        results.append(
            ("Tabs found", "PASS" if tab_count >= 5 else f"FAIL: got {tab_count}")
        )

        # 3. List tab names
        tab_texts = []
        for tab in tabs:
            text = await tab.inner_text()
            tab_texts.append(text.strip())
        results.append((f"Tab labels: {tab_texts}", "PASS"))

        # 4. Click 3D Visualization tab via JS (Gradio workaround)
        await page.evaluate(
            """() => {
            const tabs = document.querySelectorAll('[role="tab"]');
            for (const t of tabs) {
                if (t.textContent.includes('3D Visualization')) t.click();
            }
        }"""
        )
        await page.wait_for_timeout(1000)

        # 5. Check generate button
        gen_btn = await page.query_selector(
            "button:has-text('Generate 3D Visualisation')"
        )
        results.append(
            ("3D Viz button", "PASS" if gen_btn else "FAIL: not found")
        )

        # 6. Click generate and check Plotly
        if gen_btn:
            await gen_btn.click()
            await page.wait_for_timeout(5000)
            canvas = await page.query_selector(".js-plotly-plot")
            results.append(
                ("Plotly renders", "PASS" if canvas else "FAIL: no plot detected")
            )
        else:
            results.append(("Plotly renders", "SKIP: no button"))

        # 7. Test Simulated Demo
        await page.evaluate(
            """() => {
            const tabs = document.querySelectorAll('[role="tab"]');
            for (const t of tabs) {
                if (t.textContent.includes('Simulated Demo')) t.click();
            }
        }"""
        )
        await page.wait_for_timeout(1000)

        demo_btn = await page.query_selector(
            "button:has-text('Generate and Analyse')"
        )
        results.append(
            ("Demo button", "PASS" if demo_btn else "FAIL: not found")
        )

        if demo_btn:
            await demo_btn.click()
            await page.wait_for_timeout(3000)
            content = await page.content()
            has_results = (
                "detected" in content.lower()
                or "overall status" in content.lower()
            )
            results.append(
                (
                    "Demo inference",
                    "PASS" if has_results else "WARN: no results detected",
                )
            )
        else:
            results.append(("Demo inference", "SKIP: no button"))

        await browser.close()
        return results


if __name__ == "__main__":
    results = asyncio.run(smoke_test())
    print()
    print("=" * 50)
    print("  SMOKE TEST RESULTS")
    print("=" * 50)
    for desc, status in results:
        if status == "PASS":
            mark = "  PASS"
        elif "WARN" in status:
            mark = "  WARN"
        elif "SKIP" in status:
            mark = "  SKIP"
        else:
            mark = "  FAIL"
        print(f"  [{mark}] {desc}: {status}")
    print(f"\n  {len(results)} checks total")
    passes = sum(1 for _, s in results if s == "PASS")
    fails = sum(1 for _, s in results if s != "PASS")
    print(f"  {passes} passed, {fails} non-pass")
    sys.exit(0 if fails == 0 else 1)
