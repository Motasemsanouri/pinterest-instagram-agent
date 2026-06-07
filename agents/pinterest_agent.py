from __future__ import annotations

import asyncio
import random
import re
import time
import urllib.parse
from collections import Counter
from typing import Any, Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from agents.base_agent import BaseAgent
from agents.models import PinData, PinterestScrapingResult
from agents.state import AgentState
from config.settings import get_settings
from utils.logger import get_logger
from utils.retry import async_retry

logger = get_logger("pinterest_agent")

PINTEREST_SEARCH_URL = "https://www.pinterest.com/search/pins/?q={query}&rs=typed"
PINTEREST_BASE_URL = "https://www.pinterest.com"

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]

_STEALTH_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    window.chrome = { runtime: {} };
"""


class PinterestScraper:
    """
    Async Playwright-based Pinterest scraper.

    Applies anti-detection techniques (navigator.webdriver override, randomised
    user-agent, human-like scrolling) and falls back gracefully when selectors
    change.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = get_logger("pinterest_scraper")

    # ─── Public API ───────────────────────────────────────────────────────────

    async def scrape(self, query: str, max_pins: int) -> PinterestScrapingResult:
        """Run the full scraping pipeline and return structured results."""
        start_time = time.time()

        async with async_playwright() as pw:
            browser = await self._launch_browser(pw)
            try:
                context = await self._create_context(browser)
                page = await context.new_page()
                await self._apply_stealth(page)

                if self.settings.pinterest_email and self.settings.pinterest_password:
                    await self._login(page)

                pins = await self._scrape_search(page, query, max_pins)
                duration = time.time() - start_time

                return PinterestScrapingResult(
                    query=query,
                    pins=pins,
                    total_scraped=len(pins),
                    total_attempted=max_pins,
                    scraping_duration_seconds=duration,
                    metadata={
                        "user_agent": await page.evaluate("navigator.userAgent"),
                        "authenticated": bool(self.settings.pinterest_email),
                    },
                )
            except Exception as exc:
                duration = time.time() - start_time
                self.logger.error("scraping_failed", query=query, error=str(exc))
                return PinterestScrapingResult(
                    query=query,
                    pins=[],
                    total_scraped=0,
                    total_attempted=max_pins,
                    scraping_duration_seconds=duration,
                    errors=[str(exc)],
                )
            finally:
                await browser.close()

    # ─── Browser setup ────────────────────────────────────────────────────────

    async def _launch_browser(self, pw: Playwright) -> Browser:
        return await pw.chromium.launch(
            headless=self.settings.headless_browser,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )

    async def _create_context(self, browser: Browser) -> BrowserContext:
        return await browser.new_context(
            user_agent=random.choice(_USER_AGENTS),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            },
        )

    async def _apply_stealth(self, page: Page) -> None:
        await page.add_init_script(_STEALTH_SCRIPT)

    # ─── Authentication ───────────────────────────────────────────────────────

    async def _login(self, page: Page) -> None:
        """Attempt Pinterest login — non-fatal on failure."""
        try:
            self.logger.info("attempting_login")
            await page.goto(
                "https://www.pinterest.com/login/",
                wait_until="networkidle",
                timeout=self.settings.browser_timeout,
            )
            await asyncio.sleep(random.uniform(1.0, 2.0))

            email_el = await page.wait_for_selector("#email", timeout=10_000)
            if email_el:
                await email_el.fill(self.settings.pinterest_email)
                await asyncio.sleep(random.uniform(0.3, 0.7))

            pwd_el = await page.wait_for_selector("#password", timeout=10_000)
            if pwd_el:
                await pwd_el.fill(self.settings.pinterest_password)
                await asyncio.sleep(random.uniform(0.3, 0.7))

            await page.keyboard.press("Enter")
            await page.wait_for_load_state("networkidle", timeout=self.settings.browser_timeout)
            await asyncio.sleep(random.uniform(2.0, 3.0))
            self.logger.info("login_completed")

        except Exception as exc:
            self.logger.warning("login_failed", error=str(exc), note="continuing unauthenticated")

    # ─── Scraping ─────────────────────────────────────────────────────────────
    async def _goto_with_retry(self, page: Page, url: str) -> None:
        """Navigate with retry/backoff using settings-driven attempts."""

        @async_retry(
            max_attempts=self.settings.max_retries,
            wait_min=self.settings.retry_delay,
            wait_max=self.settings.retry_delay * 5,
        )
        async def _go() -> None:
            t0 = time.time()
            await page.goto(
                url,
                wait_until="networkidle",
                timeout=self.settings.browser_timeout,
            )
            self.logger.debug(
                "navigation_succeeded",
                url=url,
                load_time_s=round(time.time() - t0, 2),
                page_title=await page.title(),
            )

        await _go()
    async def _scrape_search(self, page: Page, query: str, max_pins: int) -> list[PinData]:
        """Navigate to the search page and scroll until we have enough pins."""
        url = PINTEREST_SEARCH_URL.format(query=urllib.parse.quote(query))
        self.logger.info("navigating_to_search", url=url, max_pins=max_pins)

        await self._goto_with_retry(page, url)
        await asyncio.sleep(random.uniform(2.0, 4.0))

        pins: list[PinData] = []
        seen_ids: set[str] = set()
        max_scrolls = max(max_pins // 5 + 10, 20)
        stale_scrolls = 0

        for scroll_num in range(max_scrolls):
            new_pins = await self._extract_pins_from_page(page, seen_ids)
            added = 0
            for pin in new_pins:
                if len(pins) >= max_pins:
                    break
                pins.append(pin)
                seen_ids.add(pin.pin_id)
                added += 1

            stale_scrolls = 0 if added else stale_scrolls + 1
            if stale_scrolls and stale_scrolls % 3 == 0:
                self.logger.warning(
                    "scroll_yielding_no_new_pins",
                    scroll=scroll_num,
                    consecutive_stale_scrolls=stale_scrolls,
                    scraped=len(pins),
                    note="page may be rate-limited, blocked, or selectors stale",
                )

            self.logger.info(
                "scraping_progress",
                scraped=len(pins),
                target=max_pins,
                scroll=scroll_num,
                new_pins_this_scroll=added,
            )

            if len(pins) >= max_pins:
                break

            await self._human_scroll(page)
            await asyncio.sleep(
                random.uniform(self.settings.scroll_delay_min, self.settings.scroll_delay_max)
            )

        return pins[:max_pins]

    async def _extract_pins_from_page(self, page: Page, seen_ids: set[str]) -> list[PinData]:
        pins: list[PinData] = []
        skip_reasons: Counter = Counter()
        try:
            link_elements = await page.query_selector_all('a[href*="/pin/"]')
            self.logger.debug("pin_links_found", count=len(link_elements))
            for el in link_elements:
                try:
                    pin, skip_reason = await self._extract_pin_from_element(el)
                    if pin is None:
                        skip_reasons[skip_reason or "unknown"] += 1
                    elif pin.pin_id in seen_ids:
                        skip_reasons["already_seen"] += 1
                    else:
                        pins.append(pin)
                except Exception as exc:
                    skip_reasons["exception"] += 1
                    self.logger.debug("pin_element_failed", error=str(exc))
        except Exception as exc:
            self.logger.warning("page_extraction_failed", error=str(exc))

        if skip_reasons:
            self.logger.debug("pin_extraction_skips", new_pins=len(pins), reasons=dict(skip_reasons))
        return pins

    async def _extract_pin_from_element(self, link_el: Any) -> tuple[Optional[PinData], Optional[str]]:
        """Returns (pin, skip_reason). skip_reason is set (and pin is None) when skipped."""
        href = await link_el.get_attribute("href")
        if not href or "/pin/" not in href:
            return None, "no_pin_href"

        match = re.search(r"/pin/(\d+)/", href)
        if not match:
            return None, "href_no_pin_id"

        pin_id = match.group(1)
        pin_url = f"{PINTEREST_BASE_URL}{href}" if href.startswith("/") else href

        img_el = await link_el.query_selector("img")
        image_url = ""
        alt_text: Optional[str] = None

        if img_el:
            srcset = await img_el.get_attribute("srcset")
            src = await img_el.get_attribute("src")
            alt_text = await img_el.get_attribute("alt")

            if srcset:
                image_url = self._best_from_srcset(srcset)
            if not image_url and src and "pinimg.com" in src:
                image_url = src
            if not image_url and src:
                image_url = src

        if not image_url:
            return None, "no_img_element" if not img_el else "no_image_url"

        title = (alt_text[:500] if alt_text and len(alt_text) > 500 else alt_text) or None

        pin = PinData(
            pin_id=pin_id,
            title=title,
            image_url=image_url,
            original_image_url=self._upscale_pinimg_url(image_url),
            thumbnail_url=image_url,
            pin_url=pin_url,
            alt_text=alt_text,
            raw_data={"href": href},
        )
        return pin, None

    # ─── Image URL helpers ────────────────────────────────────────────────────

    def _best_from_srcset(self, srcset: str) -> str:
        """Return the highest-width URL from a srcset attribute string."""
        entries: list[tuple[int, str]] = []
        for chunk in srcset.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            tokens = chunk.split()
            if not tokens:
                continue
            url = tokens[0]
            width = 0
            if len(tokens) >= 2:
                try:
                    width = int(tokens[1].rstrip("w"))
                except ValueError:
                    pass
            entries.append((width, url))

        if not entries:
            return ""
        entries.sort(key=lambda x: x[0], reverse=True)
        return entries[0][1]

    def _upscale_pinimg_url(self, url: str) -> str:
        """Try to replace Pinterest thumbnail resolution with originals."""
        if "pinimg.com" not in url:
            return url
        for size in ("/60x60/", "/236x/", "/474x/", "/736x/"):
            if size in url:
                return url.replace(size, "/originals/")
        return url

    # ─── Scrolling ────────────────────────────────────────────────────────────

    async def _human_scroll(self, page: Page) -> None:
        scroll_px = random.randint(600, 1200)
        await page.evaluate(
            f"window.scrollBy({{top: {scroll_px}, left: 0, behavior: 'smooth'}})"
        )
        await asyncio.sleep(random.uniform(0.3, 0.8))


class PinterestAgent(BaseAgent):
    """
    LangGraph node: scrapes Pinterest for pins matching the query.

    Reads from state:  query, max_pins
    Writes to state:   pinterest_result, current_step, errors
    """

    def __init__(self) -> None:
        super().__init__(name="PinterestAgent")
        self.scraper = PinterestScraper()

    async def run(self, state: AgentState) -> dict[str, Any]:
        self.logger.info("agent_started", query=state["query"], max_pins=state["max_pins"])

        try:
            result = await self.scraper.scrape(
                query=state["query"],
                max_pins=state["max_pins"],
            )
            self.logger.info(
                "agent_completed",
                total_scraped=result.total_scraped,
                duration_s=round(result.scraping_duration_seconds, 2),
            )
            return {
                "pinterest_result": result,
                "current_step": "pinterest_agent:completed",
                "errors": result.errors,
            }
        except Exception as exc:
            return self._error_update(state, exc)
