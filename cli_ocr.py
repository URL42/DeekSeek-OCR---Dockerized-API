#!/usr/bin/env python3
"""
Interactive CLI helper for the Ollama-backed DeepSeek-OCR API.
Guides you through choosing a file, optional prompt, and CSV extraction.
"""

import os
import sys
import pathlib
import requests

def default_api_base() -> str:
    port = os.environ.get("API_PORT", "8000")
    return os.environ.get("OCR_API_BASE", f"http://localhost:{port}")


def prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or (default or "")


def prompt_bool(text: str, default: bool = False) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    value = input(f"{text}{suffix}: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "1", "true", "on"}


def choose_file() -> pathlib.Path:
    while True:
        candidate = pathlib.Path(prompt("Path to PDF or image file"))
        if candidate.exists() and candidate.is_file():
            return candidate
        print("File not found, please try again.")


def process_file(api_base: str, path: pathlib.Path, prompt_text: str, as_csv: bool) -> None:
    is_pdf = path.suffix.lower() == ".pdf"
    endpoint = "/ocr/pdf" if is_pdf else "/ocr/image"
    url = f"{api_base.rstrip('/')}{endpoint}"
    mime = "application/pdf" if is_pdf else "application/octet-stream"
    files = {"file": (path.name, open(path, "rb"), mime)}
    data = {}
    if prompt_text:
        data["prompt"] = prompt_text
    if is_pdf and as_csv:
        data["as_csv"] = "true"

    print(f"\nSending to {url} (as_csv={as_csv if is_pdf else 'n/a'})...")
    resp = requests.post(url, files=files, data=data, timeout=600)
    if resp.status_code != 200:
        print(f"Request failed: {resp.status_code} {resp.text}")
        return
    try:
        payload = resp.json()
    except Exception:
        print(f"Non-JSON response:\n{resp.text}")
        return

    print("\n--- OCR Response ---")
    if is_pdf and "results" in payload:
        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            page = item.get("page_count")
            success = item.get("success")
            print(f"\nPage {page} (success={success}):")
            print(item.get("result") or item.get("error", ""))
        if as_csv and payload.get("csv"):
            print("\n--- CSV (from markdown tables) ---")
            print(payload["csv"])
    else:
        print(payload.get("result") or payload.get("error", payload))


def main() -> int:
    api_base = default_api_base()
    print(f"Using API base: {api_base}")
    path = choose_file()
    prompt_text = prompt("Custom prompt (leave blank for default)", "")
    as_csv = False
    if path.suffix.lower() == ".pdf":
        as_csv = prompt_bool("Extract CSV from markdown tables?", default=False)
    try:
        process_file(api_base, path, prompt_text, as_csv)
    except KeyboardInterrupt:
        print("\nAborted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
