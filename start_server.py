#!/usr/bin/env python3
"""
DeepSeek-OCR FastAPI bridge to a locally running Ollama model.
"""

import os
import io
import tempfile
import base64
import json
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
import requests

# Default prompt mirrors the previous behavior but can be overridden.
DEFAULT_PROMPT = os.environ.get("DEFAULT_PROMPT", "<image>\n<|grounding|>Convert the document to markdown.")

# Ollama settings (point to your locally running Ollama instance).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-ocr:latest")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")

# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek-OCR API",
    description="OCR service using DeepSeek-OCR via Ollama backend",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OCRResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    page_count: Optional[int] = None

class BatchOCRResponse(BaseModel):
    success: bool
    results: List[OCRResponse]
    total_pages: int
    filename: str
    csv: Optional[str] = None

def pdf_to_images_high_quality(pdf_data: bytes, dpi: int = 144) -> List[Image.Image]:
    """Convert PDF bytes to high-quality PIL Images"""
    images = []
    
    # Save PDF data to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
        temp_pdf.write(pdf_data)
        temp_pdf_path = temp_pdf.name
    
    try:
        pdf_document = fitz.open(temp_pdf_path)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert to PIL Image
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        pdf_document.close()
    finally:
        # Clean up temporary file
        os.unlink(temp_pdf_path)
    
    return images

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string for Ollama."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_ollama(prompt: str, images: List[Image.Image]) -> str:
    """Send a prompt + images to Ollama and return the response text."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "images": [encode_image_to_base64(img) for img in images],
    }

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    except Exception as exc:  # network failure
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama returned {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama response was not JSON: {resp.text}") from exc

    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")

    return data.get("response", "")


def process_single_image(image: Image.Image, prompt: str = DEFAULT_PROMPT) -> str:
    """Process a single image using the Ollama DeepSeek-OCR model."""
    print(f"[DEBUG] Sending request to Ollama model={OLLAMA_MODEL} prompt_len={len(prompt)}")
    result = call_ollama(prompt, [image])
    print(f"[DEBUG] Ollama output length: {len(result)}")
    return result


def coerce_bool(value: Optional[str]) -> bool:
    """Convert form values to bool with lenient handling."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def markdown_tables_to_csv(markdown_text: str) -> str:
    """
    Extract simple GitHub-style tables from markdown and return CSV text.
    Tables are separated by a blank line in the CSV output.
    """
    import csv
    lines = [line.strip() for line in markdown_text.splitlines()]
    tables = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "|" in line and i + 1 < len(lines) and "|" in lines[i + 1] and "-" in lines[i + 1]:
            header_line = lines[i]
            data_lines = []
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                data_lines.append(lines[i])
                i += 1

            def parse_row(row_line: str) -> List[str]:
                parts = [cell.strip() for cell in row_line.split("|")]
                if parts and parts[0] == "":
                    parts = parts[1:]
                if parts and parts[-1] == "":
                    parts = parts[:-1]
                return parts

            header = parse_row(header_line)
            rows = [parse_row(dl) for dl in data_lines] if data_lines else []
            if header:
                tables.append([header] + rows)
        else:
            i += 1

    if not tables:
        return ""

    output = io.StringIO()
    writer = csv.writer(output)
    for idx, table in enumerate(tables):
        for row in table:
            writer.writerow(row)
        if idx != len(tables) - 1:
            writer.writerow([])
    return output.getvalue()


def ensure_output_dir() -> str:
    """Create and return the output directory path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def combine_markdown_from_results(results: List[OCRResponse]) -> str:
    """Combine per-page markdown results into a single string with separators."""
    parts = []
    for item in results:
        if not isinstance(item, OCRResponse):
            continue
        label = f"Page {item.page_count}:\n" if item.page_count else ""
        content = item.result or item.error or ""
        if content:
            parts.append(f"{label}{content}")
    return "\n\n---\n\n".join(parts)


def save_outputs(base_name: str, markdown_text: str, csv_text: Optional[str] = None):
    """Save markdown and optional CSV to the outputs directory."""
    out_dir = ensure_output_dir()
    md_path = Path(out_dir) / f"{base_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    if csv_text:
        csv_path = Path(out_dir) / f"{base_name}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_text)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "DeepSeek-OCR API is running", "status": "healthy"}

@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
    """Simple web UI for uploading PDFs/images and toggling CSV extraction."""
    escaped_prompt = json.dumps(DEFAULT_PROMPT)
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>DeepSeek OCR (Ollama)</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          max-width: 960px;
          margin: 20px auto;
          padding: 0 16px;
          line-height: 1.4;
        }}
        textarea, input[type="text"] {{
          width: 100%;
          box-sizing: border-box;
        }}
        #output {{
          white-space: pre-wrap;
          background: #f5f5f5;
          padding: 12px;
          border-radius: 6px;
          border: 1px solid #ddd;
        }}
        .row {{
          margin-bottom: 12px;
        }}
        button {{
          padding: 10px 16px;
          font-size: 14px;
        }}
      </style>
    </head>
    <body>
      <h2>DeepSeek OCR (Ollama-backed)</h2>
      <div class="row">
        <input id="file" type="file" accept=".pdf,image/*"/>
      </div>
      <div class="row">
        <label for="prompt">Prompt (optional)</label><br/>
        <textarea id="prompt" rows="3" placeholder="Leave empty for default prompt"></textarea>
      </div>
      <div class="row">
        <label><input type="checkbox" id="asCsv" checked/> Extract CSV from markdown tables (PDF only)</label>
      </div>
      <div class="row">
        <label><input type="checkbox" id="saveOutput" checked/> Save output to server outputs folder</label>
      </div>
      <div class="row">
        <button onclick="runOCR()">Run OCR</button>
      </div>
      <div class="row">
        <strong>Status:</strong> <span id="status">Idle</span>
      </div>
      <div class="row">
        <h4>Output</h4>
        <div id="output"></div>
      </div>
      <script>
        const defaultPrompt = {escaped_prompt};

        async function runOCR() {{
          const fileInput = document.getElementById('file');
          if (!fileInput.files.length) {{
            alert('Select a PDF or image file first.');
            return;
          }}
          const file = fileInput.files[0];
          const prompt = document.getElementById('prompt').value || defaultPrompt;
          const asCsv = document.getElementById('asCsv').checked;
          const saveOutput = document.getElementById('saveOutput').checked;
          const formData = new FormData();
          formData.append('file', file);
          if (prompt) formData.append('prompt', prompt);
          if (saveOutput) formData.append('save', 'true');
          let endpoint = '/ocr/image';
          if (file.name.toLowerCase().endsWith('.pdf')) {{
            endpoint = '/ocr/pdf';
            if (asCsv) formData.append('as_csv', 'true');
          }}
          setStatus('Sending request...');
          try {{
            const resp = await fetch(endpoint, {{
              method: 'POST',
              body: formData
            }});
            const text = await resp.text();
            let parsed;
            try {{ parsed = JSON.parse(text); }} catch (e) {{}}
            if (!resp.ok) {{
              setOutput(text || 'Request failed');
              setStatus('Error');
              return;
            }}
            if (parsed) {{
              if (parsed.results) {{
                let combined = parsed.results.map(r => (r.page_count ? `Page ${{r.page_count}}:\\n` : '') + (r.result || r.error || '')).join('\\n\\n---\\n\\n');
                if (parsed.csv) {{
                  combined += "\\n\\n--- CSV ---\\n" + parsed.csv;
                }}
                setOutput(combined);
              }} else {{
                setOutput(parsed.result || parsed.error || text);
              }}
            }} else {{
              setOutput(text);
            }}
            setStatus('Done');
          }} catch (err) {{
            setOutput('Error: ' + err);
            setStatus('Error');
          }}
        }}

        function setStatus(msg) {{
          document.getElementById('status').textContent = msg;
        }}
        function setOutput(msg) {{
          document.getElementById('output').textContent = msg;
        }}
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/health")
async def health_check():
    """Detailed health check with a light Ollama ping."""
    ollama_ok = False
    ollama_error = None
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=5)
        ollama_ok = resp.status_code == 200
        if not ollama_ok:
            ollama_error = f"Unexpected status {resp.status_code}"
    except Exception as exc:
        ollama_error = str(exc)

    return {
        "status": "healthy",
        "ollama_reachable": ollama_ok,
        "ollama_error": ollama_error,
        "ollama_model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
    }

@app.post("/ocr/image", response_model=OCRResponse)
async def process_image_endpoint(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    save: Optional[str] = Form(None),
):
    """Process a single image file with optional custom prompt"""
    try:
        print(f"[DEBUG] Image endpoint called for file: {file.filename}")
        
        # Read image data
        image_data = await file.read()
        print(f"[DEBUG] Read {len(image_data)} bytes of image data")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        print(f"[DEBUG] Converted to PIL Image, size: {image.size}")
        
        # Debug logging
        print(f"[DEBUG] Received prompt parameter: {repr(prompt)}")
        print(f"[DEBUG] Default PROMPT: {repr(DEFAULT_PROMPT)}")
        
        # Use provided prompt or default
        use_prompt = prompt if prompt else DEFAULT_PROMPT
        print(f"[DEBUG] Image endpoint selected prompt: {repr(use_prompt)}")
        print(f"[DEBUG] Using custom prompt: {prompt is not None}")
        
        # Process with DeepSeek-OCR via Ollama
        print(f"[DEBUG] Sending image to DeepSeek-OCR via Ollama...")
        result = process_single_image(image, use_prompt)
        print(f"[DEBUG] OCR complete, output length: {len(result)}")
        save_flag = coerce_bool(save)
        if save_flag and result:
            base_name = Path(file.filename).stem
            save_outputs(base_name, result, None)
            print(f"[DEBUG] Saved output for {base_name} in {OUTPUT_DIR}")
        
        return OCRResponse(
            success=True,
            result=result,
            page_count=1
        )
        
    except Exception as e:
        print(f"[ERROR] Image endpoint failed: {str(e)}")
        return OCRResponse(
            success=False,
            error=str(e)
        )

@app.post("/ocr/pdf", response_model=BatchOCRResponse)
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    as_csv: Optional[str] = Form(None),
    save: Optional[str] = Form(None),
):
    """Process a PDF file with optional custom prompt"""
    try:
        print(f"[DEBUG] PDF endpoint called for file: {file.filename}")
        print(f"[DEBUG] Received prompt parameter: {repr(prompt)}")
        print(f"[DEBUG] Default PROMPT: {repr(DEFAULT_PROMPT)}")
        
        # Read PDF data
        pdf_data = await file.read()
        print(f"[DEBUG] Read {len(pdf_data)} bytes of PDF data")
        
        # Convert PDF to images
        images = pdf_to_images_high_quality(pdf_data, dpi=144)
        print(f"[DEBUG] Converted PDF to {len(images)} images")
        
        if not images:
            print(f"[DEBUG] No images extracted from PDF")
            return BatchOCRResponse(
                success=False,
                results=[],
                total_pages=0,
                filename=file.filename
            )
        
        # Use provided prompt or default
        use_prompt = prompt if prompt else DEFAULT_PROMPT
        print(f"[DEBUG] PDF endpoint selected prompt: {repr(use_prompt)}")
        print(f"[DEBUG] Using custom prompt: {prompt is not None}")
        as_csv_flag = coerce_bool(as_csv)
        print(f"[DEBUG] CSV conversion requested: {as_csv_flag}")
        save_flag = coerce_bool(save)
        print(f"[DEBUG] Save requested: {save_flag}")
        
        # Process each page
        results = []
        for page_num, image in enumerate(tqdm(images, desc="Processing pages")):
            try:
                print(f"[DEBUG] Processing page {page_num + 1}/{len(images)}")
                result = process_single_image(image, use_prompt)
                results.append(OCRResponse(
                    success=True,
                    result=result,
                    page_count=page_num + 1
                ))
                print(f"[DEBUG] Page {page_num + 1} processed successfully, output length: {len(result)}")
            except Exception as e:
                print(f"[ERROR] Page {page_num + 1} failed: {str(e)}")
                results.append(OCRResponse(
                    success=False,
                    error=f"Page {page_num + 1} error: {str(e)}",
                    page_count=page_num + 1
                ))
        
        print(f"[DEBUG] PDF processing complete: {len(results)} pages processed")
        csv_content = None
        csv_content = None
        if as_csv_flag:
            combined_markdown = "\n\n".join(
                r.result for r in results if r.success and r.result is not None
            )
            csv_content = markdown_tables_to_csv(combined_markdown)
            print(f"[DEBUG] CSV extraction complete, length: {len(csv_content) if csv_content else 0}")
        combined_for_save = combine_markdown_from_results(results)
        if save_flag and combined_for_save:
            base_name = Path(file.filename).stem
            save_outputs(base_name, combined_for_save, csv_content)
            print(f"[DEBUG] Saved outputs for {base_name} in {OUTPUT_DIR}")
        return BatchOCRResponse(
            success=True,
            results=results,
            total_pages=len(images),
            filename=file.filename,
            csv=csv_content,
        )
        
    except Exception as e:
        print(f"[ERROR] PDF endpoint failed: {str(e)}")
        return BatchOCRResponse(
            success=False,
            results=[OCRResponse(success=False, error=str(e))],
            total_pages=0,
            filename=file.filename
        )

@app.post("/ocr/batch")
async def process_batch_endpoint(files: List[UploadFile] = File(...), prompt: Optional[str] = Form(None)):
    """Process multiple files (images and PDFs) with optional custom prompt"""
    results = []
    
    for file in files:
        if file.filename.lower().endswith('.pdf'):
            result = await process_pdf_endpoint(file, prompt)
        else:
            result = await process_image_endpoint(file, prompt)
        
        results.append({
            "filename": file.filename,
            "result": result
        })
    
    return {"success": True, "results": results}

if __name__ == "__main__":
    print("Starting DeepSeek-OCR API server...")
    uvicorn.run(
        "start_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
