from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from typing import Dict
import re

# === Keyword detection for spectroscopy ===
def is_spectro_query(q: str) -> bool:
    keywords = [
        r"spectra", r"spectrum", r"spectroscopy", r"uv[- ]?vis", r"uv", r"ir", r"nmr",
        r"mass spec", r"show.*spectrum", r"show.*spectra"
    ]
    ql = q.lower()
    for kw in keywords:
        if re.search(kw, ql):
            return True
    return False

# === Microservice URLs ===
RETRO_URL = os.getenv("RETRO_URL", "https://chemgpt-se-production.up.railway.app")
EXTRACT_URL = os.getenv("EXTRACT_URL", "https://chemgpt-extract-production.up.railway.app")
SPECTRO_URL = os.getenv("SPECTRO_URL", "https://chemgpt-spectro-production.up.railway.app")

app = FastAPI(
    title="ChemGPT API Gateway",
    description="Routes and orchestrates all ChemGPT microservices.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok", "service": "gateway"}

# === Request Models ===
class RetroRequest(BaseModel):
    smiles: str

class ExtractRequest(BaseModel):
    text: str

class MoleculeRequest(BaseModel):
    molecule: str

# === Direct endpoints ===
@app.post("/retro")
async def retro(data: RetroRequest):
    payload = {"smiles": data.smiles}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{RETRO_URL}/retrosynthesis", json=payload)
        return resp.json()

@app.post("/extract")
async def extract(data: ExtractRequest):
    payload = {"text": data.text}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{EXTRACT_URL}/extract", json=payload)
        return resp.json()

@app.post("/spectro")
async def spectro(data: MoleculeRequest):
    payload = {"molecule": data.molecule}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{SPECTRO_URL}/spectroscopy", json=payload)
        return resp.json()

# === "Chat is the Brain" endpoint ===
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_router(data: ChatRequest) -> Dict:
    q = data.question
    ql = q.lower()

    # Extraction (compound/entity recognition)
    if "extract" in ql or "compound" in ql:
        payload = {"text": q}
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{EXTRACT_URL}/extract", json=payload)
            answer = await resp.json()
        return {
            "type": "extract",
            "answer": answer,
            "tool": "ChemDataExtractor"
        }

    # 🟣 Spectroscopy — Smart matching and molecule extraction
    elif is_spectro_query(q):
        # 1. Try "of/for <mol>" pattern (case-insensitive, allow dashes, parens, spaces, etc.)
        match = re.search(r"(?:of|for|of the|for the)\s+([a-zA-Z0-9\-\(\)\[\] ]+)", q, re.IGNORECASE)
        if match:
            mol = match.group(1).strip()
        else:
            # 2. Fallback: remove keywords, what's left is probably the molecule
            mol = re.sub(
                r"(spectra|spectrum|spectroscopy|uv[- ]?vis|uv|ir|nmr|mass spec|show|please|plot|give|draw|for|of|the)",
                "", q, flags=re.IGNORECASE
            ).strip()

        # 3. Edge cases: if empty, default to "benzene"
        if not mol or mol.lower() in ["spectrum", "spectra", "spectroscopy", "uv", "ir", "nmr"]:
            mol = "benzene"

        # Optional: clean up excess spaces, commas, etc.
        mol = mol.strip(",.;: ")

        # Debug log
        print(f"🔬 [DEBUG] Spectro request parsed molecule: '{mol}' from question: '{q}'")

        payload = {"molecule": mol}
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{SPECTRO_URL}/spectroscopy", json=payload)
            answer = await resp.json()
        return {
            "type": "spectro",
            "answer": answer,
            "tool": "ChemGPT Spectro"
        }

    # Retrosynthesis (for keywords like 'retro', 'synth', 'route', 'smiles')
    elif any(word in ql for word in ["retro", "synth", "route", "smiles"]):
        smiles = None
        match = re.search(r'([A-Za-z0-9@+\-\[\]\(\)=#$%]+)', q)
        if match:
            smiles = match.group(1)
        if not smiles:
            smiles = "c1ccccc1"
        payload = {"smiles": smiles}
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{RETRO_URL}/retrosynthesis", json=payload)
            answer = await resp.json()
        return {
            "type": "retro",
            "answer": answer,
            "tool": "AiZynthFinder"
        }

    # Fallback for unrecognized input
    else:
        return {
            "type": "unknown",
            "answer": "Sorry, I don't recognize this request yet. Try asking for extraction, spectrum, or retrosynthesis!",
            "tool": None
        }
