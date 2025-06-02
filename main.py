from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from typing import Dict

import re

def is_spectro_query(q: str) -> bool:
    # Smarter: match "spectrum", "spectra", "spectroscopy", "uv", "ir", "nmr", etc.
    keywords = [
        r"spectra", r"spectrum", r"spectroscopy", r"uv[- ]?vis", r"uv", r"ir", r"nmr",
        r"mass spec", r"show.*spectrum", r"show.*spectra"
    ]
    for kw in keywords:
        if re.search(kw, q):
            return True
    return False

# Microservice URLs
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
    allow_origins=["*"],   # Open for now; lock down in prod!
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok", "service": "gateway"}

# --- Models for each endpoint ---
class RetroRequest(BaseModel):
    smiles: str

class ExtractRequest(BaseModel):
    text: str

class MoleculeRequest(BaseModel):
    molecule: str

# --- Normal endpoints for direct access ---
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

# ===========================================================
# 🟣 NEW: Add /chat endpoint for "Chat is the Brain" workflow
# ===========================================================

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_router(data: ChatRequest) -> Dict:
    q = data.question.lower()

    # Extraction (compound/entity recognition)
    if "extract" in q or "compound" in q:
        payload = {"text": data.question}
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{EXTRACT_URL}/extract", json=payload)
            answer = await resp.json()
        return {
            "type": "extract",
            "answer": answer,
            "tool": "ChemDataExtractor"
        }

    # 🟣 Spectroscopy — SMART matching!
    elif is_spectro_query(q):
        # Try to extract molecule name after "of", or fallback to last word, or default to benzene
        mol = None
        match = re.search(r"(?:of|for|of the|for the)\s+([a-zA-Z0-9 \-\(\)\[\]]+)", q)
        if match:
            mol = match.group(1).strip()
        else:
            # Fallback: strip the keywords and take what's left
            mol = re.sub(
                r"(spectra|spectrum|spectroscopy|uv[- ]?vis|uv|ir|nmr|mass spec|show|please|plot|give|draw|for|of|the)",
                "", q
            ).strip()
        if not mol or mol in ["spectrum", "spectra", "spectroscopy", "uv", "ir", "nmr"]:
            mol = "benzene"
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
    elif any(word in q for word in ["retro", "synth", "route", "smiles"]):
        import re
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
