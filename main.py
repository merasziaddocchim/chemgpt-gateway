from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

# Load microservice URLs from env (or hardcode for now)
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

@app.post("/retro")
async def retro(req: Request):
    body = await req.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{RETRO_URL}/retrosynthesis", json=body)
        return resp.json()

@app.post("/extract")
async def extract(req: Request):
    body = await req.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{EXTRACT_URL}/extract", json=body)
        return resp.json()

# --- NEW: Use a Pydantic model for /spectro so Swagger UI shows an input box! ---
class MoleculeRequest(BaseModel):
    molecule: str

@app.post("/spectro")
async def spectro(data: MoleculeRequest):
    payload = {"molecule": data.molecule}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{SPECTRO_URL}/spectroscopy", json=payload)
        return resp.json()

# Add /chat, /auth, etc. as needed!
