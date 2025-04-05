from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import Optional
import tempfile
import shutil
from models.multimodal_fusion import MultimodalMarketIntelligence

# Initialize the application
app = FastAPI(
    title="Nielsen Market Intelligence API",
    description="API for market trend analysis and product intelligence",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the multimodal system
print("Initializing multimodal market intelligence system...")
mmi = MultimodalMarketIntelligence()
print("System initialized successfully!")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Nielsen Market Intelligence API"}

@app.post("/api/analyze_trends")
async def analyze_trends(
    category: Optional[str] = None,
    with_competitors: bool = False
):
    """
    Analyze market trends.
    
    Args:
        category: Product category (optional)
        with_competitors: Include competitor analysis
    """
    try:
        result = mmi.identify_market_trends(
            category=category,
            with_competitors=with_competitors
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze_product")
async def analyze_product(
    description: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Analyze a product using text and optional image.
    
    Args:
        description: Product description
        image: Product image (optional)
    """
    try:
        # If image is provided, save it temporarily
        if image:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                shutil.copyfileobj(image.file, tmp)
                tmp_path = tmp.name
            
            # Analyze with image
            result = mmi.analyze_product(description, tmp_path)
            
            # Remove temporary file
            os.unlink(tmp_path)
        else:
            # Analyze without image
            result = mmi.analyze_product(description)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local development
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
