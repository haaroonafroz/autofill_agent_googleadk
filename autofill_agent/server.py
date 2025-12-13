import os
import sys
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .agent import AutofillAgent

# Fix for Windows Event Loop if running directly (though we are removing playwright backend)
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(title="Autofill Agent API")

# Add CORS so the browser extension can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development. In prod, strict limit to extension ID.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = AutofillAgent()

class ProcessPageRequest(BaseModel):
    url: str
    html: str
    user_id: str

@app.get("/")
def read_root():
    return {"status": "Autofill Agent API is running"}

@app.post("/upload_cv")
async def upload_cv(
    file: UploadFile = File(...), 
    user_id: str = Form(...)
):
    """
    Uploads and processes the CV PDF for a specific user.
    """
    try:
        file_location = f"temp_{user_id}_{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        agent.process_pdf(file_location, user_id=user_id)
        os.remove(file_location)
        
        return {"message": f"Successfully processed {file.filename} for user {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_page")
async def process_page(request: ProcessPageRequest):
    """
    Receives HTML from the extension, decides what to fill, and returns actions.
    """
    try:
        actions = await agent.generate_form_actions(request.html, request.user_id)
        return {"actions": actions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    pass

@app.on_event("shutdown")
async def shutdown_event():
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
