import os
import shutil
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from pydantic import BaseModel
from typing import Optional
from .agent import AutofillAgent

app = FastAPI(title="Autofill Agent API")

# Global Agent Instance
# Note: In a real multi-user env, agent might need to be instantiated per request or handle multiple contexts.
# For now, we assume the agent class can handle dynamic user_ids per call.
agent = AutofillAgent()

class FillRequest(BaseModel):
    url: str
    user_id: str # Required to identify which CV data to use

@app.get("/")
def read_root():
    return {"status": "Autofill Agent API is running"}

@app.post("/upload_cv")
async def upload_cv(
    file: UploadFile = File(...), 
    user_id: str = Form(...) # Expect user_id as form data
):
    """
    Uploads and processes the CV PDF for a specific user.
    """
    try:
        # Create a temp file
        file_location = f"temp_{user_id}_{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        # Process the PDF for this user
        agent.process_pdf(file_location, user_id=user_id)
        
        # Cleanup
        os.remove(file_location)
        
        return {"message": f"Successfully processed {file.filename} for user {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fill_form")
async def fill_form(request: FillRequest, background_tasks: BackgroundTasks):
    """Triggers the agent to fill the form at the given URL using the specified user's data."""
    try:
        # Run in background to not block response
        background_tasks.add_task(agent.fill_form, request.url, request.user_id)
        return {"message": f"Started filling form at {request.url} for user {request.user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    # We delay browser start until needed or start here
    pass

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await agent.close_browser()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
