import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from .agent import AutofillAgent

app = FastAPI(title="Autofill Agent API")

# Global Agent Instance
agent = AutofillAgent()

class FillRequest(BaseModel):
    url: str

@app.get("/")
def read_root():
    return {"status": "Autofill Agent API is running"}

@app.post("/upload_cv")
async def upload_cv(file: UploadFile = File(...)):
    """Uploads and processes the CV PDF."""
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        # Process the PDF
        agent.process_pdf(file_location)
        
        # Cleanup
        os.remove(file_location)
        
        return {"message": f"Successfully processed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fill_form")
async def fill_form(request: FillRequest, background_tasks: BackgroundTasks):
    """Triggers the agent to fill the form at the given URL."""
    try:
        # Run in background to not block response
        background_tasks.add_task(agent.fill_form, request.url)
        return {"message": f"Started filling form at {request.url}"}
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

