from fastapi.middleware.cors import CORSMiddleware
from Router.router import router
from fastapi import FastAPI
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)