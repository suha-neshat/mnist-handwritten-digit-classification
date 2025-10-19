from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_from_b64

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MNIST API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # dev only
    allow_origin_regex=".*",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
async def predict(data: dict):
    img_b64 = data["image"]
    result = predict_from_b64(img_b64)
    return {"prediction": result}
