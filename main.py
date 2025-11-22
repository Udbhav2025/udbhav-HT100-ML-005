from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
import io

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

MODEL_PATH = "spiral_model.keras"

# Only used for display, NOT for decision now
ASSUME_OUTPUT_IS_PROB_PARKINSON = True

# Roughness threshold (tuned from your observation)
# Healthy:  roughness > 0.0001
# Parkinson: roughness < 0.0001
# So: if roughness <= ROUGHNESS_THRESHOLD => Parkinson-like
ROUGHNESS_THRESHOLD = 0.00011


# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------

app = FastAPI(title="Tremor Tracker Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ok for hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeResponse(BaseModel):
    decision: bool          # True -> Parkinson-like tremor
    probability: float      # CNN probability of Parkinson (for info only)
    message: str            # explanation


# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully from:", MODEL_PATH)
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


# -------------------------------------------------
# PREPROCESSING
# -------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert any input image to the format the model expects:
    (1, 224, 224, 3) with values in [0, 1].
    """
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr


# -------------------------------------------------
# ROUTES
# -------------------------------------------------

@app.get("/")
def root():
    return {"message": "Tremor Tracker backend is running."}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_spiral(spiral: UploadFile = File(...)):
    """
    Accepts an uploaded image file (canvas or upload),
    uses a roughness heuristic for decision, and shows CNN probability.
    """

    if model is None:
        return AnalyzeResponse(
            decision=False,
            probability=0.0,
            message="Model is not loaded on server. Check backend configuration.",
        )

    try:
        # 1. Read and open image
        img_bytes = await spiral.read()
        image = Image.open(io.BytesIO(img_bytes))

        # 2. Preprocess image
        x = preprocess_image(image)

        # 3. CNN prediction (for info only)
        preds = model.predict(x)
        raw_prob = float(preds[0][0])

        if ASSUME_OUTPUT_IS_PROB_PARKINSON:
            parkinson_prob = raw_prob
        else:
            parkinson_prob = 1.0 - raw_prob

        # 4. ROUGHNESS-BASED HEURISTIC  (main decision)
        roughness = float(np.mean(np.abs(np.diff(x))))

        # Debug prints in terminal
        print("\n===============================")
        print(f"RAW MODEL OUTPUT (CNN): {raw_prob:.4f}")
        print(f"INTERPRETED P(Parkinson): {parkinson_prob:.4f}")
        print(f"ROUGHNESS SCORE:          {roughness:.8f}")
        print(f"THRESHOLD (roughness):    {ROUGHNESS_THRESHOLD:.8f}")

        # Decision purely from roughness:
        # smaller roughness => Parkinson-like (per your observation)
        decision = roughness >= ROUGHNESS_THRESHOLD

        print(f"FINAL DECISION:  {'YES' if decision else 'NO'}")
        print("===============================\n")

        # 5. Build message for UI
        if decision:
            msg = (
                "⚠️ Parkinson-like tremor pattern detected (roughness-based).\n"
                f"Model probability (Parkinson): {parkinson_prob:.2f}\n"
                f"Roughness score: {roughness:.8f}\n"
                "This is NOT a diagnosis; please consult a doctor."
            )
        else:
            msg = (
                "✅ No strong Parkinson tremor pattern detected (roughness-based).\n"
                f"Model probability (Parkinson): {parkinson_prob:.2f}\n"
                f"Roughness score: {roughness:.8f}\n"
                "This screening does not replace a medical checkup."
            )

        return AnalyzeResponse(
            decision=decision,
            probability=parkinson_prob,
            message=msg,
        )

    except Exception as e:
        print("Error during analysis:", e)
        return AnalyzeResponse(
            decision=False,
            probability=0.0,
            message=f"Error analyzing image: {e}",
        )


# -------------------------------------------------
# RUN (for local dev)
# -------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,        # your frontend uses this
        reload=False,
    )