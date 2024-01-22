import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.kinematics.human_profile import HumanProfile
from src.types.human import Human

app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))


@app.get("/api/healthchecker")
def healthchecker():
    return {"status": "ok"}


@app.post("/api/project")
def project():
    return {"status": "ok"}


@app.post("/api/human")
def human(
    human: Human,
    preprocess_interpolate_on=False,
    preprocess_filter_on=False,
    preprocess_smoothing_on=True,
    postcalculate_filter_on=True,
    postcalculate_smoothing_on=True,
):
    human_profile = HumanProfile()
    human_profile.init_with_data(np.array(human.body_joints))
    human_profile.compute(
        preprocess_interpolate_on=preprocess_interpolate_on,
        preprocess_filter_on=preprocess_filter_on,
        preprocess_smoothing_on=preprocess_smoothing_on,
        postcalculate_filter_on=postcalculate_filter_on,
        postcalculate_smoothing_on=postcalculate_smoothing_on,
    )
    metrics = human_profile.get_metrics()
    return {"status": "ok", "metrics": metrics}


@app.post("/api/human/upload")
def human_upload(file: UploadFile = File(...)):
    return {"status": "ok"}


@app.post("/api/point")
def point():
    return {"status": "ok"}
