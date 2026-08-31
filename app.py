from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse

import torch
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification


app = FastAPI(
    title="ResNet Image Classifier",
    description="CIFAR-10 Image Classification using ResNet-18",
    version="2.0.0"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model_path = "./resnet_cifar10"


processor = AutoImageProcessor.from_pretrained(
    model_path
)


model = ResNetForImageClassification.from_pretrained(
    model_path
)


model.to(device)
model.eval()


@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse("index.html")


@app.get("/style.css")
async def style():
    return FileResponse(
        "style.css",
        media_type="text/css"
    )


@app.get("/script.js")
async def script():
    return FileResponse(
        "script.js",
        media_type="application/javascript"
    )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if not file.content_type:
        return {
            "error": "Invalid file."
        }


    if not file.content_type.startswith("image/"):
        return {
            "error": "Please upload a valid image file."
        }


    image = Image.open(
        file.file
    ).convert("RGB")


    inputs = processor(
        images=image,
        return_tensors="pt"
    )


    pixel_values = inputs[
        "pixel_values"
    ].to(device)


    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values
        )


    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )


    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=5,
        dim=1
    )


    predictions = []


    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):

        label = model.config.id2label[
            index.item()
        ]


        predictions.append({
            "class": label,
            "confidence": round(
                probability.item() * 100,
                2
            )
        })


    return {
        "prediction": predictions[0]["class"],
        "confidence": predictions[0]["confidence"],
        "predictions": predictions
    }

