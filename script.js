const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const uploadContent = document.getElementById("uploadContent");
const imagePreview = document.getElementById("imagePreview");
const previewImage = document.getElementById("previewImage");
const removeImage = document.getElementById("removeImage");

const analyzeBtn = document.getElementById("analyzeBtn");
const buttonText = document.getElementById("buttonText");

const predictionContent = document.getElementById("predictionContent");
const predictionResult = document.getElementById("predictionResult");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const confidenceText = document.getElementById("confidenceText");

const predictionList = document.getElementById("predictionList");
const confidenceCircle = document.getElementById("confidenceCircle");


let selectedFile = null;


imageInput.addEventListener("change", function () {

    if (this.files.length > 0) {
        handleFile(this.files[0]);
    }

});


dropZone.addEventListener("dragover", function (event) {

    event.preventDefault();

    dropZone.classList.add("dragover");

});


dropZone.addEventListener("dragleave", function () {

    dropZone.classList.remove("dragover");

});


dropZone.addEventListener("drop", function (event) {

    event.preventDefault();

    dropZone.classList.remove("dragover");

    const files = event.dataTransfer.files;

    if (files.length > 0) {
        handleFile(files[0]);
    }

});


function handleFile(file) {

    if (!file.type.startsWith("image/")) {

        alert("Please select a valid image file.");

        return;
    }


    selectedFile = file;


    const reader = new FileReader();


    reader.onload = function (event) {

        previewImage.src = event.target.result;

        uploadContent.style.display = "none";

        imagePreview.classList.add("active");

        analyzeBtn.disabled = false;

    };


    reader.readAsDataURL(file);

}


removeImage.addEventListener("click", function (event) {

    event.stopPropagation();

    selectedFile = null;

    imageInput.value = "";

    previewImage.src = "";

    imagePreview.classList.remove("active");

    uploadContent.style.display = "block";

    analyzeBtn.disabled = true;

    resetPrediction();

});


analyzeBtn.addEventListener("click", async function () {

    if (!selectedFile) {
        return;
    }


    buttonText.textContent = "Analyzing...";

    analyzeBtn.disabled = true;

    analyzeBtn.classList.add("loading");


    const formData = new FormData();

    formData.append("file", selectedFile);


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok || data.error) {

            throw new Error(
                data.error || "Prediction failed."
            );

        }


        showPrediction(data);


    } catch (error) {

        alert(
            "Error: " + error.message
        );

    } finally {

        buttonText.textContent = "Analyze Image";

        analyzeBtn.disabled = false;

        analyzeBtn.classList.remove("loading");

    }

});


function showPrediction(data) {

    const topPrediction = data.prediction;

    const topConfidence = data.confidence;


    predictionContent
        .querySelector(".waiting")
        .style.display = "none";


    predictionResult.classList.add("active");


    prediction.textContent = topPrediction;

    confidence.textContent =
        topConfidence + "%";

    confidenceText.textContent =
        topPrediction;


    confidenceCircle.style.background =
        `conic-gradient(
            #38bdf8 ${topConfidence}%,
            rgba(148, 163, 184, 0.08) ${topConfidence}%
        )`;


    predictionList.innerHTML = "";


    data.predictions.forEach(
        function (item) {

            const predictionItem =
                document.createElement("div");

            predictionItem.className =
                "prediction-item";


            predictionItem.innerHTML = `

                <span class="prediction-name">
                    ${item.class}
                </span>

                <div class="progress">

                    <div
                        class="progress-bar"
                        style="width: ${item.confidence}%"
                    ></div>

                </div>

                <span class="prediction-percent">
                    ${item.confidence}%
                </span>

            `;


            predictionList.appendChild(
                predictionItem
            );

        }
    );

}


function resetPrediction() {

    predictionResult.classList.remove("active");


    predictionContent
        .querySelector(".waiting")
        .style.display = "block";


    prediction.textContent = "-";

    confidence.textContent = "0%";

    confidenceText.textContent = "-";


    confidenceCircle.style.background =
        "transparent";


    predictionList.innerHTML = `

        <div class="empty-list">
            Predictions will appear here
        </div>

    `;

}

