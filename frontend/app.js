// Canvas setup and drawing functionality
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
const predictBtn = document.getElementById("predictBtn");

// Flask backend URL
const API_URL = "http://localhost:5000";

let isDrawing = false;
let hasDrawn = false;

// Configure canvas drawing settings
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white";
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.lineJoin = "round";

// Mouse event handlers for desktop drawing
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

// Touch event handlers for mobile drawing
canvas.addEventListener("touchstart", handleTouch);
canvas.addEventListener("touchmove", handleTouch);
canvas.addEventListener("touchend", stopDrawing);

function startDrawing(e) {
  isDrawing = true;
  hasDrawn = true;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
}

function draw(e) {
  if (!isDrawing) return;

  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
}

function stopDrawing() {
  isDrawing = false;
  ctx.beginPath();
}

function handleTouch(e) {
  e.preventDefault();

  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;

  if (e.type === "touchstart") {
    isDrawing = true;
    hasDrawn = true;
    ctx.beginPath();
    ctx.moveTo(x, y);
  } else if (e.type === "touchmove" && isDrawing) {
    ctx.lineTo(x, y);
    ctx.stroke();
  }
}

function clearCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  hasDrawn = false;

  const resultContainer = document.getElementById("resultContainer");
  resultContainer.innerHTML = `
        <div class="result-label">Draw a digit to get started</div>
        <div style="color: #999; font-size: 18px;">The AI is ready to recognize your handwriting!</div>
    `;
}

async function recognizeDigit() {
  if (!hasDrawn) {
    showError("Please draw a digit first!");
    return;
  }

  const resultContainer = document.getElementById("resultContainer");
  resultContainer.innerHTML = `
        <div class="result-label">Analyzing your drawing...</div>
        <div class="loading"></div>
    `;

  predictBtn.disabled = true;

  try {
    const imageData = canvas.toDataURL("image/png");

    const response = await fetch(`${API_URL}/api/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: imageData,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    displayResult(data);
  } catch (error) {
    console.error("Prediction error:", error);
    showError(
      "Unable to connect to the server. Make sure the Python backend is running on http://localhost:5000"
    );
  } finally {
    predictBtn.disabled = false;
  }
}

function displayResult(data) {
  const resultContainer = document.getElementById("resultContainer");

  let probabilitiesHTML = "";
  if (data.probabilities) {
    probabilitiesHTML =
      '<div class="probabilities"><strong>Confidence levels:</strong>';
    for (let i = 0; i < data.probabilities.length; i++) {
      const prob = (data.probabilities[i] * 100).toFixed(1);
      probabilitiesHTML += `
                <div class="probability-bar">
                    <div class="digit-label">${i}:</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${prob}%"></div>
                    </div>
                    <div class="percentage">${prob}%</div>
                </div>
            `;
    }
    probabilitiesHTML += "</div>";
  }

  resultContainer.innerHTML = `
        <div class="result-label">Prediction Result</div>
        <div class="result-digit">${data.digit}</div>
        <div class="result-confidence">
            Confidence: ${(data.confidence * 100).toFixed(1)}%
        </div>
        ${probabilitiesHTML}
    `;
}

function showError(message) {
  const resultContainer = document.getElementById("resultContainer");
  resultContainer.innerHTML = `
        <div class="result-label">Error</div>
        <div class="error-message">${message}</div>
    `;
}
