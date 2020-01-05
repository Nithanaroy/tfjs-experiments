// Shared a snippet at https://stackoverflow.com/a/59480783/1585523 with playground option that describes the main idea of how to create a drawing canvas

class ModelInference {
    lastX = 0; // variable names that start with # are private to the class
    lastY = 0;
    self;
    constructor({ visionModel, imgWidth = 28, imgHeight = 28 }) {
        this.visionModel = visionModel;
        this.imgWidth = imgWidth;
        this.imgHeight = imgHeight;
        self = this; // to make this available in user event handlers
        const [canvasId, imgPlaceholderId] = ["inference_canvas", "img_placeholder"]

        self.canvas = document.getElementById(canvasId);
        self.rawImage = document.getElementById(imgPlaceholderId);
        self.ctx = self.canvas.getContext("2d");

        self.canvas.addEventListener("mousedown", self.setLastCoords); // fires before mouse left btn is released
        self.canvas.addEventListener("mousemove", self.freeForm);
        // self.canvas.addEventListener("click", self.penTool);
        self.canvas.addEventListener("click", self.captureImage);

        self.addRequiredCanvasStyles(Math.max(imgWidth * 10, 280), Math.max(imgHeight * 10, 280));

        document.getElementById("predictBtn").addEventListener("click", self.predict);
        document.getElementById("clearBtn").addEventListener("click", self.erase);
    }

    addRequiredCanvasStyles(width = 280, height = 280) {
        self.canvas.style.background = "#000";
        // Dimensions of the canvas should be in the same order of input trained images
        self.canvas.setAttribute("width", width); // IMPORTANT: Don't use CSS https://stackoverflow.com/a/8693791/1585523
        self.canvas.setAttribute("height", height); // IMPORTANT: Don't use CSS https://stackoverflow.com/a/8693791/1585523
    }

    setLastCoords(e) {
        const { x, y } = self.canvas.getBoundingClientRect();
        self.lastX = e.clientX - x;
        self.lastY = e.clientY - y;
    }

    freeForm(e) {
        if (e.buttons !== 1) return; // left button is not pushed yet
        self.penTool(e);
    }

    penTool(e) {
        const { x, y } = self.canvas.getBoundingClientRect();
        const newX = e.clientX - x;
        const newY = e.clientY - y;

        self.ctx.beginPath();
        self.ctx.lineWidth = 5;
        self.ctx.moveTo(self.lastX, self.lastY);
        self.ctx.lineTo(newX, newY);
        self.ctx.strokeStyle = 'white';
        self.ctx.stroke();
        self.ctx.closePath();

        self.lastX = newX;
        self.lastY = newY;
    }

    captureImage(e) {
        self.rawImage.src = self.canvas.toDataURL('image/png');
    }

    erase() {
        self.ctx.clearRect(0, 0, self.canvas.width, self.canvas.height);
        self.rawImage.removeAttribute("src");
    }

    async predict() {
        const raw = tf.browser.fromPixels(self.rawImage, 1).arraySync();
        const predictionProbs = await self.visionModel.predict(raw, self.imgWidth, self.imgHeight);
        const [prediction, confidence] = [tf.argMax(predictionProbs, 1).arraySync(), tf.max(predictionProbs, 1).arraySync()];

        console.debug(`Probabilities for each class: ${predictionProbs}`);
        document.getElementById("prediction_confidence_container").innerHTML = (confidence * 100).toFixed(2);
        document.getElementById("prediction_container").innerHTML = prediction;

        return { prediction, confidence };
    }
}