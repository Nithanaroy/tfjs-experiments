let state = {
    dataBunch: null,
    dataExtraBunch: null,
    visionModel: null,
    alertObj: null,
    inferenceViewModel: null
}

const trainImgBtn = document.getElementById("parseTrainImageBtn");
const testImgBtn = document.getElementById("parseTestImageBtn");
const trainDataBtn = document.getElementById("parseTrainDataBtn");
const testDataBtn = document.getElementById("parseTestDataBtn");
const testDataPrepBtns = [trainImgBtn, testImgBtn, trainDataBtn, testDataBtn];

function init() {
    state.alertObj = new Alert(document.getElementById("alert_div"));

    trainImgBtn.addEventListener('click', ev => showExampleOneOffOnClick(ev, "train"));
    testImgBtn.addEventListener('click', ev => showExampleOneOffOnClick(ev, "test"));
    trainDataBtn.addEventListener('click', ev => showExampleOnClick(ev, "train"));
    testDataBtn.addEventListener('click', ev => showExampleOnClick(ev, "test"));

    document.getElementById("predictBtn").addEventListener('click', ev => noModelForInferenceWarning());
    document.getElementById("clearBtn").addEventListener('click', ev => noModelForInferenceWarning());
}

function noModelForInferenceWarning() {
    if (!state.visionModel) {
        state.alertObj.showMsg("Train a model first to use inference options", "warning");        
    }
}

function showImageFromImageData(canvasId, imageData) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    ctx.putImageData(imageData, 0, 0);
}

function showImageFromTensor(canvasId, imgsTensor, imgIndex) {
    const img = imgsTensor.slice([imgIndex, 0, 0, 0], 1).squeeze(0);
    tf.browser.toPixels(img, document.getElementById(canvasId));
}

/**
 * Parses only the image asked for everytime on click
 * @param {string} type One of "train" or "test" to indicate the source dataset to sample from
 */
async function showRandomExampleOneOff(type = "train") {
    if (!state.dataExtraBunch) {
        state.dataExtraBunch = new DataExtra();
        await state.dataExtraBunch.main();
    }

    const { testXBuffer, testY, numTestImages, trainXBuffer, trainY, numTrainImages, imgHeight, imgWidth } = state.dataExtraBunch.state;
    const [X, y, dataSize] = type === "test" ? [testXBuffer, testY, numTestImages] : [trainXBuffer, trainY, numTrainImages]
    const imageIndex = parseInt(Math.random() * dataSize);
    const dv = new DataView(X);
    const offset = (imageIndex * imgHeight * imgWidth) + 16
    const { imageData, imgTensor } = DataExtra.parseBytesAsImage(dv, offset, imgWidth, imgHeight)

    showImageFromImageData("testImageCanvas", imageData);
    document.getElementById("textContainer").innerHTML = `Showing example ${imageIndex} of ${dataSize} whose label is ${y.slice(imageIndex, 1).arraySync()}`;
}

function showExampleOneOffOnClick(ev, type) {
    testDataPrepBtns.forEach(el => el.disabled = true);
    showRandomExampleOneOff(type).finally(() => testDataPrepBtns.forEach(el => el.disabled = false));
}

/**
 * Parses all data in the first call and computes in-memory for the subsequent clicks
 * Useful when you expect user clicks multiple times
 * @param {string} type One of "train" or "test" to indicate the source dataset to sample from
 */
async function showRandomExample(type = "train") {
    if (!state.dataBunch) {
        state.dataBunch = new Data();
        await state.dataBunch.fetchDataAndSetupState();
    }

    const { testY, numTestImages, trainY, numTrainImages, trainX, testX } = state.dataBunch.state;
    const [X, y, dataSize] = type === "test" ? [testX, testY, numTestImages] : [trainX, trainY, numTrainImages]
    const imageIndex = parseInt(Math.random() * dataSize);

    showImageFromTensor("testImageCanvas", X, imageIndex);
    document.getElementById("textContainer").innerHTML = `Showing example ${imageIndex} of ${dataSize} whose label is ${y.slice(imageIndex, 1).arraySync()}`;
}

function showExampleOnClick(ev, type) {
    testDataPrepBtns.forEach(el => el.disabled = true);
    showRandomExample(type).finally(() => testDataPrepBtns.forEach(el => el.disabled = false));
}

async function train() {
    if (!state.visionModel) {
        state.visionModel = new VisionModel(document.getElementById("tensorboard"));
    }
    const t0 = performance.now();
    state.alertObj.showMsg("Training session is in progress...", "info");
    return state.visionModel.run(parseForm()).then(trainingHistory => {
        setUpInference();
        const t1 = performance.now();
        state.alertObj.showMsg(`Successfully trained the model in ${Math.round(t1 - t0)}ms. Try it with your own writing`, "success");
    }, e => {
        state.alertObj.showMsg("Unable to train. Check browser console for more details", "danger");
        console.error(e);
    });
}

function parseForm() {
    return {
        batchSize: parseInt(document.getElementById("batchSizeTb").value) || 1024,
        epochs: parseInt(document.getElementById("epochsTb").value) || 1,
        trainExisting: !document.getElementById("createNewCb").checked || false
    }
}

async function setUpInference() {
    // Inference can be done only after the model has been trained at least once.
    state.inferenceViewModel = new ModelInference({
        visionModel: state.visionModel,
        imgWidth: await state.visionModel.imgWidth,
        imgHeight: await state.visionModel.imgHeight
    });
}

document.getElementById("trainingForm").addEventListener('submit', function (e) {
    e.preventDefault();
    const startTrainingBtn = document.getElementById("startTrainingBtn");
    startTrainingBtn.disabled = true;
    train().finally(() => startTrainingBtn.disabled = false);
});

init();