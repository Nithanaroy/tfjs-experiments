let state = {
    trainX: null,
    trainXBuffer: null,
    testX: null,
    testXBuffer: null,
    trainY: null,
    testY: null,
    numTrainImages: -1,
    numTestImages: -1,
    imgHeight: -1,
    imgWidth: -1
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
async function showRandomExampleOneOff(type="train") {
    if (!state.trainXBuffer || !state.testXBuffer) {
        const dataModel = new DataExtra();
        const newState = await dataModel.main();
        state = { ...state, ...newState };
    }

    const [X, y, dataSize] = type === "test" ? [state.testXBuffer, state.testY, state.numTestImages] : [state.trainXBuffer, state.trainY, state.numTrainImages]
    const imageIndex = parseInt(Math.random() * dataSize);
    const dv = new DataView(X);
    const offset = (imageIndex * state.imgHeight * state.imgWidth) + 16
    const { imageData, imgTensor } = DataExtra.parseBytesAsImage(dv, offset, state.imgWidth, state.imgHeight)
    
    showImageFromImageData("testImageCanvas", imageData);
    document.getElementById("textContainer").innerHTML = `Showing example ${imageIndex} of ${dataSize} whose label is ${y.slice(imageIndex, 1).arraySync()}`;
}

/**
 * Parses all data in the first call and computes in-memory for the subsequent clicks
 * Useful when you expect user clicks multiple times
 * @param {string} type One of "train" or "test" to indicate the source dataset to sample from
 */
async function showRandomExample(type = "train") {
    if (!state.trainX || !state.trainY) {
        const dataModel = new Data();
        const newState = await dataModel.fetchDataAndSetupState();
        state = { ...state, ...newState };
    }
    const [X, y, dataSize] = type === "test" ? [state.testX, state.testY, state.numTestImages] : [state.trainX, state.trainY, state.numTrainImages]
    const imageIndex = parseInt(Math.random() * dataSize);

    showImageFromTensor("testImageCanvas", X, imageIndex);
    document.getElementById("textContainer").innerHTML = `Showing example ${imageIndex} of ${dataSize} whose label is ${y.slice(imageIndex, 1).arraySync()}`;
}