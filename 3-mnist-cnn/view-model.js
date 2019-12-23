let state = {
    imgsTensor: null,
    xBuffer: null,
    numImages: -1,
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
 * @param {Event} e User event instance
 */
async function showRandomImageOneOff(e) {
    if (!state.xBuffer) {
        const dataModel = new DataExtra();
        const newState = await dataModel.main();
        state = { ...state, ...newState };
    }
    const imageIndex = parseInt(Math.random() * state.numImages);
    const dv = new DataView(state.xBuffer);
    const offset = (imageIndex * state.imgHeight * state.imgWidth) + 16
    const { imageData, imgTensor } = DataExtra.parseBytesAsImage(dv, offset, state.imgWidth, state.imgHeight)
    showImageFromImageData("testImageCanvas", imageData);
    document.getElementById("textContainer").innerHTML = `Showing image number, ${imageIndex}`;
}

/**
 * Parses all data in the first call and computes in-memory for the subsequent clicks
 * Useful when you expect user clicks multiple times
 * @param {Event} e User event instance
 */
async function showRandomImage(e) {
    if (!state.imgsTensor) {
        const dataModel = new Data();
        const newState = await dataModel.fetchDataAndSetupState();
        state = { ...state, ...newState };
    }
    const imageIndex = parseInt(Math.random() * state.numImages);
    showImageFromTensor("testImageCanvas", state.imgsTensor, imageIndex);
    document.getElementById("textContainer").innerHTML = `Showing image number, ${imageIndex}`;
}