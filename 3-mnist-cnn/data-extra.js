/** 
 * Add additional features to Data useful for debugging the data layer
*/

class DataExtra extends Data {
    constructor() {
        super();

        this.state = { ...this.state, ...{ "trainXBuffer": null, "testXBuffer": null } };
    }

    /**
     * Creates a tf.Tensor from raw bytes of several MNIST images
     * Background theory behind ImageData is available at https://developer.mozilla.org/en-US/docs/Web/API/ImageData/ImageData#Initializing_ImageData_with_an_array
     * @param {DataView} dataView a view into the MNIST image's ArrayBuffer instance
     * @param {int} offset byte offset in the dataview to start reading the image data
     * @param {int} width width of the image in bytes / pixels
     * @param {int} height height of the image in bytes / pixels
     */
    static parseBytesAsImage(dataView, offset, width, height) {
        const channels = 4;
        const imgAsArray = new Uint8ClampedArray(width * height * channels);
        for (let i = 0, j = offset; i < imgAsArray.length; i += channels, j++) {
            imgAsArray[i + 0] = dataView.getUint8(j, false);     // R value 
            imgAsArray[i + 1] = 0;     // G value 
            imgAsArray[i + 2] = 0;     // B value 
            imgAsArray[i + 3] = 255;   // A value 
        }
        const image = new ImageData(imgAsArray, width, height);
        return {
            "imageData": image,
            "imgTensor": tf.browser.fromPixels(image, channels)
        }
    }

    async main() {
        const {trainXBuffer, testXBuffer} = await super.fetchDataAndSetupState();
        this.state = { ...this.state, trainXBuffer, testXBuffer };
        return this.state;
    }
}