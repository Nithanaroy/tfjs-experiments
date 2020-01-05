/** 
 * https://javascript.info/arraybuffer-binary-arrays is an excellent resource to learn about array buffers and their manipulation in JavaScript
*/

class Data {
    constructor() {
        this.state = {
            trainX: null,
            trainY: null,
            testX: null,
            testY: null,
            imgHeight: -1,
            imgWidth: -1,
            numTrainImages: -1,
            numTestImages: -1,
        }
    }

    get trainX() { return this.state.trainX; }
    get testX() { return this.state.testX; }
    get trainY() { return this.state.trainY; }
    get testY() { return this.state.testY; }
    get numTrainImages() { return this.state.numTrainImages; }
    get numTestImages() { return this.state.numTestImages; }

    set trainX(newVal) { this.state.trainX = newVal; }
    set testX(newVal) { this.state.testX = newVal; }
    set trainY(newVal) { this.state.trainY = newVal; }
    set testY(newVal) { this.state.testY = newVal; }
    
    static parseBuffer(buffer, offset = 16, isBigEndianProcessor = false) {
        let allData = null;
        if (isBigEndianProcessor) {
            // Let native JavaScript decode the bytes as MNIST dataset is encoded in Big Endian format 
            allData = new Uint8Array(buffer, offset)
        }
        else {
            const dataView = new DataView(buffer);
            const numBytes = dataView.byteLength - offset;
            allData = new Uint8Array(numBytes);
            for (let i = offset, j = 0; i < numBytes; i++ , j++) {
                allData[j] = dataView.getUint8(i, isBigEndianProcessor);
            }
        }
        return allData;
    }

    /**
     * Parse all images from the buffer and return a 4D tensor of shape [num images, width, height, channels]
     * As this is a static method, it can be used independantly with any MNIST like array buffer :)
     * @param {ArrayBuffer} buffer Training data as an array buffer. Can be from the response of fetch()
     * @param {int} offset Number of bytes to ignore from the beginning to start parsing for input data
     * @param {int} width width of each image in pixels
     * @param {int} height height of each image in pixels
     * @param {int} numImages total number of training examples in the buffer
     * @param {bool} isBigEndianProcessor endiannes of the machine is needed to read the raw bytes
     */
    static parseAllImages(buffer, offset = 16, width = 28, height = 28, numImages = 60000, isBigEndianProcessor = false) {
        return tf.tensor4d(Data.parseBuffer(buffer, offset, isBigEndianProcessor), [numImages, width, height, 1]);
    }

    static parseAllLabels(buffer, offset = 8, isBigEndianProcessor = false) {
        return tf.tensor1d(Data.parseBuffer(buffer, offset, isBigEndianProcessor));
    }

    /**
     * Identifies the current system's endianness 
     * Source: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DataView#Endianness
     */
    checkIfLittleEndianProcessor() {
        const buffer = new ArrayBuffer(2);
        new DataView(buffer).setInt16(0, 256, true /* littleEndian */);
        // Int16Array uses the platform's endianness.
        return new Int16Array(buffer)[0] === 256;
    }

    /**
     * Download training data and update state with parsed information
     * Returns the downloaded data as array buffers
     */
    async fetchDataAndSetupState() {
        const trainXReq = fetch("data/train-images-idx3-ubyte");
        const trainYReq = fetch("data/train-labels-idx1-ubyte");
        const testXReq = fetch("data/t10k-images-idx3-ubyte");
        const testYReq = fetch("data/t10k-labels-idx1-ubyte");

        // Parse Train X
        const trainXBuffer = await (await trainXReq).arrayBuffer();
        let dv = new DataView(trainXBuffer)
        // Decode bytes as mentioned in "FILE FORMATS FOR THE MNIST DATABASE" section in http://yann.lecun.com/exdb/mnist/
        const isBigEndianProcessor = !this.checkIfLittleEndianProcessor();
        const magicNumber = dv.getInt32(0, isBigEndianProcessor);
        this.state.numTrainImages = dv.getInt32(4, isBigEndianProcessor);
        this.state.imgWidth = dv.getInt32(8, isBigEndianProcessor);
        this.state.imgHeight = dv.getInt32(12, isBigEndianProcessor);
        this.state.trainX = Data.parseAllImages(trainXBuffer, 16, this.state.imgWidth, this.state.imgHeight, this.numTrainImages, isBigEndianProcessor);

        // Parse Test X
        const testXBuffer = await (await testXReq).arrayBuffer();
        dv = new DataView(testXBuffer);
        this.state.numTestImages = dv.getInt32(4, isBigEndianProcessor);
        this.state.testX = Data.parseAllImages(testXBuffer, 16, this.state.imgWidth, this.state.imgHeight, this.numTestImages, isBigEndianProcessor);

        // Parse Train Y
        const trainYBuffer = await (await trainYReq).arrayBuffer();
        dv = new DataView(trainYBuffer);
        this.state.trainY = Data.parseAllLabels(trainYBuffer, 8, isBigEndianProcessor);

        // Parse Test Y
        const testYBuffer = await (await testYReq).arrayBuffer();
        dv = new DataView(testYBuffer);
        this.state.testY = Data.parseAllLabels(testYBuffer, 8, isBigEndianProcessor);

        console.debug("Processed data");
        return { trainXBuffer, testXBuffer }
    }
}