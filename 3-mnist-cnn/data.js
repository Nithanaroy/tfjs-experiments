/** 
 * https://javascript.info/arraybuffer-binary-arrays is an excellent resource to learn about array buffers and their manipulation in JavaScript
*/

class Data {
    constructor() {
        this.state = {
            imgsTensor: null,
            imgHeight: -1,
            imgWidth: -1,
            numImages: -1
        }
    }

    parseAllImages(buffer, offset = 16, width = 28, height = 28, numImages = 60000, isBigEndianProcessor = false) {
        let allImgData = null;
        if (isBigEndianProcessor) {
            // Let native JavaScript decode the bytes as MNIST dataset is encoded in Big Endian format 
            allImgData = new Uint8Array(buffer, offset)
        }
        else {
            const dataView = new DataView(buffer);
            const numBytes = dataView.byteLength - offset;
            allImgData = new Uint8Array(numBytes);
            for (let i = offset, j = 0; i < numBytes; i++ , j++) {
                allImgData[j] = dataView.getUint8(i, isBigEndianProcessor);
            }
        }
        return tf.tensor4d(allImgData, [numImages, width, height, 1]);
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
        const r = await fetch("data/train-images-idx3-ubyte");
        const b = await r.arrayBuffer();
        const dv = new DataView(b)

        // Decode bytes as mentioned in "FILE FORMATS FOR THE MNIST DATABASE" section in http://yann.lecun.com/exdb/mnist/
        const isBigEndianProcessor = !this.checkIfLittleEndianProcessor();
        const magicNumber = dv.getInt32(0, isBigEndianProcessor);
        const numImages = dv.getInt32(4, isBigEndianProcessor);
        const imgWidth = dv.getInt32(8, isBigEndianProcessor);
        const imgHeight = dv.getInt32(12, isBigEndianProcessor);

        this.state.imgsTensor = this.parseAllImages(b, 16, imgWidth, imgHeight, numImages, isBigEndianProcessor);
        this.state = { ...this.state, numImages, imgWidth, imgHeight };

        console.debug("Downloaded");
        return {
            xBuffer: b
        }
    }
}