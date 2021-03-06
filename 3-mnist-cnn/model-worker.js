/**
 * All the model code that is run in a web worker and is Comlink compliant
 */

importScripts("https://cdnjs.cloudflare.com/ajax/libs/tensorflow/1.3.2/tf.min.js", "https://unpkg.com/comlink/dist/umd/comlink.js", "constants.js", "data.js");

window = globalThis; // window object is needed by tfjs for saving the model to  disk

class VisionModelWorker {

    constructor() {
        this.model = null; // holds tfjs model
        this.dataBunch = null; // holds the X and y datasets
        this.trainingHistories = []; // holds the logs of training results
    }
    async create(saveToLocalStorage = false) {
        this.model = tf.sequential();

        this.model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], kernelSize: 3, filters: 8, activation: 'relu' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        this.model.add(tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        this.model.add(tf.layers.flatten());
        this.model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        this.model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

        this.model.compile({ optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
        this.model.summary();

        if (saveToLocalStorage) {
            await this.model.save(Constants.MODEL_DISK_PATH());
        }
    }
    async getData(forceFetch = false) {
        if (!!this.dataBunch && !forceFetch) {
            return;
        }
        const numClasses = 10; // number of unique digits to classify
        this.dataBunch = new Data();
        await this.dataBunch.fetchDataAndSetupState(); // TODO: need a tf.tidy() around this
        this.dataBunch.trainY = tf.oneHot(this.dataBunch.trainY, numClasses);
        this.dataBunch.testY = tf.oneHot(this.dataBunch.testY, numClasses);
    }
    async train(epochs, batchSize, vizCallbacks) {
        const historyObj = await this.model.fit(this.dataBunch.trainX, this.dataBunch.trainY, {
            batchSize: batchSize,
            validationData: [this.dataBunch.testX, this.dataBunch.testY],
            epochs: epochs,
            shuffle: true,
            callbacks: vizCallbacks
        });
        this.trainingHistories.push(historyObj);
        console.debug(`Training history: ${JSON.stringify(historyObj.history)}`);
        return historyObj;
    }

    async run(batchSize = 1024, epochs = 1, trainExisting = true, onBatchEndCb = null, onEpochEndCb = null) {
        if (!this.model || !trainExisting) {
            await this.create();
        }
        await this.getData();
        const vizCallbacks = {
            onBatchEnd: onBatchEndCb,
            onEpochEnd: onEpochEndCb
        }
        return this.train(epochs, batchSize, vizCallbacks);
    }
    async predict(imgAsArray, imgWidth, imgHeight) {
        if (!this.model) {
            console.error("A model should be trained first to make predictions");
            return;
        }
        const x = tf.tensor3d(imgAsArray);
        const resized = tf.image.resizeBilinear(x, [imgWidth, imgHeight]);
        const tensor = resized.expandDims(0);
        return this.model.predict(tensor).arraySync();
    }
}

Comlink.expose(VisionModelWorker);