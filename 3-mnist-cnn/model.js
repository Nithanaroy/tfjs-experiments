class VisionModel {
    /**
     * Creates an instance of Vision Model
     * @param {HTMLElement} tensorboardDiv Instance of a Div tag where to show live model training
     */
    constructor(tensorboardDiv) {
        this.model = null;
        this.dataBunch = null;
        this.trainingHistories = [];
        this.tensorboardDiv = tensorboardDiv;
    }
    create() {
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

        return this.model;
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
    async train(epochs, batchSize) {
        const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
        const fitCallbacks = tfvis.show.fitCallbacks(this.tensorboardDiv, metrics);
        tfvis.show.modelSummary(this.tensorboardDiv, this.model);
        const historyObj = await this.model.fit(this.dataBunch.trainX, this.dataBunch.trainY, {
            batchSize: batchSize,
            validationData: [this.dataBunch.testX, this.dataBunch.testY],
            epochs: epochs,
            shuffle: true,
            callbacks: fitCallbacks
        });
        this.trainingHistories.push(historyObj);
        console.debug(`Training history: ${JSON.stringify(historyObj.history)}`);
        return historyObj;
    }
    async run(batchSize = 1024, epochs = 1, trainExisting = true) {
        if (!this.model || !trainExisting) {
            this.create();
        }
        await this.getData();
        await this.train(epochs, batchSize);
    }
}