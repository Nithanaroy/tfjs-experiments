/**
 * Use Comlink to trigger all the model functions in a web worker
 */

function callback(batch, logs) {
    console.log(`Batch: ${batch}`);
}

class VisionModel {
    /**
    * Creates an instance of a wrapper of Vision Model that resides in the worker thread
    * This has access to the DOM and can show visualizations, unlike the worker
    * @param {HTMLElement} tensorboardDiv Instance of a Div tag where to show live model training
    */
    constructor(tensorboardDiv) {
        this.tensorboardDiv = tensorboardDiv;
    }
    async init() {
        const VisionModelWorker = Comlink.wrap(new Worker("model-worker.js"));
        this.visionModelWorker = await new VisionModelWorker();
    }
    get imgHeight() {
        return this.visionModelWorker.dataBunch.state.imgWidth
    }
    get imgWidth() {
        return this.visionModelWorker.dataBunch.state.imgHeight
    }
    async run(batchSize = 1024, epochs = 1, trainExisting = true) {
        if (!this.visionModelWorker) {
            await this.init();
        }
        const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
        const vizCallbacks = tfvis.show.fitCallbacks(this.tensorboardDiv, metrics);
        // tfvis.show.modelSummary(this.tensorboardDiv, this.model);
        // Note: Comlink doesn't work with JS named arguments
        return this.visionModelWorker.run(batchSize, epochs, trainExisting, Comlink.proxy(vizCallbacks.onBatchEnd), Comlink.proxy(vizCallbacks.onEpochEnd));
    }
    predict(imgAsArray, imgWidth, imgHeight) {
        return this.visionModelWorker.predict(imgAsArray, imgWidth, imgHeight);
    }
}