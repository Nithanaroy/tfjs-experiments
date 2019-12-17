/**
 * Tensorflow JS code which trains and tests a model
 * Assumes tfjs is imported
 */
const DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [4],
        units: 5,
        activation: "relu"
    }));
    model.add(tf.layers.dense({
        activation: "softmax",
        units: 3
    }));

    model.compile({
        loss: "categoricalCrossentropy",
        optimizer: tf.train.adam(0.003),
        metrics: ['accuracy']
    })
    return model;
}

function createData(irisUrl, batchSize = 10) {
    const oneHotEncoderY = {
        "Iris-setosa": [1, 0, 0],
        "Iris-versicolor": [0, 1, 0],
        "Iris-virginica": [0, 0, 1]
    }
    const csvDataset = tf.data.csv(
        irisUrl, {
            hasHeader: false,
            columnNames: ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
            columnConfigs: {
                species: {
                    isLabel: true
                }
            }
        });
    // csvDataset is of the form:
    // xs: {sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2}
    // ys: {species: "Iris-setosa"}
    const data = csvDataset.map(({
        xs,
        ys
    }) => {
        // Convert xs(features) and ys(labels) from object form (keyed by
        // column name) to array form.
        return {
            xs: [xs.sepal_length, xs.sepal_width, xs.petal_length, xs.petal_width],
            ys: [ys.species]
        };
    })
        .map(({
            xs,
            ys
        }) => {
            // One hot encode y column
            return {
                xs,
                ys: oneHotEncoderY[ys]
            }
        })
        .batch(batchSize);
    return data;
}

async function train(model, data, epochs = 100) {
    const history = await model.fitDataset(data, {
        epochs: epochs,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                if (epoch % 10 == 0) {
                    console.debug(`Metrics: ${JSON.stringify(logs)} after ${epoch} epochs`);
                }
            }
        }
    });
    return history;
}

async function main() {
    const model = createModel();
    const data = await createData("iris_data.csv");
    // data.forEachAsync(d => d.print());
    const history = await train(model, data);
    // console.debug(history);
    console.info("Done training!")
    return model;
}

async function modelPredict(model, input = [5.8, 2.7, 5.1, 1.9]) {
    const flowerNameFromIndex = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }

    console.debug(`Predicting, ${input}`);
    const probabilities = model.predict(tf.tensor2d(input, [1, 4]))
    console.debug(`Probabilities:`);
    probabilities.print();
    const predictions = await probabilities.argMax(-1).array();
    const confidenceScores = await probabilities.max(-1).array();
    console.debug(
        `This is ${flowerNameFromIndex[predictions[0]]} with ${(confidenceScores[0] * 100).toFixed(2)}% confidence`
    );
    return { "prediction": flowerNameFromIndex[predictions[0]], "confidence": confidenceScores[0]};
}

async function callMain() {
    const model = await main();
    await modelPredict(model);
    return model;

    // return tf.tidy(async () => {
    // })
}