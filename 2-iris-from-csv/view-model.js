/**
 * The script that connection functions between the UI and business logic
 */

let state = {
    model: null,
    alertObj: null
}

function init() {
    state.alertObj = document.getElementById("alert_div")
}

/**
 * helper method to sets an alert message with the required priority level
 * @param {string} msg what message to show in the alert box
 * @param {string} type the bootstrap css class from https://getbootstrap.com/docs/4.4/components/alerts/#examples
 */
function setAlertMsg(msg = "", type = "info") {
    const validTypes = ["primary", "secondary", "success", "danger", "warning", "info", "light", "dark"];
    const isValidType = validTypes.indexOf(type) >= 0;
    const newCssClass = isValidType ? `alert-${type}` : "alert-info";
    const {
        alertObj
    } = state

    if (alertObj) {
        if (isValidType) {
            // Remove every other "alert-" CSS class
            Array.from(alertObj.classList).filter(c => c.startsWith("alert-")).forEach(c => alertObj.classList.remove(c));
            alertObj.classList.add(newCssClass);
        }
        alertObj.innerHTML = msg;
        alertObj.scrollIntoView();
    } else {
        alert(msg);
    }
}

async function onStartTraininbBtnClick() {
    setAlertMsg(`<strong>Training started!</strong> Check the status in browser console.`)
    state.model = await callMain(); // available from model.js which is imported before this file
    setAlertMsg("A model is trained!", "success");
}

async function onPredictBtnClick() {
    const sepalLength = parseFloat(document.getElementById("sepal_length").value);
    const sepalWidth = parseFloat(document.getElementById("sepal_width").value);
    const petalLength = parseFloat(document.getElementById("petal_length").value);
    const petalWidth = parseFloat(document.getElementById("petal_width").value);

    const predictionBox = document.getElementById("prediction_box");
    const isTrainedModelAvailable = !!state.model

    try {
        if (!isTrainedModelAvailable) {
            setAlertMsg("Please train a model first to make predictions.");
            return;
        }
        const { prediction, confidence } = await modelPredict(state.model, [sepalLength, sepalWidth, petalLength, petalWidth]);
        document.getElementById("prediction_confidence_container").innerHTML = `${(confidence * 100).toFixed(2)}%`
        document.getElementById("prediction_container").innerHTML = prediction;

        predictionBox.classList.remove("d-none");
        predictionBox.scrollIntoView();
    } catch (error) {
        setAlertMsg("<strong>Something wrong happened!</strong> Please check browser console for details.", "danger");
        console.error(error);
    }
    
}

init();