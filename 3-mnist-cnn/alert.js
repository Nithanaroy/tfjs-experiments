class Alert {

    /**
     * Manages the alert container
     * @param {HTMLElement} alertDiv Instance of a div HTML element
     */
    constructor(alertDiv) {
        if (!alertDiv) {
            throw Error("An HTML UI container instance is needed")
        }
        this.htmlContainer = alertDiv;
    }

    /**
     * helper method to sets an alert message with the required priority level
     * @param {string} msg what message to show in the alert box
     * @param {string} type the bootstrap css class from https://getbootstrap.com/docs/4.4/components/alerts/#examples
     */
    showMsg(msg = "", type = "info") {
        const validTypes = ["primary", "secondary", "success", "danger", "warning", "info", "light", "dark"];
        const isValidType = validTypes.indexOf(type) >= 0;
        const newCssClass = isValidType ? `alert-${type}` : "alert-info";
        const alertObj = this.htmlContainer;

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

}