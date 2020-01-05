class Constants {
    static MODEL_DISK_PATH() {
        return `indexeddb://${Constants.MODEL_NAME()}`;
    }
    static MODEL_NAME() {
        return "my-model-1";
    }
}