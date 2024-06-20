from python import Python
from mojo.net import Net

fn save_model(model: PythonObject, path: String) raises:
    """
    Save the state dictionary of a PyTorch model to a file.

    Args:
        model: The PyTorch model to save.
        path: The file path where the model state dictionary will be saved.
    
    Raises:
        Any error encountered during saving.
    """
    try:
        var torch = Python.import_module("torch")
        torch.save(model.state_dict(), path)
    except e:
        print("error saving model:", e)
        raise e

fn load_model(path: String) -> Net:
    """
    Load a trained model from a file and return an instance of Net.

    Args:
        path: The file path from which to load the model.

    Returns:
        Net: An instance of the Net class loaded with the model parameters.
    """
    var model = Net()
    try:
        var torch = Python.import_module("torch")
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        return model
    except e:
        print("error loading model:", e)
        return model

fn predict_number(model: Net, input_data: PythonObject) -> PythonObject:
    """
    Perform a prediction using a trained model.

    Args:
        model: An instance of the Net class containing the trained model.
        input_data: Input data for prediction.

    Returns:
        PythonObject: The predicted class index.
    """
    try:
        var torch = Python.import_module("torch")

        # Convert input_data to torch tensor
        var tensor = torch.tensor(input_data, dtype=torch.float32)

        # Forward pass through the model
        var output = model.forward(tensor.unsqueeze(0))  # Unsqueeze to add batch dimension
        var prediction = output.argmax(dim=1).item()  # Get the predicted class index

        return prediction

    except e:
        print("error during number prediction:", e)
        return
