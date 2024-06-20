from python import Python

struct Net:
    """
    Simple neural network for classification.

    Attributes:
        model: Sequential model containing layers of the network.
    """
    var model: PythonObject

    fn __init__(inout self):
        """
        Initializes the neural network layers.
        """
        try:
            var torch = Python.import_module("torch")
            var nn = torch.nn
            var device = torch.device("cpu")  # assuming CPU for simplicity
            self.model = nn.Sequential(
                nn.Linear(2, 5),
                nn.ReLU(), 
                nn.Linear(5, 5),
                nn.ReLU(),
                nn.Linear(5, 2)
            ).to(device)
        except:
            print("error importing PyTorch")
            self.model = None

    fn __copyinit__(inout self, other: Net):
        """
        Initializes a copy of Net from another instance.

        Args:
            other: Another instance of Net.
        """
        self.model = other.model

    fn forward(self, x: PythonObject) raises -> PythonObject:
        """
        Defines the forward pass of the network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the network.
        """
        try:
            return self.model(x)
        except:
            print("error during forward pass")
            raise

    fn backward(self, loss: PythonObject) raises:
        """
        Performs backward pass and updates gradients.

        Args:
            loss: Loss tensor calculated during forward pass.
        """
        try:
            loss.backward()
        except:
            print("error during backward pass")
            raise

    fn predict_probabilities(self, x: PythonObject) raises -> PythonObject:
        """
        Calculates class probabilities using softmax after forward pass.

        Args:
            x: Input tensor.

        Returns:
            Probability distribution over classes.
        """
        try:
            var torch = Python.import_module("torch")
            var F = torch.nn.functional
            var x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Convert input to tensor
            var logits = self.model(x_tensor)
            var probabilities = F.softmax(logits, dim=1)
            return probabilities
        except:
            print("error calculating probabilities")
            raise

    fn predict_number(self, x: PythonObject) raises -> Int:
        """
        Predicts the class label using the trained model.

        Args:
            x: Input tensor.

        Returns:
            Predicted class label.
        """
        try:
            var torch = Python.import_module("torch")
            var x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # convert input to tensor
            var logits = self.model(x_tensor)
            var prediction = logits.argmax(dim=1).item()  # get the index of the max probability
            return prediction
        except:
            print("error during prediction")
            raise
