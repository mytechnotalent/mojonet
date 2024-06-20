from python import Python
from mojo.net import Net 
from testing import assert_true

fn test_net_init() raises:
    """
    Test the initialization of the Net class.
    """
    var net: Net = Net()
    assert_true(net.model is not None, "Model should be initialized")

fn test_net_forward() raises:
    """
    Test the forward pass of the Net class.
    """
    var net: Net = Net()
    var torch: PythonObject = Python.import_module("torch")
    var train_inputs: PythonObject = torch.tensor([0.2656, -0.0026])
    var output: PythonObject = net.forward(train_inputs)
    assert_true(output is not None, "Forward pass should produce an output")

fn test_net_backward() raises:
    """
    Test the backward pass of the Net class.
    """
    var net: Net = Net()
    var torch: PythonObject = Python.import_module("torch")
    var nn: PythonObject = torch.nn
    var criterion: PythonObject = nn.MSELoss()
    var train_inputs: PythonObject = torch.tensor([0.0, 1.0])
    var train_targets: PythonObject = torch.tensor([0.5, -0.5])
    var output: PythonObject = net.forward(train_inputs)
    var loss: PythonObject = criterion(output, train_targets)
    var backward_output: PythonObject = net.backward(loss)
    assert_true(backward_output is None, "Backward pass should not produce an error")
    
fn test_net_predict_probabilities() raises:
    """
    Test the predict_probabilities method of the Net class.
    """
    var net: Net = Net()
    var probabilities: PythonObject = net.predict_probabilities([0.5, -0.5])
    assert_true(probabilities is not None, "Predict probabilities should produce an output")

fn test_net_predict_number() raises:
    """
    Test the predict_number method of the Net class.
    """
    var net: Net = Net()
    var prediction: Int = net.predict_number([0.5, -0.5])
    assert_true(0 <= prediction < 2, "Prediction should be between 0 and 1")

fn main() raises:
    try:
        test_net_init()
        test_net_forward()
        test_net_backward()
        test_net_predict_probabilities()
        test_net_predict_number()
    except e:
        print(e)
