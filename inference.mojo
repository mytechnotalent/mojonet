from python import Python
from mojo.net import Net

fn main() raises:
    try:
        var model = Net()
        var torch = Python.import_module("torch")
        model.model.load_state_dict(torch.load("model.pth"))
        var sample_input = [1.0, 2.0]
        var prediction = model.predict_number(sample_input)
        var probabilities = model.predict_probabilities(sample_input)
        print('prediction for the sample input:', prediction)
        print('probabilities:', probabilities.tolist())

    except e:
        print("error during execution:", e)
        raise e
