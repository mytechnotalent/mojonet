from python import Python
from mojo.net import Net
from mojo.model import save_model, load_model

fn main() raises:
    try:
        # load the model back for testing
        var loaded_model = load_model("net.pth")

        # example prediction using a sample input
        var sample_input = [1.0, 2.0]  # replace with your own input data
        var prediction = loaded_model.predict_number(sample_input)
        var probabilities = loaded_model.predict_probabilities(sample_input)
        print('prediction for the sample input:', prediction)
        print('probabilities:', probabilities.tolist())

    except e:
        print("error during execution:", e)
        raise e
