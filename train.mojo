from python import Python
from mojo.net import Net

fn main() raises:
    try:
        var torch = Python.import_module("torch")
        var sklearn = Python.import_module("sklearn.model_selection")
        var train_test_split = sklearn.train_test_split
        var optim = torch.optim
        var model = Net()
        var seed_value = 42
        torch.manual_seed(seed_value) 
        var input_data = torch.randn(64, 2)  # example input tensor with batch size 64 and input size 2
        var target_data = torch.randint(0, 2, (64,))  # example target tensor with batch size 64 and 2 classes
        var split_result = train_test_split(input_data, target_data, test_size=0.2, random_state=seed_value)
        var train_inputs = split_result[0]
        var test_inputs = split_result[1]
        var train_targets = split_result[2]
        var test_targets = split_result[3]
        var criterion = torch.nn.CrossEntropyLoss()
        var optimizer = optim.Adam(model.model.parameters(), lr=0.01)
        # training loop
        var num_epochs = 100
        for epoch in range(num_epochs):
            model.model.train()  # set the model to training mode
            optimizer.zero_grad()  # zero the gradients
            var output = model.forward(train_inputs)  # forward pass
            var loss = criterion(output, train_targets)  # calculate the loss
            model.backward(loss)  # backward pass
            optimizer.step()  # update weights
            print('epoch, loss:', epoch + 1, num_epochs, loss.item())
        torch.save(model.model.state_dict(), "model.pth")
        # evaluate the model on test data
        model.model.eval()  # set the model to evaluation mode
        var test_output = model.forward(test_inputs)  # forward pass on test data
        var test_loss = criterion(test_output, test_targets)  # calculate test loss
        print('test loss:', test_loss.item())

    except e:
        print("error during execution:", e)
