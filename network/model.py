import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NeuralNet(nn.Module):
    '''
    Creates a neural net based on the torch framework combining CNN and RNN using Convolutional and LSTM layers

        Parameters:
            speaker (bool): Describes the datasets goal: speaker recognition (true) or speech recognition (false)
            voices (int): An integer corresponding to the ammount of voices to recognize
            duration (float): A floating number corresponding to the duration of singular fragments
            dropout (float): A floating number describing the percentage of dropout used for  the LSTM layer
    '''
    def __init__(self, speaker: bool, voices: int, duration: float, dropout: float = 0.3):
        super(NeuralNet, self).__init__()
        final_size = int(duration*32)
        # Wrapper for conv_layers
        conv_layers = []

        # CNN Component
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        conv_layers += [self.conv1, self.relu1, self.bn1]
        self.conv2 = nn.Conv2d(8, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        conv_layers += [self.conv2, self.relu2, self.bn2]
        self.conv3 = nn.Conv2d(32, final_size, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(final_size)
        conv_layers += [self.conv3, self.relu3, self.bn3]
        self.cnn = nn.Sequential(*conv_layers)

        # Adaptive Pooling
        self.ap = nn.AdaptiveAvgPool1d(output_size=1)

        # LSTM Component
        self.lstm = nn.LSTM(input_size=final_size, hidden_size=10, num_layers=10, batch_first=True, dropout=dropout)

        # FC Output Layer
        self.fc = nn.Linear(in_features=final_size, out_features=(lambda x: voices + 1 if speaker else 2)(0))

    def forward(self, x):
        # Run CNN
        x = self.cnn(x)
        # Max Pool 3D Layer may be useful here!
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        # Run LSTM
        x, _ = self.lstm(x)
        # Flatten
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        # Linear Layer
        x = self.fc(x)

        return x


def training(model, train_data, num_epochs: int = 50, lr: float = 0.001, verbose: bool = True):
    '''
    Takes a model and runs the training pipeline for the defined parameters and returns the accuracy of the training

        Parameters:
            model: A torch-based neural network to use for the training cycle
            train_data: The data to train the model on
            num_epochs (int): An integer corresponding to the amount of iterations for the training loop
            lr (float): A floating number corresponding to the learning rate for the training loop
            verbose (bool): An integer describing whether the model shall print updates to the console regularly

        Returns:
            acc (float): A floating number identifying the final accuracy of the training loop
    '''
    # Define the writer and add the graph
    model.to(DEVICE)
    time = dt.now().isoformat(' ', 'seconds').replace(":", "-")
    path = f'../../../../../../../../tmp/tensorboard/runs/speaker/{time}'
    writer = SummaryWriter(path)
    examples = iter(train_data)
    example_data, _ = examples.next()
    writer.add_graph(model, example_data.float().to(DEVICE))

    # Loss Function, Optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=int(len(train_data)),
                                                    epochs=num_epochs, anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        running_loss = 0.0
        correct_pred = 0
        total_pred = 0
        model.train()

        # Repeat per batch
        for i, data in enumerate(train_data):
            # Get Input and Target Labels and put them on the GPU
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass, backpropagation & optimization
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Stats for loss and accuracy
            running_loss += loss.item()

            # Predicted class with highest score
            _, prediction = torch.max(outputs, 1)
            # Count predictions that match the target
            correct_pred += (prediction == labels).sum().item()
            total_pred += prediction.shape[0]

        # Print stats at end of epoch
        num_batches = len(train_data)
        avg_loss = running_loss / num_batches
        acc = correct_pred/total_pred

        if verbose:
            print(f'Epoch: {epoch+1}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

        # Add the data to the writer
        writer.add_scalar('Loss', avg_loss, epoch)
        writer.add_scalar('Accuracy', acc, epoch)

    if verbose:
        print('Finished Training')

    return acc


# -------------
#   Inference
# -------------
def inference(model, inference_data, verbose: bool = True):
    '''
    Takes a model and grades its predictions on a set of data

        Parameters:
            model: A torch-based neural network to use for the test cycle
            inference_data: The data to make predictions from
            verbose (bool): An integer describing whether the model shall print updates to the console regularly

        Returns:
            acc (float): A floating number identifying the accuracy of the predictions
    '''
    correct_prediction = 0
    total_prediction = 0
    model.to(DEVICE)

    # Disable Gradient updates
    with torch.no_grad():
        model.eval()
        for data in inference_data:
            # Get input features and target labels and put them on the GPU
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            # Normalize the inputs between -1 and 1
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get prediction with highest confidence
            _, prediction = torch.max(outputs, 1)

            # Count predictions with matching labels
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        acc = correct_prediction / total_prediction

        if verbose:
            print(f'Inference on {total_prediction} total items --> Accuracy: {acc:.2f}')

    return acc


def single_inference(model, inference_data):
    '''
    Runs a single inference loop with a given model on a given singular data point. Used for live recognition

        Parameters:
            model: A torch-based neural network to use for the prediction
            inference_data: The data to make predictions from

        Returns:
            pred (int): An integer value corresponding to the predicted output with the highest confidence
    '''
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        inputs = inference_data.to(DEVICE)
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        out = model(inputs)
        _, pred = torch.max(out, 1)

        return pred


def save(model, speaker, voices, duration):
    '''
    Saves a specified model in form of a torch-readable state_dict

        Parameters:
            model: A torch-based neural network to be saved

            Other parameters are used for description of the dict title
            speaker (bool): Describes the datasets goal: speaker recognition (true) or speech recognition (false)
            voices (int): An integer corresponding to the ammount of voices to recognize
            duration (float): A floating number corresponding to the duration of singular fragments
    '''
    if speaker:
        torch.save(model.state_dict(), f'./models/new_speaker_{voices}p_{duration}s.model')
    else:
        torch.save(model.state_dict(), f'./models/new_speech_{duration}s.model')


def load(model, path: str):
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    return model