from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from network.data import Data
from network.model import NeuralNet
from network import model


class Optimizer:
    '''
    Provides a full optimization pipeline based on Bayesian Optimization

        Parameters:
            path (str): A string corresponding to the path of the data
            speaker (bool): Describes the datasets goal: speaker recognition (true) or speech recognition (false)
            voices (int): An integer corresponding to the ammount of voices to recognize
            duration (float): A floating number corresponding to the duration of singular fragments
            threshold (float): A floating number corresponding to the minimum accuracy set for recognizing nets as good
    '''
    def __init__(self, path: str, speaker: bool, voices: int, duration: float,  threshold: float = 0.95,
                 verbose: bool = True):
        self.path = path
        self.threshold = threshold
        self.voices = voices
        self.duration = duration
        self.speaker = speaker
        self.verbose = verbose

    def _train(self, lr: float, epochs: float, dropout: float) -> float:
        '''
        Provides a method to optimize for the bayesian loop

            Parameters:
                lr (float): A floating number corresponding to the learning rate for the training loop
                epochs (int): An integer corresponding to the amount of iterations for the training loop
                dropout (float): A floating number describing the percentage of dropout used for  the LSTM layer

            Returns:
                test_acc (float): A floating number identifying the accuracy, used for optimization
        '''
        loader = Data(self.path)
        train_data, test_data = loader.prepare_data(speaker=self.speaker, voices=self.voices, duration=self.duration)

        network = NeuralNet(speaker=self.speaker, voices=self.voices, dropout=dropout, duration=self.duration)
        train_acc = model.training(network, train_data, num_epochs=int(epochs), lr=lr, verbose=False)
        test_acc = model.inference(network, test_data)

        if self.verbose:
            print(f'Accuracy of the training: {train_acc * 100:.4f}%')
            print(f'Accuracy of the validation: {test_acc * 100:.4f}%')

        if test_acc > self.threshold:
            model.save(network, speaker=self.speaker, voices=self.voices, duration=self.duration)
            self.threshold = test_acc
            if self.verbose:
                print("New threshold at:", self.threshold)

        return test_acc

    def optimize(self, iterations: int, log_path: str = None):
        '''
        Optimizes a network on the given parameters of the class

            Parameters:
                 iterations (int): An integer value corresponding to the amount of iterations of the optimizer, where
                    one iteration corresponds to one training cycle with n epochs
                 log_path (str): A string corresponding to the path for logs to be read from and save in
        '''
        pbounds = {'lr': (0.001, 0.01), 'epochs': (20, 100), 'dropout': (0.1, 0.7)}

        optimizer = BayesianOptimization(
            f=self._train,
            pbounds=pbounds,
            random_state=42
        )

        if log_path is not None:
            load_logs(optimizer, logs=log_path)

            print("The optimizer is now aware of {} points.".format(len(optimizer.space)))

            logger = JSONLogger(path=log_path)
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        optimizer.maximize(
            init_points=2,
            n_iter=iterations
        )

        print(optimizer.max)

        for i, res in enumerate(optimizer.res):
            print('Iteration {}: \n\t{}'.format(i, res))