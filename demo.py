from network import model
from network.data import Data
from network.model import NeuralNet
from network.optimizer import Optimizer
from network.recognition import LiveRecognition

# HYPERPARAMS
PATH = './voice/'
SPEAKER = True
VOICES = 4
DURATION = 1
THRESHOLD = 0.94
SAMPLE_RATE = 44100
EXAMPLE = ['./models/paragon_speech.model', './models/optimized-speaker_2s.model']
SESSION = './models/session_speaker-1.5s.model'


def entire_pipeline():
    loader = Data(PATH)
    network = NeuralNet(SPEAKER, VOICES, DURATION)
    optim = Optimizer(PATH, SPEAKER, VOICES, DURATION, THRESHOLD)

    train, test = loader.prepare_data(SPEAKER, VOICES, DURATION)
    acc_train = model.training(network, train)
    acc_test = model.inference(network, test)


    print(f'Test Run -> Train: {acc_train*100:.2f}% \nTest: {acc_test*100:.2f}%')
    print(f'Now optimizing...')
    optim.optimize(50)

    #if acc_test > THRESHOLD:
        #model.save(network, SPEAKER, VOICES, DURATION)
        #recognizer = LiveRecognition(network, SPEAKER, DURATION)
        #recognizer.loop()


def recognition():
    network = NeuralNet(SPEAKER, VOICES, DURATION)
    if SPEAKER:
        network = model.load(network, SESSION)
    else:
        network = model.load(network, EXAMPLE[0])

    recognizer = LiveRecognition(network, SPEAKER, DURATION)
    recognizer.loop()

entire_pipeline()
#recognition()