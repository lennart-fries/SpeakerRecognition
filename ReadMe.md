# Speaker Recognition
This project is used to test trained speaker recognizer models to conduct a user study

Additionally,it includes the network training and optimization for the speaker recognizer and sample audio data.

## Installation
You will need to clone this repository and then install the requirements as follows:

Instructions are for using Windows as your OS! Making environments for Linux & Mac requires a different approach.
For further instructions for your operating system, please refer to the [official documentation](https://docs.python.org/3/library/venv.html)

```
git clone <this repository>
cd speaker_recognition
python -m venv speakerrecognition
./speakerrecognition/Scripts/activate
pip install -r requirements.txt
```

## Relevant for the Study
### Prerequisities
- Python 3.7.11 or higher
- Stereomix

Visit the [Wiki](https://gitlab2.informatik.uni-wuerzburg.de/s352532/speaker-recognition/-/wikis/home) for detailled information about this

### webapp.py
This is the logic for the basic web interface for the user study

Execute it like so on a Windows machine:

```
cd speaker_recognition
./<name of your virtual env>/Scripts/activate
streamlit run webapp.py
```


## Structure
### Models
Folder contains the pre-trained and optimized model samples

### Network
Folder contains the network files for training models

### Voice
Folder containes the voice samples for training, replace for your own training

### demo.py
Allows for a quick demonstration of the functionality

Use *entire_pipeline()* for training a new model

Use *recognition()* for testing a sample model

Adjust the parameters ***SPEAKER*** to **True** for recognizing speakers or to **False** for recognizing human speech


