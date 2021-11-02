# Laptop Test:

To test the model you can enter our Jupyter Notebook in Google Colab and everything is ready to run the notebook.

Link: 

## Model Creation:

Inside the [Model](./model) folder our model called "BlinkModel.t7" already exists, which is the one I use for all tests.

However the model can be trained by yourself with the code called [train.py](./train/train.py) in the folder [Train](./train).

The database that was used, is a database with 4846 images of left and right eyes, open and closed, where approximately half are open and closed so that the network was able to identify the state of the eyes, the database is in the following folder, it is a **.zip** file unzip before starting the training:

[Database](./train/dataset/dataset_B_Eye_Images.zip)

The training has the following parameters as input.

- input image shape: (24, 24)
- validation_ratio: 0.1
- batch_size: 64
- epochs: 40
- learning rate: 0.001
- loss function: cross entropy loss
- optimizer: Adam

In the first part of the code you can modify the parameters, according to your training criteria.

Example how i train the model with VS code.
<img src="https://i.ibb.co/c1rBQvQ/image.png" width="1000">

# How does it work:

The flicker detection algorithm is as follows:

- Detection that there is a face of a person behind the wheel:

<img src="https://i.ibb.co/ZMvwvfp/Face.png" width="600">

- If a face is detected, perform eye search.

<img src="https://i.ibb.co/StK0t2x/Abiertos.png" width="600">

- Once we have detected the eyes, we cut them out of the image so that we can use them as input for our convolutional PyTorch network.

<img src="https://i.ibb.co/0FYT0DN/Abiertoss.png" width="600">

- The model is designed to detect the state of the eyes, therefore it is necessary that at least one of the eyes is detected as open so that the algorithm does not start generating alerts, if it detects that both eyes are closed for at least 2 seconds , the alert will be activated, since the security of the system is the main thing, the algorithm has a second layer of security explained below.

- Because a blink lasts approximately 350 milliseconds then a single blink will not cause problems, however once the person keeps blinking for more than 2 or 3 seconds (according to our criteria) it will mean for the system that the person is falling asleep. Not separating the eyes from the road being one of the most important rules of driving.

<img src="https://i.ibb.co/kQ12W79/alert1.png" width="600">
<img src="https://i.ibb.co/LdVD7v2/alert2.png" width="600">

- Also during the development I found a incredible use case, when one turns to look at his cell phone, the system also detects that you are not seeing the road. This being a new aspect that we will be exploring in future versions of the system to improve detection when a driver is not looking at the road and is distracted. But, for now it is one of those unintended great finds.

<img src="https://i.ibb.co/mHZ4VdX/Cel.png" width="600">
<img src="https://i.ibb.co/3k512YS/cel2.png" width="600">