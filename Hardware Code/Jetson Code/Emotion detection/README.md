# Laptop Test:

To test the model you can enter our Jupyter Notebook in Google Colab and everything is ready to run the notebook.

Link: 

## Model Creation:

Inside the [Model](./model) folder our model called "emotions.t7" already exists, which is the one I use for all tests, however the model can be trained by yourself with the code called [train.py](./train/train.py) in the folder [Train](./train).

The database that was used, is a database with 28710 images of 'Angry','Disgust','Fear','Happy','Sad','Surprised','Neutral' people in CSV format, so that the network was able to identify the state of person face, the database is in the following folder link, download the CSV file into dataset folder:

www.kaggle.com/altaga/emotions

- The model is saved in the train/model folder every 10 epoch.

Example how i train the model with VS code.
<img src="https://i.ibb.co/nsn5sSy/image.png" width="1000">

# How does it work:

The emotion detection algorithm is as follows:

- Detection that there is a face of a person behind the wheel:

<img src="https://i.ibb.co/ZMvwvfp/Face.png" width="600">

- Once we have detected the face, we cut it out of the image so that we can use them as input for our convolutional PyTorch network.

<img src="https://i.ibb.co/xDgvMBD/Neutral.png" width="600">

- The model is designed to detect the emotion of the face, this emotion will be saved in a variable to be used by our song player.

<img src="https://i.ibb.co/1Q7M3ks/image.png" width="600">

- According to the detected emotion we will randomly reproduce a song from one of our playlists:

    - If the person is angry we will play a song that generates calm
    - If the person is sad, a song for the person to be happy
    - If the person is neutral or happy we will play some of their favorite songs

Note: If the detected emotion has not changed, the playlist will continue without changing the song.