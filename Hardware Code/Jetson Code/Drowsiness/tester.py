import vlc
import time
import os

path = os.path.dirname(os.path.abspath(__file__))



player = vlc.MediaPlayer(path+"/alarm.mp3")
player.play()
time.sleep(5)
player.stop()

