# Forza AI

A work-in-progress program to detect roadway boundaries and obstacles in Forza Horizon 4.

## Requirements

- Python 3.6
- All modules in requirements.txt
- [RetinaNet](https://github.com/fizyr/keras-retinanet) for object recognition
  - Download retinanet.h5 and move it to the project's main directory

## How to Run

2 monitors are required for the program to work properly.

- Start Forza Horizon 4
- `python3 run.py` and move the PyGame window to a secondary display
- Focus the Forza game window
  - The pause menu will automatically be closed and the program will start capturing screenshots after a few seconds.

## Screenshots

### Lane Detection

![](https://cdn.discordapp.com/attachments/376375897109954560/777567162076102746/python_2020-11-15_11-12-28.png)

![](https://cdn.discordapp.com/attachments/376375897109954560/777567175149355038/python_2020-11-15_11-13-47.png)

### Object Recognition

![](https://cdn.discordapp.com/attachments/376375897109954560/777567761848860692/unknown.png)

![](https://cdn.discordapp.com/attachments/599592659379683343/758124020830044190/unknown.png)