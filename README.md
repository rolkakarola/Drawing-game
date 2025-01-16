# Virtual Painter

## Description
The project utlizes [OpenCv library](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) and [Mediapipe framework](https://ai.google.dev/edge/mediapipe/solutions/guide) to create a real time virtual painter.
The script uses hand tracking to enable drawing on the screen with the index finger, change color and clean screen.

## Environment Setup

Environment should have the following requirements (also defined in requirments.txt):

- **Python:** 3.11
- **mediapipe:** 0.10.20
- **numpy:**  1.26.4
- **opencv-contrib-python:** 4.10.0.84
- **scikit-learn:**  1.6.1

### Installation

1. **Python 3.11**: Ensure Python 3.11 is installed on your system. If not, download and install it from the [official Python website](https://www.python.org/downloads/) or use homebrew (Linux/macOS):
   ```bash
   brew install python@3.11

2.Clone the repository to your local machine
    ```
      cd path/to/your/directory
      git clone https://github.com/rolkakarola/Virtual-Painter
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt



### Running the script

#### Steps to execute

##  Project Structure

    Virtual-Painter/
    │
    ├── src/           
    │   │
    │   ├── collect_imgs.py         
    │   ├── create_dataset.py           
    │   ├── data.pickle     
    │   ├── main.py  
    │   ├── train_classifier.py
    │   └── model.p            
    ├── requirements.txt    
    ├── README.md            
    └── results/             

        







