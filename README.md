# Hand Gesture Classification with Trained Model

This is continuation of our previous [project](https://github.com/LordMahi19/signlang) where we have trained a simple machine learning model that can predict a few sign language charachters. In the previous project we generated a few data ourselves which we used to train the model. But in this project we used a very large dataset to train a model with the same technique. We are adding a few new techniques to work with large dataset. 
- We have used pythons built in multiprocessing function to allocate all of our cpu cores when creating the dataset. Since python is by default a single core process, this new technique allows us process the data significantly faster.
- We have also made a UI out of this model with pythons tkinter library. This UI allows us to make words and sentences with the model. 

This project demonstrates real-time hand gesture classification using a pre-trained model. Follow the steps below to run the model on your machine.

**Requirements:**

- Install git if its not installed already: [git download page](https://git-scm.com/downloads)
- Navigate to a directory wheer you want to download this project. Open the termianl for that directory and run the following commands
- `git clone https://github.com/LordMahi19/ASL-detection.git`
- `cd "ASL-detection`
- `pip install -r requirements.txt`
- If you are using conda then you can recreate my environment using the command `conda create --name newenv --file condaenvlist.txt`
   remember to activate the newly created environment `conda activate newenv`

**Model Download:**

- Download the `model.p` file (trained model) from this link: [model.p](https://lut-my.sharepoint.com/:u:/g/personal/mahi_talukder_student_lut_fi/EagauFcTJ9RKq__ozxKwZbEBs5PcS7pplqnXgB_rW2rFGA?e=T1avhq)
- Place the downloaded `model.p` file in the project's root directory. (inside the "ASL-detection" folder.

**Running the Inference:**
   
1. **Run the script:** Navigate to the project directory in terminal and run `python test.py` to start the real-time classification.
   to run the full user interface that lets you create word and sentence, run `python tkinter_ui.py`.

**Dateset:** The dataset we used to train our model can be found here: [Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data)
