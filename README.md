# Hand Gesture Classification with Trained Model

This project demonstrates real-time hand gesture classification using a pre-trained model.

**Requirements:**

- Python 3.10
- Necessary libraries (install using `pip install -r requirements.txt`)

**Model Download:**

- Download the `model.p` file (trained model) from this link: [https://lut-my.sharepoint.com/:u:/g/personal/mahi_talukder_student_lut_fi/EagauFcTJ9RKq\_\_ozxKwZbEBs5PcS7pplqnXgB_rW2rFGA?e=T1avhq](https://lut-my.sharepoint.com/:u:/g/personal/mahi_talukder_student_lut_fi/EagauFcTJ9RKq__ozxKwZbEBs5PcS7pplqnXgB_rW2rFGA?e=T1avhq)
- Place the downloaded `model.p` file in the project's root directory.

**Running the Inference:**

1. **Install dependencies:** Execute `pip install -r requirements.txt` to install the required libraries.
   if you are using conda then you can recreate my environment using the command `conda create --name newenv --file condaenvlist.txt`
   remember to activate the newly created environment `conda activate newenv`
2. **Run the script:** Navigate to the project directory and run `python test.py` to start the real-time classification.
   to run the full user interface that lets you create word and sentence, run the `python tkinter_ui.py` file

3. **Dateset:** The dataset we used to train our model can be found here: [https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data]
