# Project description
Sepsis is a life-threatening condition that occurs when the body's response to infection causes tissue damage, organ failure, or death. In the U.S., nearly 1.7 million people develop sepsis and 270,000 people die from sepsis each year; over one third of people who die in U.S. hospitals have sepsis (CDC). Internationally, an estimated 30 million people develop sepsis and 6 million people die from sepsis each year; an estimated 4.2 million newborns and children are affected (WHO). Early detection and antibiotic treatment of sepsis are critical for improving sepsis outcomes, where each hour of delayed treatment has been associated with roughly an 4-8% increase in mortality (Kumar et al., 2006; Seymour et al., 2017). To help address this problem, clinicians have proposed new definitions for sepsis (Singer et al., 2016), but the fundamental need to detect and treat sepsis early still remains, and basic questions about the limits of early detection remain unanswered.

This project explores numerous methods for performing an accurate early prediction of sepsis given time-series data.

This formed the [2019 Physionet Computing in Cardiology Challenge.](https://physionet.org/content/challenge-2019/1.0.0/)

# Data availability
The two training sets have been sourced from the 2019 Physionet Computing in Cardiology Challenge and are freely available.

# Requirements
- A device with a GPU that supports CUDA 11/12 is strongly recommended.
-  A  Linux device is necessary for TensorFlow GPU support.
Please ensure you are running a Python version between 3.9 and 3.12, as TensorFlow is only compatible with these versions.
- Make sure that your version of pip is up to date; otherwise, you risk installing older, incompatible packages.

# Setup instructions for development
- Clone the repository with the following command
```
git clone https://github.com/ciarakbrown/Data-Science.git
```
- Create a virtual environment at the top-level of this project 
```
python -m venv .venv
```
and activate it with
```
source .venv/bin/activate
```
- Install the Python dependencies from requirements.txt
```
pip install -r requirements.txt
```
- Download the dataset
```
python get_data.py
```
- Preprocess the data
```
python preprocessing.py
```
- Run the program
```
python runner.py
```

# Git assistance
To start, clone the repository, either with an SSH key or through HTTP
```
git clone https://github.com/ciarakbrown/Data-Science.git
```
When starting development, create a new branch with
```
git branch branch_name
```
To move into this branch, use
```
git checkout branch_name
```
Alternatively, you can make a branch on GitHub and use this command to check it out locally
```
git switch branch_name
```
When you make any changes, use this to stage and commit your files to a local branch
```
git add .
git commit -m "commit message"
```
To push your changes to GitHub
```
git push
```
Make sure you stay up to date with changes frequently to reduce conflicts
```
git fetch origin
git merge origin/main
```
To stay up to date with your branch (say two people are working on the same branch)
```
git pull
```
When resolving merge conflicts, you can use a visual tool called GitKraken.

