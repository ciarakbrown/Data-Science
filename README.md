# Project description
Sepsis is a life-threatening condition that occurs when the body's response to infection causes tissue damage, organ failure, or death. In the U.S., nearly 1.7 million people develop sepsis and 270,000 people die from sepsis each year; over one third of people who die in U.S. hospitals have sepsis (CDC). Internationally, an estimated 30 million people develop sepsis and 6 million people die from sepsis each year; an estimated 4.2 million newborns and children are affected (WHO). Early detection and antibiotic treatment of sepsis are critical for improving sepsis outcomes, where each hour of delayed treatment has been associated with roughly an 4-8% increase in mortality (Kumar et al., 2006; Seymour et al., 2017). To help address this problem, clinicians have proposed new definitions for sepsis (Singer et al., 2016), but the fundamental need to detect and treat sepsis early still remains, and basic questions about the limits of early detection remain unanswered.

This project explores numerous methods for performing an accurate early prediction of sepsis given time-series data.

This formed the [2019 Physionet Computing in Cardiology Challenge.](https://physionet.org/content/challenge-2019/1.0.0/)

# Data availability
The two training sets have been sourced from the 2019 Physionet Computing in Cardiology Challenge and are freely available.

# Requirements
A device with a GPU that supports CUDA 11/12 is strongly recommended. For tensorflow GPU support, a Linux device is necessary.
Please ensure you are running a Python version between 3.9 - 3.12, as tensorflow is only compatible with these versions. Make sure that
your version of pip is up to date, otherwise you run the risk of installing older incompatible packages.

# Setup instructions for development
- Clone the repository with the following command
```
git clone https://github.com/ciarakbrown/Data-Science.git
```
- Create a virtual environment at the top-level of this project and activate it with
```
python -m venv .venv
source .venv/bin/activate
```
- Install the Python dependencies from requirements.txt
```
pip install -r requirements.txt
```
