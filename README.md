# Intent-Detector
Intent detection module deals with the intent of the user. A user can make a request for a task in several ways in English language. Hence it won’t be a wise decision to hardcode the vocab used by a user to get some task completed. In order to deal with this problem, a neural network has been trained on a dataset of some user commands. This neural network will predict the intent of the user no matter what way the user uses to command the assistant for a specific task.

## How to use
1. Clone this repository
2. Install the dependencies
3. Run the command "python intentDetector.py"
4. Then use the trained model "IntentDetectormodel.h5" to predict the intent using the prediction() function.

A very high accuracy of 96.79% has been achieved on the training set and 83% on the unseen data.
