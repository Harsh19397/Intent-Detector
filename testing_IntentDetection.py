#Try to fix or just remove it from the folder
import intentDetector as int_det

#Testing
def get_intent(text):
    #text = "Please search about Jarvis on google"
    pred = int_det.predictions(text)
    print("Intent Detected: "+ int_det.get_final_output(pred, int_det.unique_intent))
