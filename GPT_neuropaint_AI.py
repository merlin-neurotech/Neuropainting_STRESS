import UnicornPy
from neurol import streams
from neurol.connect_device import get_lsl_EEG_inlets
from neurol.BCI import generic_BCI, automl_BCI
from neurol import BCI_tools
from neurol.models import classification_tools
from sys import exit
from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import openai
import time

openai.api_key = ''

class MyGPTTherapist:
    def __init__(self, run_length=60, interval_length=10, verbose=False):
        self.health = 0 # Start with neutral health. healthy = health>0, unhealthy = health<0
        self.health_sum = []
        self.health_average = None
        self.verbose = verbose
        self.start_time = None
        self.interval_start_time = None
        self.run_length = run_length
        self.interval_length = interval_length
        self.timer_start = False

    def updateHealth(self, EEG_output):

        if not self.timer_start:
            self.start_time = time.time()
            self.interval_start_time = self.start_time
            self.timer_start = True

        current_time = time.time()

        if abs(current_time - self.interval_start_time) >= self.interval_length: 
            therapist.computeAverage()
            self.printHealth()
            self.getCompletion()
            robot_UI(self.health_average)
            EEG = [EEG1, EEG2, EEG3, EEG4, EEG5, EEG6, EEG7, EEG8]
            plot_waves(xs, EEG)
            self.interval_start_time = time.time()

        if abs(current_time - self.start_time) >= self.run_length: 
            print("Great study sesh, Scott! Goodbye!")
            exit()

        if EEG_output == 'LOW' and self.health < 10:
            self.health += +1
        elif EEG_output == 'HIGH' and self.health > -10:
            self.health += -1

        self.health_sum.append(self.health)

    def computeAverage(self):
        self.health_average = int(sum(self.health_sum)/len(self.health_sum))

    def printHealth(self):
        print("Your current stress level is " + str(self.health_average) + " on a scale between -10 and 10.")
        if self.verbose: 
            print(self.health_sum)
    
    def getCompletion(self):
        messages = [
                    {"role": "system",
                     "content": "You will generate a comforting response based on the \
                     number between -10 and 10 provided. This number is an indication of the persons beta brain waves. -10 corresponds to someone who has a high beta wave output and is therefore feeling anxious and \
                         stressed. 0 corresponds to someone who feeling relatively normal, but could always use positive reinforcement. 10 corresponds to someone who is \
                             very mental healthy. Keep the responses short, and don't mention the scale or beta waves."},
                    {"role": "user", "content": "Number: " + str(self.health_average)}]
        
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                                messages=messages, 
                                                temperature=0)
        
        print(response.choices[0].message["content"] + "\n")

clb = lambda stream:  BCI_tools.band_power_calibrator(stream, ['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8'], 'unicorn', bands=['beta'],
                                                        percentile=5, recording_length=10, epoch_len=1, inter_window_interval=0.25)

gen_tfrm = lambda buffer, clb_info: BCI_tools.band_power_transformer(buffer, clb_info, ['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8'], 'unicorn', bands=['beta'],
                                                        epoch_len=1)

global xs, EEG1, EEG2, EEG3, EEG4, EEG5, EEG6, EEG7, EEG8                                                   
xs = []
EEG1 = []
EEG2 = []
EEG3 = []
EEG4 = []
EEG5 = []
EEG6 = []
EEG7 = []
EEG8 = []

def clf(clf_input, clb_info):

    clf_input = clf_input[:clb_info.shape[0]]

    #EEG1.append(clf_input[0])
    EEG1.append(clf_input[0])
    EEG2.append(clf_input[1])
    EEG3.append(clf_input[2])
    EEG4.append(clf_input[3])
    EEG5.append(clf_input[4])
    EEG6.append(clf_input[5])
    EEG7.append(clf_input[6])
    EEG8.append(clf_input[7])
    xs.append(len(EEG1))

    binary_label = classification_tools.threshold_clf(clf_input, clb_info, clf_consolidator='all')

    label = classification_tools.decode_prediction(
    binary_label, {True: 'HIGH', False: 'LOW'})
 
    return label

def plot_waves(x, EEG):
    names = ['EEG1','EEG2','EEG3','EEG4','EEG5','EEG6','EEG7','EEG8']
    plt.figure()
    for i in range(8):
        plt.plot(x, EEG[i], label=names[i])
    plt.xlabel('Measurements (time)')
    plt.ylabel('Beta Wave Activity')
    plt.show()

def robot_UI(health):
    if health >= -10 and health < -3: image_plotter("sad_robot.jpg")
    if health >= -3 and health <= 3: image_plotter("neutral_robot.jpg")
    if health > 3 and health <= 10: image_plotter("happy_robot.jpg")

def image_plotter(fileName):
    img = mpimg.imread(fileName)
    plt.imshow(img)
    plt.show()

streams1 = resolve_stream("name='Unicorn'")
inlet = StreamInlet(streams1[0])
stream = streams.lsl_stream(inlet, buffer_length=1024)

def intro(q1, prompt):
    messages = [
                {"role": "system", "content": "You have just asked me the following question: " + q1 + ". Provide a short reply to the users input without asking a follow up question. Type it out slowly."},
                {"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                            messages=messages, 
                                            temperature=0)

    print(response.choices[0].message["content"] + "\n")

q1 = "Hey Scott, how are you doing today?"
prompt1 = input("Hey Scott, how are you doing today?\n")
intro(q1, prompt1)
prompt2 = input("\nWhat are doing today?\n")
intro("\nWhat are doing today?\n", prompt2)
prompt3 = input("\nHow long are you working for?\n")
prompt4 = input("\nAnd how long do you want stress level updates?\n")
print("Great, thanks Scott! Good luck!")

therapist = MyGPTTherapist(run_length=int(prompt3), interval_length=int(prompt4), verbose=False)
GPT_Generic = generic_BCI(clf, transformer=gen_tfrm, action=therapist.updateHealth, calibrator=clb)
GPT_Generic.calibrate(stream)
GPT_Generic.run(stream)



