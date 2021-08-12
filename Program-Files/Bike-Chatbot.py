import aiml, csv, os, pandas, random, re, requests, tensorflow, \
    tkinter, uuid, wikipedia
    
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, \
    SpeechSynthesizer
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from tkinter import filedialog

## Needed to use pip to install aiml, azure.cognitiveservices.speech, matplotlib,
## sklearn, tensorflow, textblob, tkinter, uuid and wikipedia modules


def Normalize_String(myString):
    myString = myString.lower()
    myString = re.sub("[\W_]", ' ', myString)
    return myString


def Load_Model():
    # Uses CNN-Model_256-Dense_Vehicles-Dataset_100-Epochs_0-1-Zoom
    filepath = "Required-Files/CNN-Model.h5"
    if os.path.isfile(filepath):
        cnnModel = tensorflow.keras.models.load_model(filepath)
        return cnnModel
    return None
    

def Load_Image():
    ## Loads an image using a tkinter GUI
    root = tkinter.Tk()
    root.withdraw()
    root.update()
    filepath = filedialog.askopenfilename()
    if filepath:
        print("Loaded: " + filepath)
        imageSize = (256, 256)
        image = tensorflow.keras.preprocessing.image.\
            load_img(filepath, target_size = imageSize)
        imageArray = tensorflow.keras.preprocessing.image.img_to_array(image)
        imageArray = tensorflow.expand_dims(imageArray, 0)
        return imageArray
    return None
    
    
def Get_Class_Names():
    ## Loads the class names from a txt file
    filepath = "Required-Files/Classes.txt"
    if os.path.isfile(filepath):
        classFile = open(filepath, 'r')
        classList = []
        lines = classFile.readlines()
        for line in lines:
            classList.append(line)
            classFile.close()
        return classList
    return None
    

def Predict_Image_Class(image, cnnModel):
    ## Predicts the class based on the model and loaded image
    predictions = cnnModel.predict(image)
    prediction = predictions.argmax(axis = -1)
    predictedClass = int(prediction)
    return predictedClass
    

def Evaluate_Prediction(userInput, cnnModel, classList):
    ## Loads the image and predicts its class against the existing list
    image = Load_Image()
    if image != None:
        predictedClass = Predict_Image_Class(image, cnnModel)
        return ("\nThat is a " + classList[predictedClass])
    else:
        return ("No image loaded. Let's continue where we left off.")


def Fetch_Info(myString, fileName):
    ## Assumes the input is already normalized and had punctuation removed
    bestMatchVal = 0
    bestRow = -1
    with open(fileName,'r', newline='\n') as csvFile:   
        csvReader = csv.reader(csvFile, delimiter = ',')
        for row in csvReader:
            csvSentence = row[0].split(' ')
            csvSentence.append(myString)
            
            ## Vectorizes the different words in the row
            vectorizer = TfidfVectorizer(stop_words = "english")
            wordVector = vectorizer.fit_transform(csvSentence)
            ## Calculates the cosine similarity of the input and the row            
            cosSimValues = cosine_similarity(wordVector[-1], wordVector)            
            matchedResponse = cosSimValues.flatten()
            matchedResponse.sort()
            
            ## Check to see how well the row matched the input
            if matchedResponse[-2] != 0:
                if matchedResponse[-2] > bestMatchVal:
                    bestMatchVal = matchedResponse[-2]
                    bestRow = row[1:]
                    
        csvFile.close()
    return(bestRow)


def WikiMedia_API_Request(url):
    ## Makes the API request and gets the JSON response
    raw_page_data = requests.get(url).json()
    
    ## Formats and parses the response, fixing the spaces
    page_data = raw_page_data["parse"]["wikitext"]["*"]
    page_data = page_data.replace("&nbsp;", '')

    ## Searches the string for the start of an annotation hyperlink and saves that index
    link_locations = []
    for m in re.finditer('<', page_data):
        link_locations.append(m.start())

    ## Removes the annotation links
    for i in range(1, len(link_locations)):
        start = link_locations[i - 1]
        stop = link_locations[i] + 5
        page_data = page_data[0: start:] + page_data[stop + 1::]
    return page_data


def Answer_Query(userInput):    
    ## Returns a response based on the csv file
    response = Fetch_Info(userInput,"Required-Files/Q&A.csv")
    if response != -1:
        if "Wikipedia request:" in response[0]:
            ## Uses the wikipedia or wikimedia api using the pre-made query
            try:
                response = response[0].replace("Wikipedia request: ", '')
                if "https:" in response:
                    return WikiMedia_API_Request(response)
                else:
                    return wikipedia.summary(response)
            except:
                return ("An error reaching Wikipedia has occured")
        else:
            return response[0]
    else:
        return ("I am sorry but I can't find an answer. "
                "Please ask something else")


def Get_Suggestion(userInput):
    ## Returns a suggestion for a bike based on a style
    response = Fetch_Info(userInput, "Required-Files/Bike-List.csv")
    if response != -1:
        choice = random.randint(1, (len(response) - 1))   
        return response[choice]
    else:
        return ("Unfortunately I can not find a recommendation for that style"
              "of bike. Currently I only have suggestions for adventure"
              "bikes, cruisers, dirt bikes, mopeds, naked bikes,"
              "retro/classic bikes and sport bikes.")
 
        
def Load_Knowledgebase(knowBase):
    ## Check that file exists
    filepath = "Required-Files/Knowledgebase.csv"
    if os.path.isfile(filepath):
        data = pandas.read_csv(filepath, header = None)
        [knowBase.append(read_expr(row)) for row in data[0]]
        ## If integrity of knowledgebase is compromised it always returns True
        if ResolutionProver().prove(read_expr("car" + "(horse)"), knowBase):
            print("Error in the knowledgebase")
            return False
        return True
    else:
        print("Knowledgebase file not found!")
        return False
    

def Add_To_Knowledgebase(obj, subj, knowBase):
    ## Does some basic formatting on the strings
    obj = obj.replace(' ', '_')
    subj = subj.replace(' ', '_')
    strExpr = (subj + " (" + obj + ')').lower()
    ## Checks to see if the statement is contradicted or is already known 
    if (ResolutionProver().prove(read_expr('-' + strExpr), knowBase)):
        return ("I am sorry but that contradicts my knowledgebase!")
    elif (ResolutionProver().prove(read_expr(strExpr), knowBase)):
        return ("I already know that! Thanks anyway")
    else:
        ## Appends the knew logic statement to the array and the file
        knowBase.append(read_expr(strExpr))
        filepath = "Required-Files/Knowledgebase.csv"
        if os.path.isfile(filepath):
            with open(filepath, 'a') as knowFile:
                knowFile.write('"' + strExpr + '"' +' \n')
        return ("OK, I will remember that! Thank you.")
  
                
def Confirm_Logic_Statement(obj, subj, knowBase):
    ## Does some basic formatting on the strings
    obj = obj.replace(' ', '_')
    subj = subj.replace(' ', '_')
    expr = read_expr((subj + '(' + obj + ')').lower())
    result = ResolutionProver().prove(expr, knowBase)
    ## Checks to see if the statement is outright true
    if result == True:
        return ("That's true.")
    elif result == False:
        ## Checks to see if the statement is outright false or unknown
        expr = read_expr('-' + subj + '(' + obj + ')')
        result = ResolutionProver().prove(expr, knowBase)
        if result == True:
            return ("I'm afraid that's not true.")
        else:
            return ("Unfortunately I don't know. It could be true or false")
      
        
def Load_Azure_Details():
    fileName = "Required-Files/Azure-Details.txt"
    azureDetails = []
    ## Assumes that the file is key then endpoint and then region on separate lines
    try:
        with open(fileName, 'r') as azureFile:
            for line in azureFile:
                azureDetails.append(line.rstrip())
    except:
        print("\nUnfortunately, the Microsoft Azure services such as",
              "translation and text-to-speech will be unavailable due to",
              "missing credentials.")
        azureDetails = "Unavailable"
    return azureDetails


def Translate_Text(oldLanguage, newLanguage, azureDetails, text):
    if azureDetails == "Unavailable":
        return text
    ## API request url
    apiPath = "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0"
    apiParameters = '&from={}&to={}'.format(oldLanguage, newLanguage)
    apiUrl = apiPath + apiParameters
    
    ## Constructs the API request headers
    apiHeaders = {
        "Ocp-Apim-Subscription-Key": azureDetails[0],
        "Ocp-Apim-Subscription-Region": azureDetails[2],
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4())
        }
    
    ## Add the text to be translated to the body
    apiBody = [{
        "text": text
    }]
    
    ## Make the translation API request
    try:
        apiRequest = requests.post(apiUrl, headers = apiHeaders, 
                                   json = apiBody)
        apiResponse = apiRequest.json()
    except:
        print("Failed API request")

    try:
        return apiResponse[0]["translations"][0]["text"]
    except:
        return text


def Translate_Input(preferredLanguage, userInput, azureDetails):
    ## If user input is not English then translate the input into English
    if preferredLanguage != "en" and azureDetails != "Unavailable":
        try:
            userInput = Translate_Text(preferredLanguage, "en", azureDetails, 
                                   userInput)
        except:
            """An error occurred during translation"""
    return userInput
 
    
def Translate_Output(preferredLanguage, chatOutput, azureDetails):
    ## If the conversation is not English then translate the output
    if preferredLanguage != "en" and azureDetails != "Unavailable":
        chatOutput = Translate_Text("en", preferredLanguage, azureDetails, 
                                    chatOutput)
    return chatOutput


def Transcribe_Speech(preferredLanguage, azureDetails):
    if azureDetails == "Unavailable":
        print("Azure services are unavailable.")
        return ""
    ## Uses the user's default audio input to convert speech-to-text
    speechConfig = SpeechConfig(azureDetails[0], azureDetails[2])   
    speechRecognizer = SpeechRecognizer(speech_config = speechConfig)
    
    text = "Please speak into your microphone (times out after 15 seconds of silence)"
    response = Translate_Output(preferredLanguage, text, azureDetails)
    
    print(response)
    ### Decided not to read out this line as it takes a while and may confuse
    ### the user
    #Speak_Text(response, azureDetails)
    try:
        ## Uses the Azure module for the conversion
        speech = speechRecognizer.recognize_once_async().get()
        ## Displays the transcribed message to assist in the chat log
        print("> " + speech.text)
        return speech.text
    except:
        return ""


def Speak_Text(text, azureDetails):
    if azureDetails == "Unavailable":
        print("Credentials for text-to-speech are unavailable.")
        return
    ## Uses the Azure module to perform text-to-speech
    speechConfig = SpeechConfig(azureDetails[0], azureDetails[2])
    audioConfig = AudioOutputConfig(use_default_speaker = True)
    speechSynthesizer = SpeechSynthesizer(speechConfig, 
                                          audio_config = audioConfig)
    ## Outputs the data that has been converted
    try:
        speechSynthesizer.speak_text(text)
    except:
        """An error occurred during speech-to-text synthesis"""


## Initialises the variables and AIML environment required for simple chat 
kern = aiml.Kernel()
kern.learn("std-startup.xml")
kern.respond("LOAD AIML")
if os.path.isfile("bot_brain.brn"):
    kern.bootstrap(brainFile = "bot_brain.brn")
else:
    kern.bootstrap(learnFiles = "std-startup.xml", commands = "LOAD AIML")
    kern.saveBrain("bot_brain.brn")


## Initialises the CNN variables required for image recognition
cnnModel = Load_Model()
classList = Get_Class_Names()


## Initialises NLTK inference and knowledgebase for logic statements
read_expr = Expression.fromstring
knowBase = []
knowBaseIsValid = Load_Knowledgebase(knowBase)


## Initialises the Microsoft Azure services and the variables for multiple
## languages and audio I/O
azureDetails = Load_Azure_Details()
preferredLanguage = "en"
voiceEnabled = False


if cnnModel != None and classList != None and knowBase != [] \
    and knowBaseIsValid:
    print("\n\nHello and welcome to this chat bot that can assist you in the",
      "processes of obtaining a motorcycle license in the UK. Please feel",
      "free to ask any questions that you may have. I can even suggest a few",
      "bikes for you!")
    exitFlag = False   
else:
    ## Doesn't run the loop if any of the initialisation failed
    exitFlag = True
    print("\nInitialisation failed")
    
    
def minusOne(params, userInput, preferredLanguage, azureDetails, voiceEnabled):
    ## Unknown user response
    response = Translate_Output(preferredLanguage, 
                                "I did not get that, please try again.", 
                                azureDetails)
    print(response)
    if voiceEnabled:
        Speak_Text(response, azureDetails)
    ## Loop continues
    return False, voiceEnabled
    
def zero(params, userInput, preferredLanguage, azureDetails, voiceEnabled):
    response = Translate_Output(preferredLanguage, params[1], azureDetails)
    print(response)
    if voiceEnabled:
        Speak_Text(response, azureDetails)
    ## Loop is exited
    return True, voiceEnabled

def one(params, userInput, preferredLanguage, azureDetails, voiceEnabled):
    if "am" in userInput:
        userInput = "moped licence"
    if "mod" and "1" in userInput:
        userInput = "mod1"
    if "mod" and "2" in userInput:
        userInput = "mod2"
    ## The user has asked a question about some of the terminology and 
    ## processes involved
    response = Translate_Output(preferredLanguage, Answer_Query(userInput), 
                           azureDetails)
    print(response)
    if voiceEnabled:
        Speak_Text(response, azureDetails)
    ## Loop continues
    return False, voiceEnabled
    
def two(params, userInput, preferredLanguage, azureDetails, voiceEnabled):
    if "classic" in userInput:
        userInput = "retro"
    elif "sports" in userInput:
        userInput = "sport"
    ## The user has asked for a suggestion of a bike, based on a style
    response = Translate_Output(preferredLanguage, Get_Suggestion(userInput), 
                               azureDetails)
    print(response)
    if voiceEnabled:
        Speak_Text(response, azureDetails)
    ## Loop continues
    return False, voiceEnabled
    
def three(params, userInput, preferredLanguage, azureDetails, voiceEnabled):
    response = Translate_Output(preferredLanguage,
                                Evaluate_Prediction(userInput, cnnModel, 
                                                    classList), azureDetails)
    print(response)
    if voiceEnabled:
        Speak_Text(response, azureDetails)
    ## Loop continues
    return False, voiceEnabled
    
def four(params, userInput, preferredLanguage, azureDetails, voiceEnabled):
    ## The user is wanting to add a statement to the knowledgebase
    if (params[1].find(" is ") != -1):
        obj, subj = params[1].split(" is ")
    elif (params[1].find(" make ") != -1):
        obj, subj = params[1].split(" make ")
        
    response = Translate_Output(preferredLanguage, 
                                Add_To_Knowledgebase(obj, subj, knowBase), 
                                azureDetails)
    print(response)
    if voiceEnabled:
        Speak_Text(response, azureDetails)
    ## Loop continues
    return False, voiceEnabled
    
def five(params, userInput, preferredLanguage, azureDetails, voiceEnabled):
    ## The user has asked for a confirmation of logic
    if (params[1].find(" is ") != -1):
        obj, subj = params[1].split(" is ")
    elif (params[1].find(" make ") != -1):
        obj, subj = params[1].split(" make ")
    
    response = Translate_Output(preferredLanguage, 
                           Confirm_Logic_Statement(obj, subj, knowBase), 
                           azureDetails)
    print(response)
    if voiceEnabled:
        Speak_Text(response, azureDetails)
    ## Loop continues
    return False, voiceEnabled

def six(params, userInput, preferredLanguage, azureDetails, voiceEnabled):
    ## The user wants to swap to using audio input/output
    ## continues the loop and enables using voice
    voiceEnabled = True
    response = "Switching to using audio commands and enabling text to speech"
    print(response)
    Speak_Text(response, azureDetails)
    return False, voiceEnabled
    
def seven(params, userInput, preferredLanguage, azureDetails, voiceEnabled):
    ## The user wants to swap back to using text input/output
    ## continues the loop and disables using voice
    voiceEnabled = False
    print("Switching to using text only and disabling text to speech")
    return False, voiceEnabled


userOptions = {
            -1 : minusOne,
            0 : zero,
            1 : one,
            2 : two,
            3 : three,
            4 : four,
            5 : five,
            6 : six,
            7 : seven
}


def Process_Answer(answer, userInput, preferredLanguage, voiceEnabled):
    ## If the user enters a response that isn't correctly handled it prompts
    ## them to try again
    exitFlag = False
    if answer.isspace() or not answer:
        response = Translate_Output(preferredLanguage, 
                                    "I did not get that, please try again.", 
                                    azureDetails)
        print(response)
        if voiceEnabled:
            Speak_Text(response, azureDetails)
    elif answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        
        ## Uses a dictionary implementation to replicate a switch-case in order
        ## to correctly handle the user input
        exitFlag, voiceEnabled = userOptions[cmd](params, userInput,
                                                  preferredLanguage,
                                                  azureDetails, voiceEnabled)
    else:
        response = Translate_Output(preferredLanguage, answer, azureDetails)
        print(response)
        if voiceEnabled:
            Speak_Text(response, azureDetails)
        
    return exitFlag, voiceEnabled
    
## The main loop of the bot - where the user's inputs are handled
while not exitFlag:
    ## Gets the user input
    try:
        ## Gets text input
        if not voiceEnabled:
            userInput = input("> ")
        else:
        ## Otherwise gets voice input
            userInput = Transcribe_Speech(preferredLanguage, azureDetails)
    except (KeyboardInterrupt, EOFError):
        response = "\nGoodbye!"
        print(response)
        if voiceEnabled:
            Speak_Text(response, azureDetails)
        exitFlag = True
        continue

    ## Replaces all punctuation with a space and converts input to lower case
    userInput = Normalize_String(userInput)
    
    ## Detects the language in the input and translates to English if needed
    ## User input has to be longer than 0 in case microphone input times out
    if len(userInput) > 0:
        preferredLanguage = TextBlob(userInput).detect_language()
        userInput = Translate_Input(preferredLanguage, userInput, azureDetails)
    
        ## Gets a response using the AIML module
        answer = kern.respond(userInput, "content")
        
        exitFlag, voiceEnabled = Process_Answer(answer, userInput, 
                                                preferredLanguage, 
                                                voiceEnabled)