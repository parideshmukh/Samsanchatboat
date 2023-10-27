from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import shutil

##a7c4f2,  #22698c,  9ab3b2,  #8691a3
css = """
#warning {background-color: #5a6585}
.warning1 {background-color: #8691a3}
.feedback {font-size: 34px !important;}
.feedback textarea {font-size: 34 px !important;}
.feedback1 {font-size: 34px !important;}
.feedback1 textarea {font-size: 34 px !important;}
.css_code= 'body{background-image:url("C:/Users/admin/image001.jpg");}'
"""


os.environ["OPENAI_API_KEY"] = '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'    #(paid pradeep)

#os.environ["OPENAI_API_KEY"] ='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'

#openai.api_key ='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'

#css_code= 'body{background-image:url("C:/Users/admin/image001.jpg");}'

import speech_recognition as sr
import pyttsx3

# Initialize the recognizer
r = sr.Recognizer()


# Function to convert text to
# speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


# Loop infinitely for user to
# speak

def speakprocess():
    while (1):
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                print("Did you say ", MyText)
                SpeakText(MyText)
                return (MyText)

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("unknown error occurred")


def SpeakOut(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()
def construct_index(directory_path):
    max_input_size = 40960
    num_outputs = 5120
    max_chunk_overlap = 200
    chunk_size_limit = 6000

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    #llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=.7, model_name="text - davinci - 003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index, "Success"
dirPATH='C:/Users/admin/Python Projects/MyTestChatGpt/docs/'

def train():
    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index, "Success"

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response


def chatbot1(x):
    return "Success"

def chatbot2(x):
    return "Training Complete"



index= construct_index('C:/Users/hande/OneDrive/Desktop/chatboat/venv/vectorIndex')

def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths

with gr.Blocks(css=css,theme=gr.themes.Glass()) as demo:  #theme=gr.themes.Glass()
    #with gr.Row(equal_height=True):
    with gr.Column(scale=1):
        with gr.Row():
            with gr.Column(scale=4, min_width=400):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=200):
                        img1 = gr.Image("C:/Users/hande/Downloads/logo.png", scale=1)
                        #img1 = gr.Image("C:/Users/admin/Desktop/chatbot.jpg", scale=1)
                    with gr.Column(scale=4, min_width=300):
                        Title = gr.Button('Samsan-Generative AI Chatbot',elem_classes="feedback1",elem_id="warning")
                        file_output = gr.File()
                        upload_button = gr.UploadButton("Upload a File", file_types=[".PDF"], file_count="multiple",scale=0,elem_classes="feedback",elem_id="warning")
                        upload_button.upload(upload_file, upload_button, file_output)
                        print (file_output)
                        #greet_btn20 = gr.Button('  Train  ', scale=1, elem_id="warning", elem_classes="feedback")
                        #output2 = gr.Textbox(label="Training ", lines=2, scale=1, elem_id="warning",
                                     #elem_classes="feedback1")
                greet_btn5 = gr.Button("How can I Help You?", scale=1,elem_classes="feedback")
                name1 = gr.Textbox(label="How can I Help You?", lines=5, scale=1,elem_id="warning",elem_classes="feedback1")
                greet_btn20 = gr.Button('  Speak out your query  ', scale=1, elem_id="warning", elem_classes="feedback")
                output1 = gr.Textbox(label="My Response", lines=6, scale=1, elem_id="warning", elem_classes="feedback1")
                with gr.Row(equal_height=True):
                    greet_btn11 = gr.Button(' SpeakText',scale=1,elem_id="warning",elem_classes="feedback")
                    greet_btn1 = gr.Button("Please Assist", scale=1, elem_id="warning", elem_classes="feedback")
                    #greet_btn12= gr.Button('',scale=1, elem_id="warning", elem_classes="feedback")
                    #output1 = gr.Textbox(label="My Response", lines=6, scale=1,elem_id="warning",elem_classes="feedback1")
    greet_btn1.click(fn=chatbot, inputs=name1, outputs=output1)
    greet_btn20.click(fn=speakprocess, inputs=[], outputs=name1)
    greet_btn11.click(fn=SpeakText, inputs=output1, outputs=[])


    gr.Examples(
        examples=[['How to know if there is water in fuel system'],['How can I set user speed limit higher than factory limit in UD Trucks'], ['Summerize shell project in Penguin.. Act as CEO']]                                                    ,
        inputs=name1,
        outputs=output1,
        fn=chatbot,
        cache_examples=False,
    )


demo.launch(share=True)