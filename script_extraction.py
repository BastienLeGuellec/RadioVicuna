"""
Uploaded August 2, 2023
"""

import csv
from itertools import zip_longest


from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

from fastchat.model.model_adapter import add_model_args
from fastchat.serve.inference_extraction import ChatIO, chat_loop_extraction

#Define the model path
vicuna = '/model_path/'

#Define the source txt document path
source_reports= '/your_path_here/reports.txt'

#Define the desired table name
csv_path = '/your_path_here/table.csv'

#Prompt
zero_shots =[
['Doctor', '''Your task is to list all the postitive findings of the report I will present you. One curcial rule is that you must ignore all the negative ("Pas de") or normal ("normale") findings. You must repond according to this template : "ID : [Patient's ID] - Positive findings : [Complete list of positive findings in the report. Please be sure to include every sentence in the report EXCEPT the ones beginning with the words "Pas de", or including the word "normale"] - Any positive finding ? : [your list include any positive finding ? Yes or No. "IRM cérébrale normale" is NOT a positive finding !]".'''], 
['Robot', "I will list all the positive findings of the report and ignore all the negative ones."]
]


#Few-shot
few_shots_finding =[
['Doctor', '''Your task is to list all the findings of radiology reports and tell me if at least one is abnormal. By abnormal, I mean any sort of lesion, including insignificant ones. Don't forget, stable or regressing findings are abnormal findings! First, list all the sentences of the report. Then, tell me if they describe a normal or an abnormal finding. Finally, tell me if your list contains any abnormal finding. If there is at least one abnormal finding in your list, you must respond Yes. You must respond only according to this template : ID Patient : [Patient's ID] - All findings : [List of all findings (normal or abnormal)] - Any positive finding ? : [If any abnormal finding in your list, you MUST respond Yes, otherwise: No]'''], 
['Robot', "I will list all the positive findings of the reports you will provide. If my list includes any abnormal finding, I will answer Yes to your question."],

['Doctor', "ID :13813406.0 : Absence d'argument pour une dissection des troncs supra-aortiques. Pas d'argument en faveur d'une TVC."],
['Robot', "ID Patient : 13813406.0 - All findings : Absence d'argument pour une dissection des troncs supra-aortiques (normal) ; Pas d'argument en faveur d'une TVC (normal) - Any positive finding ? : No"],

['Doctor', "ID :12145521.0 Pas d'argument pour une thrombose veineuse intracrânienne. Stabilité des signes d'hypotension du LCR. Aspect inchangé de la pachyméningite réactionnelle des convexités."],
['Robot',"ID Patient : 12145521.0 - All findings : Pas d'argument pour une thrombose veineuse intracrânienne (normal) ; Stabilité des signes d'hypotension du LCR (abnormal) ; Aspect inchangé de la pachyméningite réactionnelle des convexités (abnormal) - Any positive finding ? Yes"],
]

#Defines the ChatIO
class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)
chatio = SimpleChatIO()





"""
model_path: give the model path

device: 'cuda'

num_gpus: specify the number of GPUs on which the model will run

max_gpu_memory: specify the max GPU memory allocated to the model

load_8bit: possibility to load the model on 8bit

cpu_offloading: possibility to offload the model on the CPU

conv_template: specify the conversation template to use (change in the conversation.py file from FastChat)

temperature: model temperature (from experience, 0 or 0.2 gives best answers)

repetition_penalty: numeric handicap value for a token that has already appeared in the answer to appear again

max_new_tokens=1024

chatio=chatio

debug: debug mode

few_shots: specify the variable to use (prompt and examples)

file_path: specify the path of the source txt file 

gptq_config: possibility to load a gptq model

revision: revision mode
"""


d= chat_loop_interpret(model_path=vicuna,device='cuda',num_gpus=2,max_gpu_memory=None,load_8bit=False,cpu_offloading=False,conv_template="vicuna_v1.1",temperature=0.2,repetition_penalty=100,max_new_tokens=1024,chatio=chatio,debug=False,few_shots=few_shots_finding,file_path=source_reports, gptq_config=False, revision=False)

export_data = zip_longest(*d, fillvalue = '')

with open(csv_path, 'w', newline='') as myfile:
      wr = csv.writer(myfile)

      wr.writerows(export_data)
myfile.close()
