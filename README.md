# RadioVicuna
Script from the manuscript "Performance of an Open-Source Large Language Model in Extracting Information from Free-Text Radiology Reports" submitted to Radiology: Artificial Intelligence

## Installation

(Optional) - Download text-generation-webui https://github.com/oobabooga/text-generation-webui

1 - Download the vicuna 1.3 13B weights
If you installed text-generation-webui, just execute :

```bash
  python download-model.py lmsys/vicuna-13b-v1.5
```
2- Install FastChat (the script works for version 0.2.36 as of march 5th 2024):

```bash
    pip install git+https;//github.com/lm-sys/FastChat@v0.2.36
```
3- Edit the script_extraction.py file from this repository to update the paths to FastChat, Vicuna weights and source reports, and name the table the script will create

4- Edit the inference_extraction.py file to match the variables you want to extract from the reports

5- Launch the script !

```bash
    python script_extraction
```


## Reports preparation

Reports must be in a singe .txt file, separated by a keyword you can change in auto_inference.py (default keyword is "NEXT_CASE")
We advise that you segment your reports to concentrate on the pertinent section. Future work will be done to implement 16K context to process full reports.
(Optional) - Use the texte de-identification tool developped by Chambon and colleagues (https://github.com/MIDRC/Stanford_Penn_MIDRC_Deidentifier or https://doi.org/10.1093/jamia/ocac219). This tool replaces names and places with likely occurences, to maintain the understandability of the text while removing identifiying information.

## Prompting

You can modify the prompts from the script_extraction.py file.  
We advise you to use short, natural prompts.  
We find that asking for a list variables tagged with the caracteristics of interest is a good default prompt.  
Examples from the manuscript :  

- To extract the presence of headache from the indication :  
```python
"""I will present you short indications for radiology exams. Your task is to list all the symptoms from the text and tell me if they correspond to headache (tag them with [/headache]) or to another symptom (tag them with [/other]). If there are no symptoms in the indication (for example "Suspicion de..."), just respond 'No symptom'. Keep to the text only. You will answer only according to this template (replacing the words between brackets with your answer): 
Symptoms from INDICATION : 
- Symptom : [retrieve the symptom] [tag with /headache only if the symptom is headache, else tag with /other]
etc"""
```

- To classify a finding as causing headache or not :
```python
"""I will present you abnormal findings from radiology exams. Your task is to list all the findings from them and tell me if they usually cause headache or not. Each conclusion is independant and should not influence your answer. You will answer only according to this template: 
List of findings : 
- List all the findings /can cause headache or not ?"""
```
## Few-shot

We show that few-shot in-context learning improves the information extraction performance. We strongly encourage you to add at least 2 examples, just to be sure that the model will match the template of response you desire.
Best results are usually obtained with 2-6 examples.  
To add or remove examples, simply edit the prompt variable from the script_extraction.py file to add or remove dialogues from the list. 
   
Example : you want  to add two examples to this prompt classifying normal and abnormal MRI conclusions :

```python
['Doctor', '''I will present you conclusions from radiology exams. Your task is to list all the findings from them and tell me if they correspond to normal or abnormal findings. Keep in mind, an abnormal finding may be described as stable or regressing. You will answer only according to this template: 
List of findings : 
- List all the findings /normal or abnormal finding ?'''],
['Robot', "I am ready to respond according to the template."]]
```
Edit the file to add four new elements of dialogue (two questions, two answers)  

```python
['Doctor', '''I will present you conclusions from radiology exams. Your task is to list all the findings from them and tell me if they correspond to normal or abnormal findings. Keep in mind, an abnormal finding may be described as stable or regressing. You will answer only according to this template: 
List of findings : 
- List all the findings /normal or abnormal finding ?'''],

['Robot', "I am ready to respond according to the template."],
['Doctor', "Conclusion : Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent. Absence de thrombose veineuse cérébrale. IRM encéphalique sans anomalie par ailleurs."],
['Robot', '''List of findings : 
- Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent /abnormal
- Absence de thrombose veineuse cérébrale /normal
- IRM encéphalique sans anomalie par ailleurs /normal'''],

['Doctor', "Conclusion : Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques. Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont. Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent."],
['Robot', '''List of findings : 
- Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques /abnormal
- Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont /abnormal
- Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent /abnormal''']]
```
You may add or remove examples as needed.

## Logprobs

Logprobs are the output probabilities for each token, given the context. It is a mean of accessing the model confidence level for classification tasks, and performs better than asking the model directly for a confidence level. Because the classification method we propose relies on the detection of token of interest, it is possible to display the logprob associated with a rating.
To do so, edit the script_extraction.py file with "display_logprobs=True"

## Available models

Nothing limits this method to Vicuna. All models compatible with FastChat can be implemented by just downloading them and madding the path to the weights in the script_extraction.py file.

