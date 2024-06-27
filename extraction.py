import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

model = "vicuna-7b-v1.5"

task_prompt="Here is a radiology report. Please tell me if it is a normal or abnormal examination based on the main finding. Respond by tagging the report as /normal or /abnormal, like in these examples:\n Impression:"

sep_reports="NEXT_DOSS"

with open("reports_PATH.txt","r") as file:
  reports=file.read().split(sep=sep_reports)

for report in reports:
  prompt=task_prompt+report
  completion = openai.chat.completions.create(
    model=model,
    messages=[{"role": "system", "content": "You are  helpful assistant tagging radiology reports for a physician. Please respond concisely according to the template provided"},
             {"role":"user","content":prompt],
    logprobs=True,
    temperature=0,
    max_tokens=128
  )
  # print the completion
  print(completion.choices[0].message.content)
