import json
import os

location = os.getcwd()
subs_dict = {}
for file in os.listdir(location+'/reddit-data'):
    print(file)
    if file.endswith(".jsonlist"):
        with open('reddit-data/'+file, encoding="latin") as f:
            subs_dict[file[:-9]] = []
            for line in f:
                try:
                    line_json = json.loads(line)
                except json.decoder.JSONDecodeError:
                    continue
                if line_json not in subs_dict[file[:-9]]:
                    subs_dict[file[:-9]].append(line_json) #remove duplicates

