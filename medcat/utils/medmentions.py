import json
import pandas as pd

def original2concept_csv(data_path, out_path):
    f = open(data_path)
    csv_data = [['cui', 'type_id', 'name', 'name_status']]

    cui2names = {}

    for row in f:
        if row != '\n':
            if '|t|' not in row[0:13] and '|a|' not in row[0:13]:
                # Entity row
                parts = row.split("\t")
                cui = parts[5].strip()
                type_id = "|".join(parts[4].split(","))
                name = parts[3]
                csv_data.append([cui, type_id, name, 'A'])

                if cui in cui2names:
                    cui2names[cui].add(name)
                else:
                    cui2names[cui] = {name}

    df = pd.DataFrame(csv_data[1:], columns=csv_data[0])

    df.to_csv(out_path, index=False)

    return df

def original2pure_text(data_path, out_path):
    f = open(data_path)
    out = open(out_path, 'w')

    for row in f:
        if row != '\n':
            if '|t|' in row[0:13]:
                # It is title
                parts = row.split("|t|")
                title = parts[1].strip()
            elif '|a|' in row[0:13]:
                # Text row
                parts = row.split("|a|")
                text = parts[1].strip()
                out.write(title + " " + text + " " + "\n")
    out.close()

def original2json(data_path, out_path):
    f = open(data_path)
    data = {'projects': [{'name': 'medmentions', 'id': 0, 'documents': []}]}
    documents = []
    document = {}

    for row in f:
        if row != '\n':
            if '|t|' in row[0:13]:
                # It is title
                parts = row.split("|t|")
                doc_id = parts[0]
                title = parts[1].strip()
            elif '|a|' in row[0:13]:
                # Text row
                parts = row.split("|a|")
                text = parts[1].strip()
                document['text'] = title + " " + text
                document['annotations'] = []
            else:
                # Entity row
                parts = row.split("\t")
                start = int(parts[1])
                end = int(parts[2])
                cui = parts[5].strip()
                type_id = "|".join(parts[4].split(","))
                name = parts[3]

                document['annotations'].append({
                    'start': start,
                    'end': end,
                    'cui': cui,
                    'type_id': type_id,
                    'value': name})
        else:
            documents.append(document)
            document = {}
    data['projects'][0]['documents'] = documents

    json.dump(data, open(out_path, 'w'))
    return data
