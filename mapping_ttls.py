import json
import string

instance_type_ttl = "files_to_process/instance_types_en.ttl"

count = 0

## Opening the file
f = open(instance_type_ttl, 'r', encoding = 'utf-8')

entity_type = {}

for idx, line in enumerate(f):
    # Ignoring first line
    if idx == 0: continue

    line = line.split(" ")
    count += 1

    if count % 100000 == 0:
        print("{} lines has been processed.".format(count))

    entity_related = line[0]
    type_related = line[2]

    entity = entity_related[1:-1].split('/')[-1].replace('_', ' ').replace('  ', ' ')
    type = type_related[1:-1].split('/')[-1].replace('_', ' ').replace('  ', ' ')

    # Ignoring Thing type
    if type == 'owl#Thing': continue

    # Recreating type by concatenating "dbo:"
    type = 'dbo:' + type

    # Creating a dictionary with list of entities for a particular type
    if type in entity_type:
        entity_type[type].append(entity)
    else:
        entity_type[type] = []
        entity_type[type].append(entity)

# Saving it in the json, ensuring the characters are in utf-8 
# by ensuring_ascii to False.
with open("processed_ttl/type_entities.json", 'w',encoding='utf-8') as f:
    json.dump(entity_type, f, ensure_ascii=False)

print("{} lines processed and dumped in the JSON [processed_ttl/type_entities.json].".format(count))


short_abstract_ttl = "files_to_process/short_abstracts_en.ttl"

count = 0

## Opening the file
f = open(short_abstract_ttl, 'r', encoding = 'utf-8')

entity_abstract = {}

for idx, line in enumerate(f):
    # Ignoring first line
    if idx == 0: continue

    # Ignoring the line starting with # symbol
    if line.startswith("#"): continue

    line = line.split(" ")
    
    count += 1

    if count % 100000 == 0:
        print("{} lines has been processed.".format(count))

    entity_related = line[0]
    abstract_related = line[2:-1]
    
    entity = entity_related[1:-1].split('/')[-1].replace('_', ' ').replace('  ', ' ')
    # only taking the first sentence from the abstract
    # removing the irrelevant strings and splitting with fullstop (.)
    # formatted_abstract = ' '.join(abstract_related).replace('"', '').replace('@en', '').split(".")[0]
    formatted_abstract = ' '.join(abstract_related).replace('"', '').replace('@en', '').split(" ")[:10]
    formatted_abstract = " ".join(formatted_abstract)
    
    # Removing punctuations
    formatted_abstract = formatted_abstract.translate(str.maketrans('', '', string.punctuation))

    # Creating a dictionary of entity and formatted abstract
    entity_abstract[entity] = formatted_abstract

# Saving it in the json, ensuring the characters are in utf-8 
# by ensuring_ascii to False.
with open("processed_ttl/entity_abstracts.json", 'w', encoding='utf-8') as f:
    json.dump(entity_abstract, f, ensure_ascii=False)

print("{} lines processed and dumped in the JSON [processed_ttl/entity_abstracts.json].".format(count))

type_abstracts = {}

count = 0

for type, entities in entity_type.items():
    count += 1

    abstracts = ''

    # Fetching all the abstracts and the entities to create a json
    for entity in entities:
        if entity in entity_abstract:
            abstracts += " " + entity_abstract[entity]

    # Creating a dictionary of type and its corresponding abstracts
    # The abstracts is the concatenation of all entities belonging to the type
    # Only the first sentence is considered
    type_abstracts[type] = abstracts

# Saving it in the json, ensuring the characters are in utf-8 
# by ensuring_ascii to False.
with open("processed_ttl/type_abstracts.json", 'w',encoding='utf-8') as f:
    json.dump(type_abstracts, f, ensure_ascii=False)

print("{} lines processed and dumped in the JSON [processed_ttl/type_abstracts.json].".format(count))