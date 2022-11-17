import json
from anyio import sleep
import ijson
import os

from elasticsearch import Elasticsearch

INDEXNAME = "dbpedia_instance_abstract"
es = Elasticsearch("http://localhost:9200", timeout=120)
es.info()

INDEX_SETTINGS = {
    'mappings': {
            'properties': {
                'type': {
                    'type': 'text',
                    'term_vector': 'yes',
                    'analyzer': 'english'
                },
                'catch_all': {
                    'type': 'text',
                    'term_vector': 'yes',
                    'analyzer': 'english'
                }
            }
        }
    }

## Checking for any indexes
if es.indices.exists(INDEXNAME):
    es.indices.delete(index=INDEXNAME)

# Creating the index
es.indices.create(index=INDEXNAME, body=INDEX_SETTINGS)

# Info; Not printed; placeholder for debugging
es.info()

count = 0

def format_to_dict(type, abstract):
    """Formats the type and abstract for indexingg"""

    """Args:
        type: The type in DBpedia
        abstract: Abstract of all the entities belonging to this type

    Returns:
        Dict format of type and catch_all for indexing
    """
    if len(abstract) > 20000: abstract = abstract[:20000]
    return {'type': type, 'catch_all': abstract}

for type, the_object_type, abstract in ijson.parse(open('processed_ttl/type_abstracts.json')):
    if abstract and type:
        print("Type {} indexed.".format(type))
        es.index(index=INDEXNAME, doc_type="_doc", id=type, body=format_to_dict(type, abstract))
        count+=1
        print("Indexed {}".format(count))

