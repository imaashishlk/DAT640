from elasticsearch import Elasticsearch
from typing import Any, Dict, List, Union
import re, json
import pandas as pd


## Retriving the query
def retrival(
    es: Elasticsearch, index_name: str, query: str, k: int = 10
) -> List[str]:
    """Returns the retrived elements from the ES."""

    """Args:
        es: Elasticsearch, 
        index_name: Index Name, 
        query: Query String, 
        k: The size to output, defaults to 10

    Returns:
        List of the output form ES.
    """
    final_query = {"match": {"catch_all": query}}

    results = es.search(index=INDEXNAME, body={'query': final_query}, size=k)['hits']['hits']
    
    final_list = []

    for item in results:
        final_list.append(item['_id'])

    return final_list

## Cleaning the query
def cleaning_the_query(query):
    """Cleans the query."""

    """Args:
        query: The query string

    Returns:
        String of cleaned query.
    """
    re_html = re.compile("<[^>]+>")
    text = re_html.sub(" ", query)
    # Replace punctuation marks (including hyphens) with spaces.
    for c in '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~':#string punctuation
        text = text.replace(c," ")
    # Lowercase and split on whitespaces.
    lowered_text = text.lower().split()
    stop_words = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in',
     'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then', 'there',
      'these', 'they', 'this', 'to', 'was', 'will', 'with', 'who', 'what', 'when', 'where', 'which', 'whom', 'whose', 'why'])

    tokens_wo_stopwords= [word for word in lowered_text  if word not in stop_words]
    return " ".join(tokens_wo_stopwords)


def load_dataset_as_list(filename):
    """Loads the datasets from a JSON file."""

    """Args:
        path: Path to file from which to load data

    Returns:
        List of questions  and  list of coresponding categories .
    """
    output_data=[]
    dict = {}
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        for question in data:
            if not question['question']:  # Ignoring null questions
                continue

            dict = {"question":question['question'],
            "category": question['category']}
            output_data.append(dict)
    df = pd.DataFrame.from_dict(output_data)
    question=list(df['question'])
    category = list(df['category'])

    return question, category


if __name__ == "__main__":
    filename = "files_to_process/smarttask_dbpedia_test_questions.json"
    predicted_category = "category_prediction/categories.txt"

    INDEXNAME = "dbpedia_instance_abstract"

    es = Elasticsearch("http://localhost:9200", timeout=120)
    es.info()

    ## maximum number of terms
    k = 10

    retrived_dict = []
    temp_dict = {}

    list_predicted_categories = []
    with open(predicted_category) as f:
        for line in f:
            list_predicted_categories.append(line[:-1])
    
    with open(filename) as f:    
        data = json.load(f)
    
        for id, line in enumerate(data):
            processed_query = cleaning_the_query(data[id]['question'])
            the_id = data[id]['id']
            category = list_predicted_categories[id]

            if category == 'resource':
                retrived = retrival(es, INDEXNAME, processed_query, k)
            else:
                retrived = []

            for i, item in enumerate(retrived):
                retrived[i] = retrived[i]
                
            ## Extracting in format required by the evaluation python file.
            temp_dict = {"id": the_id, 
                        "category": category, 
                        "type": retrived}
            
            retrived_dict.append(temp_dict)

            print("Processed {}".format(id))
            

    # # Saving the retrived items for file
    with open("baseline_system_outputs/baseline_system_output.json", "w") as outfile:
        json.dump(retrived_dict, outfile)

    print("Baseline system output saved at baseline_system_outputs/baseline_system_output.json")

# To run:
# python3 evaluate.py --type_hierarchy_tsv dbpedia_types.tsv --ground_truth_json ground_truth_from_dataset.json --system_output_json ../baseline_system_outputs/baseline_system_output.json 
    