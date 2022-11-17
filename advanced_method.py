import json
from typing import Callable, Dict, List, Set, Tuple
import pickle
import numpy as np
from elasticsearch import Elasticsearch
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor

INDEX_NAME = "dbpedia_instance_abstract"

FIELDS = ["catch_all"]

INDEX_SETTINGS = {
    'mappings': {
        "properties": {
            "catch_all": {"type": "text", "term_vector": "yes", "analyzer": "english"},
        }
    }
}

FEATURES_QUERY = [
    "query_length",
    "query_sum_idf",
    "query_max_idf",
    "query_avg_idf",
]

FEATURES_DOC = ["doc_length_catch_all"]

FEATURES_QUERY_DOC = [
    "unique_query_terms_in_catch_all",
    "sum_TF_catch_all",
    "max_TF_catch_all",
    "avg_TF_catch_all",
]


def analyze_query(
    es: Elasticsearch, query: str, field: str, index: str = INDEX_NAME
) -> List[str]:
    """Analyze the query and only take the one with the term in the document.

    Args:
        es: Elasticsearch object instance.
        query: The query term.
        field: Field of document to consider for term frequencies.
        index: Name of the index where document is indexed.

    Returns:
        Dictionary of terms and their respective term frequencies in the field
        and document.
    """
    print(query)

    if query is None:
        return

    tokens = es.indices.analyze(index=index, body={"text": query})["tokens"]

    query_terms = []

    for t in sorted(tokens, key=lambda x: x["position"]):
        
        hits = (
            es.search(
                index=index,
                body={'query': {"match": {field: t["token"]}}},
                _source=False,
                size=1,
            )
            .get("hits", {})
            .get("hits", {})
        )
        doc_id = hits[0]["_id"] if len(hits) > 0 else None
        if doc_id is None:
            continue
        query_terms.append(t["token"])
    return query_terms


def get_doc_term_freqs(
    es: Elasticsearch, doc_id: str, field: str, index: str = "dbpedia_instance_abstract"
) -> Dict[str, int]:
    """Gets the term frequencies of a field of an indexed document.

    Args:
        es: Elasticsearch object instance.
        doc_id: Document identifier with which the document is indexed.
        field: Field of document to consider for term frequencies.
        index: Name of the index where document is indexed.

    Returns:
        Dictionary of terms and their respective term frequencies in the field
        and document.
    """
    term_vectors = es.termvectors(
        index=index, id=doc_id, fields=field, term_statistics=True
    )

    if term_vectors["_id"] != doc_id:
        return None
    if field not in term_vectors["term_vectors"]:
        return None

    term_freqs = {}

    for term, term_stat in term_vectors["term_vectors"][field]["terms"].items():
        term_freqs[term] = term_stat["term_freq"]

    return term_freqs


def extract_query_features(
    query_terms: List[str], es: Elasticsearch, index: str = "dbpedia_instance_abstract"
) -> Dict[str, float]:
    """Extracts features of a query.

    Args:
        query_terms: List of analyzed query terms.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.
    Returns:
        Dictionary with keys 'query_length', 'query_sum_idf',
            'query_max_idf', and 'query_avg_idf'.
    """
    # TODO
    instances_list = []
    query_features = {'query_length': len(
        query_terms), 'query_sum_idf': 0, 'query_max_idf': 0, 'query_avg_idf': 0}
    for term in query_terms:
        el_search = (
            es.search(
                index=index,
                body={'query': {"match": {"catch_all": term}}},
                # query={"match": {"body": term}},
                _source=False,
                size=1,
            )
            .get("hits", {})
            .get("hits", {})
        )

        doc_id = el_search[0]["_id"] if len(el_search) > 0 else None
        if doc_id != None:
            tv = es.termvectors(index=index, id=doc_id,
                                fields="catch_all", term_statistics=True)
            # print(tv)
            if tv["term_vectors"].get("catch_all") is not None:
                retrieve_term = tv["term_vectors"]["catch_all"]["terms"].get(
                    term)
                field_stat = tv["term_vectors"]["catch_all"]["field_statistics"]
                if retrieve_term and field_stat:
                    val = math.log(
                        field_stat['doc_count']/retrieve_term['doc_freq'])
                    instances_list.append(val)
    if len(instances_list) != 0:
        query_features['query_avg_idf'] = sum(
            instances_list) / len(query_terms)
        query_features['query_sum_idf'] = sum(instances_list)
        query_features['query_max_idf'] = max(instances_list)

    return query_features


def extract_doc_features(
    doc_id: str, es: Elasticsearch, index: str = "dbpedia_instance_abstract"
) -> Dict[str, float]:
    """Extracts features of a document.

    Args:
        doc_id: Document identifier of indexed document.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.

    Returns:
        Dictionary with key 'doc_length_catch_all'.
    """
    # TODO
    doc_features = {}
    tv = es.termvectors(
        index=index, id=doc_id, term_statistics=True, fields=["catch_all"]
    )

    doc_length_catch_all = 0
    # doc_length_title = 0

    if tv['found'] == False:
        return {"doc_length_catch_all": doc_length_catch_all}

    if tv["_id"] != doc_id:
        return None

    if "catch_all" not in tv["term_vectors"]:
        doc_features["doc_length_catch_all"] = 0
    else:
        for _, values in tv["term_vectors"]["catch_all"]["terms"].items():
            doc_length_catch_all += values["term_freq"]

    return {"doc_length_catch_all": doc_length_catch_all}


def extract_query_doc_features(
    query_terms: List[str],
    doc_id: str,
    es: Elasticsearch,
    index: str = "dbpedia_instance_abstract",
) -> Dict[str, float]:
    """Extracts features of a query and document pair.

    Args:
        query_terms: List of analyzed query terms.
        doc_id: Document identifier of indexed document.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.

    Returns:
        Dictionary with keys 'unique_query_terms_in_catch_all, 'sum_TF_catch_all',
            'max_TF_catch_all', 'avg_TF_catch_all'.
    """
    
    term_vectors = es.termvectors(
        index=index, id=doc_id, term_statistics=True, fields=["catch_all"]
    )

    query_doc_features = {
        "unique_query_terms_in_catch_all": 0,
        "sum_TF_catch_all": 0,
        "max_TF_catch_all": 0,
        "avg_TF_catch_all": 0,
    }

    catch_all_term_frequency = []
    if term_vectors['found'] == False:
        return query_doc_features
    if term_vectors["_id"] != doc_id:
        return None

    for term in query_terms:
        if term_vectors["term_vectors"].get("catch_all") is not None:
            term_catch_all = term_vectors["term_vectors"]["catch_all"]["terms"].get(term)
            if term_catch_all is not None:
                query_doc_features["unique_query_terms_in_catch_all"] += 1
                query_doc_features["sum_TF_catch_all"] += term_catch_all["term_freq"]

                catch_all_term_frequency.append(term_catch_all["term_freq"])

    if len(catch_all_term_frequency) != 0:
        query_doc_features["max_TF_catch_all"] = max(catch_all_term_frequency)
        query_doc_features["sum_TF_catch_all"] = sum(catch_all_term_frequency)
        query_doc_features["avg_TF_catch_all"] = sum(
            catch_all_term_frequency)/len(query_terms)

    return query_doc_features


def extract_features(
    query_terms: List[str],
    doc_id: str,
    es: Elasticsearch,
    index: str = "dbpedia_instance_abstract",
) -> List[float]:
    """Extracts query features, document features and query-document features
    of a query and document pair.

    Args:
        query_terms: List of analyzed query terms.
        doc_id: Document identifier of indexed document.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.

    Returns:
        List of extracted feature values in a fixed order.
    """
    query_features = extract_query_features(query_terms, es, index=index)
    feature_vect = [query_features[f] for f in FEATURES_QUERY]

    doc_features = extract_doc_features(doc_id, es, index=index)
    feature_vect.extend([doc_features[f] for f in FEATURES_DOC])

    query_doc_features = extract_query_doc_features(
        query_terms, doc_id, es, index=index
    )
    feature_vect.extend([query_doc_features[f] for f in FEATURES_QUERY_DOC])

    return feature_vect


def load_queries(data_str: str) -> Dict[str, str]:
    """ Loads the queries from the given string. The given 
        data_str is in json like structure.

    Args:
        data_str: JSON like string.

    Returns:
        Dictionary with key of id and the relevant question
    """
    queries_dict = {}

    for item in data_str:
        queries_dict[item['id']] = item['question']

    return queries_dict


def load_types(data_str: str) -> Dict[str, str]:
    """ Loads the catetgory from the given string. The given 
    data_str is in json like structure.

    Args:
        data_str: JSON like string.

    Returns:
        Dictionary with key of id and the relevant category in str, if resource 
        the returned item is resource; else the empty list which depicts the output
        is not resource.
    """
    queries_dict = {}

    for item in data_str:
        retrived_type = item['type']

        if len(retrived_type) == 1:
            if retrived_type[0] == 'boolean' or retrived_type[0] == 'string' or retrived_type[0] == 'number' or retrived_type[0] == 'date':
                retrived_type = []

        queries_dict[item['id']] = retrived_type

    return queries_dict


def load_type_list(data_str: str) -> Dict[str, List[str]]:
    """ Loads the types from the given string. The given 
    data_str is in json like structure.

    Args:
        data_str: JSON like string.

    Returns:
        Dictionary with key of id and the relevant types in list.
    """
    
    query_list_types = {}

    for item in data_str:
        if item['category'] == 'resource':
            query_list_types[item['id']] = item['type']
        else:
            query_list_types[item['id']] = []

    return query_list_types

def load_from_test(filepath: str) -> Dict[str, str]:
    """ Loads the test set in desired dict format

    Args:
        filepath: The file to use.

    Returns:
        Dictionary with key of id and the relevant questions from the test set.
    """
    queries_dict = {}

    with open(filepath, 'r+') as file:
        queries = json.load(file)
        
    for item in queries:
        queries_dict[item['id']] = item['question']

    return queries_dict



def train_split(filename):
    """ Splits the training data to desired number.

    Args:
        filename: Training set file.

    Returns:
        List with the first item being the splitted and the second empty.
        Second is kept for future, if the set needs to be changed.
    """
    with open(filename, 'r') as file:
        read_file = json.load(file)

    read_file = read_file[:1000]

    print("Total Set: ", len(read_file))

    return [read_file, []]


def prepare_ltr_training_data(
    # query_ids: List[str],
    all_queries: Dict[str, str],
    all_types_from_train: Dict[str, str],
    all_typelist: Dict[str, List[str]],
    es: Elasticsearch,
    index: str,
) -> Tuple[List[List[float]], List[int]]:

    """ Prepares the data, X and y labels for Learning to Rank.

    Args:
        all_queries: the queries to be processed,
        all_types_from_train: type dict from the train,
        all_typelist: the type list from the queries,
        es: Elasticsearch Instance,
        index: Index Name,

    Returns:
        The prepared X and y labels for LTR to train
    """

    X = []
    y = []

    count = 0
    for id, value in all_queries.items():
        count += 1
        query_terms = analyze_query(
            es=es, index=index, field="catch_all", query=value)

        if query_terms == None:
            continue

        if len(query_terms) == 0:
            continue

        el_search = es.search(
            index=index, q=" ".join(query_terms), _source=True, size=5)["hits"]["hits"]

        doc_ids = [hit["_id"] for hit in el_search]

        for item in all_types_from_train[id]:
            if item in doc_ids:
                continue
            else:
                doc_ids.append(item)

        for d_id in doc_ids:
            d_id_features = extract_features(
                doc_id=d_id, es=es, index=INDEX_NAME, query_terms=query_terms)
            X.append(d_id_features)

            y_label = 0

            if id in all_typelist and len(all_typelist[id]) > 0:
                y_label = 1

            y.append(y_label)

        print("Query processed {}".format(count))

    return X, y


class PointWiseLTRModel:
    def __init__(self) -> None:
        """Instantiates LTR model with an instance of scikit-learn regressor."""
        self.regressor = SGDRegressor(
            max_iter=1000, tol=1e-3, penalty="elasticnet", loss="huber", random_state=42)
        # self.regressor = RandomForestRegressor(max_depth = 2, n_estimators=1000)

    def _train(self, X: List[List[float]], y: List[float]) -> None:
        """Trains an LTR model. Also saves the model in the saved_models path.

        Args:
            X: Features of training instances.
            y: Relevance assessments of training instances.
        """
        # assert self.regressor is not None
        self.model = self.regressor.fit(X, y)
        
        with open('saved_models/advanced_regressor_1000_train_5k', 'wb') as file:
            pickle.dump(self.model, file)


    def rank(
        self, ft: List[List[float]], doc_ids: List[str]
    ) -> List[Tuple[str, float]]:
        """Predicts relevance labels and rank documents for a given query.

        Args:
            ft: A list of feature vectors for query-document pairs.
            doc_ids: A list of document ids.
        Returns:
            List of tuples, each consisting of document ID and predicted
                relevance label.
        """
        
        rel_labels = self.model.predict(ft)
        sort_indices = np.argsort(rel_labels)[::-1]

        results = []
        for i in sort_indices:
            results.append((doc_ids[i], rel_labels[i]))
        return results


def get_rankings(
    ltr: PointWiseLTRModel,
    all_queries: Dict[str, str],
    es: Elasticsearch,
    index: str,
    rerank: bool = False,
) -> Dict[str, List[str]]:
    """Generate rankings for each of the test queries.

    Args:
        ltr: A trained PointWiseLTRModel instance.
        all_queries: A dict type all query list.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.
        rerank: Boolean flag indicating whether the first-pass retrieval
            results should be reranked using the LTR model.

    Returns:
        A dictionary of rankings for each test query ID.
    """
    count = 0
    test_rankings = {}
    
    for id, value in all_queries.items():
        count += 1
        print("Processing {} sets.".format(count))
        
        query_terms = analyze_query(
            es=es, index=index, field="catch_all", query=value)

        if len(query_terms) == 0:
            print('Length of query is zero')
            continue

        hits = es.search(
            index=index, q=" ".join(query_terms), _source=True, size=5)["hits"]["hits"]
  
        test_rankings[id] = [hit["_id"] for hit in hits]

        # Rerank the first-pass result set using the LTR model.
        if rerank:
            document_ids = test_rankings[id]
            test_rankings[id] = []
            all_features = []
            for doc_id in document_ids:
                doc_id_features = extract_features(
                    query_terms=query_terms, doc_id=doc_id, es=es, index=index)
                all_features.append(doc_id_features)
            results = ltr.rank(all_features, document_ids)
            res = list(list(zip(*results))[0])
            test_rankings[id] = res
    return test_rankings


# The processes initiated

es = Elasticsearch("http://localhost:9200", timeout=120)

# Splitting the training set, as desired in numbers
t_t_split = train_split('files_to_process/smarttask_dbpedia_train.json')

# Loading the queries, types and the type lists
all_queries_train = load_queries(t_t_split[0])
all_types_from_train = load_types(t_t_split[0])
all_typelist_train = load_type_list(t_t_split[0])

# Preparing data for the LTR
X_train, y_train = prepare_ltr_training_data(
    all_queries=all_queries_train,
    all_types_from_train=all_types_from_train,
    all_typelist=all_typelist_train,
    es=es,
    index=INDEX_NAME,
)

# Initiating the LTR Model (PointWise)
ltr = PointWiseLTRModel()
ltr._train(X_train, y_train)

print("Testing Starts...")

# Preparing the test set
actual_test_all_queries = load_from_test(
    'files_to_process/smarttask_dbpedia_test_questions.json')

# Getting the rankings, LTR. 
rankings_ltr_atest = get_rankings(
    ltr=ltr,
    all_queries=actual_test_all_queries,
    es=es,
    index=INDEX_NAME,
    rerank=True
)

# Saving the output to the model_outputs directory
with open("model_outputs/output_from_model_reg_1000_5k.json", "w") as outfile:
    json.dump(rankings_ltr_atest, outfile)
