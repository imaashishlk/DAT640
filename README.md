Several output files and the files to process are not added here in this repository. However, the instruction to download is given below. To just run the output, the relevant system outputs are given in the respective baseline_system_outputs and advanced_method_system_outputs directories. If the manual process is intended, all the processes must be completed to get each of the output files in directories. Downloading the ttl files is a must for these files to appear.

### Instructions:

1. Download [instance_types_en.ttl](http://downloads.dbpedia.org/2016-10/core/instance_types_en.ttl.bz2) and [short_abstracts_en.ttl](http://downloads.dbpedia.org/2016-10/core/short_abstracts_en.ttl.bz2). The files downloaded will be compressed, should be decompressed and added in the folder.
2. Place it in the files_to_process folder.
3. The files in processed_ttl will be generated once the mapping_ttls.py is run. 

###### To just check the system outputs, for baseline (2) and advanced (3), without downloading the instance_types_en.ttl and short_abstracts_en.ttl

* cd outputs_check
* python3 evaluate.py --type_hierarchy_tsv dbpedia_types.tsv --ground_truth_json ground_truth_from_dataset.json --system_output_json ../baseline_system_outputs/baseline_system_output.json 
* python3 evaluate.py --type_hierarchy_tsv dbpedia_types.tsv --ground_truth_json ground_truth_from_dataset.json --system_output_json ../advanced_method_system_outputs/system_output_1000_5k.json 

###### For advanced, there are multiple files, the main outputs are the one with the 5k at the end. Running each of these files can give the results as presented in the report.


### Steps in manually running the code (this generates files in all relevant directories):

#### Category Prediction:
* To run: python3 baseline_category_predict.py
* The output, category.txt will be saved in the category_prediction folder.
* The output has the structure of line based. Each line represents the prediction for the particular question in an order of the test set for future use.

#### Baseline Method:

###### To map the ttl files to respective json.
* To run mappings: python3 mapping_ttls.py
* The output will be 3 files, under processed_ttl folder
* entity_abstracts.json is the entity description json file.
* type_entities.json is the json with type and its corresponding entities.
* type_abstracts.json is the mapping for the type and each of its entity description.

###### To index the outputs into the Elasticsearch
* To run: python3 indexing_json.py
* This indexes the type_abstracts.json, which has mappings to each of the types and the corresponding entity abstract.

##### To query from the ES and get the results for the test set.

* To run: python3 query_from_es.py
* This generates the baseline system outputs.
* Baseline system output will be saved at baseline_system_outputs/baseline_system_output.json

##### To check the results from the baseline

* cd outputs_check
* To run: python3 evaluate.py --type_hierarchy_tsv dbpedia_types.tsv --ground_truth_json ground_truth_from_dataset.json --system_output_json ../baseline_system_outputs/baseline_system_output.json 


#### Advanced Method

##### Run the advanced method.

* To run: python3 advanced_method.py
* This will start creating model, the initial train questions is set to 1000.
* Output on model_outputs/output_from_model_reg_1000_5k.json (Can try running it and kill the process. The output is already in the relevant folder.)

##### Generate output from the JSON
* To run: python3 advance_output_generate.py
* Generates the system output on advanced_method_system_outputs/system_output_1000_5k.json

##### To check the results from the advanced
* cd outputs_check
* To run: python3 evaluate.py --type_hierarchy_tsv dbpedia_types.tsv --ground_truth_json ground_truth_from_dataset.json --system_output_json ../advanced_method_system_outputs/system_output_1000_5k.json 
