Files to download, instance_types_en.ttl and short_abstracts_en.ttl and place it in the files_to_process directory.

The files downloaded will be compressed, should be decompressed and added in the folder.

To just check the system outputs,

a. cd outputs_check
b. [FOR BASELINE] python3 evaluate.py --type_hierarchy_tsv dbpedia_types.tsv --ground_truth_json ground_truth_from_dataset.json --system_output_json ../baseline_system_outputs/baseline_system_output.json 
c. [FOR ADVANCED] python3 evaluate.py --type_hierarchy_tsv dbpedia_types.tsv --ground_truth_json ground_truth_from_dataset.json --system_output_json ../advanced_method_system_outputs/system_output_1000_5k.json 


For advanced, there are multiple files, the main outputs are the one with the 5k at the end. Running each of these files can give the results as presented in the report.


Steps in running the code:

**** Category Prediction:
a. python3 baseline_category_predict.py
   
   -- The output, category.txt will be saved in the category_prediction folder.
   -- The output has the structure of line based. Each line represents the prediction
   -- for the particular question in an order of the test set for future use.

**** Baseline Method:
** To map the ttl files to respective json.
a. python3 mapping_ttls.py

   -- The output will be 3 files, under processed_ttl folder
   -- entity_abstracts.json is the entity description json file.
   -- type_entities.json is the json with type and its corresponding entities.
   -- type_abstracts.json is the mapping for the type and each of its entity           description.

** To index the outputs into the Elasticsearch
a. python3 indexing_json.py

   -- Indexes the type_abstracts.json, which has mappings to each of the types and the
   -- corresponding entity abstract.


** To query from the ES and get the results for the test set.

a. python3 query_from_es.py
   -- Generates the baseline system outputs.
   -- Baseline system output will be saved at baseline_system_outputs/     baseline_system_output.json


** To check the results from the baseline

a. cd outputs_check
b. python3 evaluate.py --type_hierarchy_tsv dbpedia_types.tsv --ground_truth_json ground_truth_from_dataset.json --system_output_json ../baseline_system_outputs/baseline_system_output.json 


**** Advanced Method

** Run the advanced method.

a. python3 advanced_method.py
   -- This will start creating model, the initial train questions is set to 1000.
   -- Output on model_outputs/output_from_model_reg_1000_5k.json
   (Can try running it and kill the process. The output is already in the relevant folder.)

b. python3 advance_output_generate.py
   -- Generates the system output on advanced_method_system_outputs/system_output_1000_5k.json

c. cd outputs_check

d. python3 evaluate.py --type_hierarchy_tsv dbpedia_types.tsv --ground_truth_json ground_truth_from_dataset.json --system_output_json ../advanced_method_system_outputs/system_output_1000_5k.json 
