import json

"""Generate the system output from the output of the model.

    Returns:
        A JSON File under advanced_method_system_outputs 
        folder which is the final system_output for the advance method applied.
"""

op_from_model = 'model_outputs/output_from_model_reg_1000_5k.json'
question_set = "files_to_process/smarttask_dbpedia_test_questions.json"
category_list = "category_prediction/categories.txt"


with open(op_from_model) as f:    
    predicted_types = json.load(f)

# Loading the predicted categories
list_predicted_categories = []
with open(category_list) as f:
    for line in f:
        list_predicted_categories.append(line[:-1])

with open(question_set) as f:    
    data = json.load(f)

retrived_dict = []

for id, line in enumerate(data):
    question_id = line['id']
    category = list_predicted_categories[id]

    if category == 'resource':
        retrived = predicted_types[question_id]
    else:
        retrived = []

    temp_dict = {"id": question_id, 
                "category": category, 
                "type": retrived}
    
    retrived_dict.append(temp_dict)

    print("Processed {}".format(id))


# # Saving the retrived items for file
with open("advanced_method_system_outputs/system_output_1000_5k.json", "w") as outfile:
    json.dump(retrived_dict, outfile)