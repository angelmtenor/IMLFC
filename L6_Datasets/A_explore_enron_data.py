"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
# Original: https://github.com/udacity/ud120-projects/
# Datasets:  https://github.com/udacity/ud120-projects/tree/master/final_project

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "br"))

print("people: {}".format(len(enron_data)))
# print(list(people for people in enron_data))

poi = [people for people in enron_data if enron_data[people]["poi"] == 1]
print("POIs in data: {}".format(len(poi)))

with open("../final_project/poi_names.txt", 'r') as f:
    poi_names = f.readlines()[2:]
print("Existing POIs: {}".format(len(poi_names)))

# print(enron_data['PRENTICE JAMES'])
stock_jp = enron_data['PRENTICE JAMES']['total_stock_value']
print("Stock James Prentice {}".format(stock_jp))

mails_to_poi_wc = enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print('mails from Wesley Colwell to POI: {}'.format(mails_to_poi_wc))

stock_options_jp = enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print("Stock Options Jeffrey K Skilling {}".format(stock_options_jp))
