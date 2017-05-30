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

# Data: https://github.com/udacity/ud120-projects/tree/master/final_project

import pickle

enron_data = pickle.load(open("final_project_dataset.pkl", "br"))

print("\n FIELDS:")
print([fields for fields in enron_data["LAY KENNETH L"]])

print("\n PEOPLE:")
print([people for people in enron_data])
n_people = len(enron_data)
print("\nn_people: {}".format(n_people))

poi = [people for people in enron_data if enron_data[people]["poi"] == 1]
print("POIs in data: {}".format(len(poi)))

with open("poi_names.txt", 'r') as f:
    poi_names = f.readlines()[2:]
print("Existing POIs: {}".format(len(poi_names)))

# print(enron_data['PRENTICE JAMES'])
stock_jp = enron_data['PRENTICE JAMES']['total_stock_value']
print("Stock James Prentice {}".format(stock_jp))

mails_to_poi_wc = enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print('mails from Wesley Colwell to POI: {}'.format(mails_to_poi_wc))

stock_options_jp = enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print("Stock Options Jeffrey K Skilling {}".format(stock_options_jp))

ceo = "LAY KENNETH L", "SKILLING JEFFREY K", "SKILLING JEFFREY K"
ceo_pay = dict()
for i in ceo:
    ceo_pay[i] = enron_data[i]["total_payments"]
max_ceo = max(ceo_pay, key=ceo_pay.get)
print("CEO with max total payments: {}, with ${}".format(max_ceo, ceo_pay[max_ceo]))

qs = [people for people in enron_data if float(enron_data[people]["salary"]) > 0]
print("folks with quantified salary: {}".format(len(qs)))

em = [i for i in enron_data if "@" in enron_data[i]["email_address"]]

print("folks with email address: {}".format(len(em)))

tp = [people for people in enron_data if float(enron_data[people]["total_payments"]) > 0]
n_ntp = n_people - len(tp)
print("folks with NaN total payments: {} ({:.1f}%)".format(n_ntp, n_ntp/n_people*100))

poi_tp = [people for people in poi if float(enron_data[people]["total_payments"]) > 0]
n_poi_ntp = len(poi) - len(poi_tp)
print("POI with NaN total payments: {} ({:.1f}%)".format(n_poi_ntp, n_poi_ntp/n_people*100))
