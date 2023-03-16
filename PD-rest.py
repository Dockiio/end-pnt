import joblib
import uvicorn
from fastapi import FastAPI, Query
import pandas as pd
import pickle
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from typing import List

app = FastAPI()

warnings.filterwarnings("ignore")


def encode_data(dataset , data_dict_weigth):
    cols = dataset.columns
    for columnName in cols:
        for i in range(len(dataset[columnName])):
            try:
            #print(data_dict[data2[columnName][i]]["weight"])
                dataset[columnName][i] = data_dict[dataset[columnName][i]]["weight"]
            except:
                pass
    dataset = dataset.fillna(0) # put empty cell to 0
    dataset = dataset.replace("foul_smell_of_urine" , 5)
    dataset = dataset.replace("dischromic__patches" , 6)
    dataset = dataset.replace("spotting__urination" , 6)
    return dataset

data_dict = pickle.load(open('disease_data_dict.pkl','rb'))
saved_sympt_dict = pickle.load(open('sympts_data_dict.pkl','rb'))
symp_lstt = [i for i in saved_sympt_dict]

l = ['Fungal_infection', 'Allergy', 'GERD', 'Chronic_cholestasis',
       'Drug_Reaction', 'Peptic_ulcer_diseae', 'AIDS', 'Diabetes',
       'Gastroenteritis', 'Bronchial_Asthma', 'Hypertension', 'Migraine',
       'Cervical_spondylosis', 'Paralysis_(brain_hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken_pox', 'Dengue', 'Typhoid', 'hepatitis_A',
       'Hepatitis_B', 'Hepatitis_C', 'Hepatitis_D', 'Hepatitis_E',
       'Alcoholic_hepatitis', 'Tuberculosis', 'Common_Cold', 'Pneumonia',
       'Dimorphic_hemmorhoids(piles)', 'Heart_attack', 'Varicose_veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo)_Paroymsal__Positional_Vertigo', 'Acne',
       'Urinary_tract_infection', 'Psoriasis', 'Impetigo']

print(symp_lstt)

columns=['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5',
       'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10',
       'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15',
       'Symptom_16', 'Symptom_17']

inpt_dict = {}
symp_lst_o = [0] * len(columns)
# print(symp_lst)

@app.get("/predict")
def predict_sentiment(symp_lst: List[str] = Query(None)):

	# for col in range (len(columns)):
	# print(symp_lst)
	for i in enumerate(symp_lst):
		# for col in range(len(columns)):
		
		symp_lst_o[(i[0])] = i[1]
			# break
		#continue
	qw = pd.DataFrame([symp_lst_o], columns=columns)
	print(qw)
	df = encode_data(qw, data_dict)
	
	df = tf.convert_to_tensor(np.asarray(df.values).astype(np.float32()))
	
	model = load_model('disease_model1.h5')
	
	return {"pred":(l[np.argmax(model.predict(df))])}
