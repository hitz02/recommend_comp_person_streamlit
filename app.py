
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import scipy
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_model():
  model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
  return model


@st.cache(allow_output_mutation=True)
def load_data():
  company = pd.read_csv('company(1).csv',encoding='latin1')
  company = company[~company['intro'].isna()]
  company.reset_index(drop=True,inplace=True)
  company.columns = ['comp_'+str(col) for col in company.columns]

  person = pd.read_csv('person(1).csv',encoding='latin1')
  person = person[~person['intro'].isna()]
  person.reset_index(drop=True,inplace=True)
  person.columns = ['person_'+str(col) for col in person.columns]
  return company, person


@st.cache(allow_output_mutation=True)
def encode_data(model, company, person):
  comp_intro_enc = model.encode(company['comp_intro'])
  company['comp_intro_encoded'] = comp_intro_enc.tolist()

  person_intro_enc = model.encode(person['person_intro'])
  person['person_intro_encoded'] = person_intro_enc.tolist()
  return company, person


def reco_comp_sbert(company, comp_id, **filter_dict):
  try:
    print('filter_dict: ',filter_dict)
    df = company[(company['comp_company_id']!=comp_id)]
    if len(filter_dict.keys())>0:
      for k,v in filter_dict.items():
        df = df[df[k]==v]

    input_comp_intro = company[company['comp_company_id']==comp_id]['comp_intro_encoded'].iloc[0]
    rest_comp_intro = np.vstack(df['comp_intro_encoded'].values)
    input_comp_intro = np.reshape(input_comp_intro,(1,-1))

    st.write('Details of Company ID selected ->')
    st.dataframe(company[company['comp_company_id']==comp_id][['comp_company_name','comp_intro']])

    distances = scipy.spatial.distance.cdist(input_comp_intro,rest_comp_intro, 'cosine')
    distances = 1-distances
    df['cosine_distance'] = distances[0]

    colnames = ['comp_company_id','comp_company_name','comp_intro'] + list(filter_dict.keys()) + ['cosine_distance']
    st.write('Recommended Companies ->')
    return df.sort_values('cosine_distance',ascending=False)[colnames].head(10).reset_index(drop=True)

  except Exception as e:
      print(str(e))
      return 'Error-> '+str(e)


def reco_person_sbert(person, person_id, **filter_dict):
  try:
    print('filter_dict: ',filter_dict)
    df = person[(person['person_person_id']!=person_id)]

    if len(filter_dict.keys())>0:
      for k,v in filter_dict.items():
        df = df[df[k]==v]

    input_person_intro = person[person['person_person_id']==person_id]['person_intro_encoded'].iloc[0]
    rest_person_intro = np.vstack(df['person_intro_encoded'].values)
    input_person_intro = np.reshape(input_person_intro,(1,-1))

    st.write('Details of Person ID selected ->')
    st.dataframe(person[person['person_person_id']==comp_id][['person_person_name','person_intro','person_industry','person_city','person_country']])

    distances = scipy.spatial.distance.cdist(input_comp_intro,rest_comp_intro, 'cosine')
    distances = 1-distances
    df['cosine_distance'] = distances[0]

    colnames = ['person_person_id','person_person_name','person_intro'] + list(filter_dict.keys()) + ['cosine_distance']
    return df.sort_values('cosine_distance',ascending=False)[colnames].head(10).reset_index(drop=True)

  except Exception as e:
      print(str(e))
      return 'Error-> '+str(e)

def reco_com_person_sbert(person, company, comp_id, **person_filter_dict):
  try:
    df = person.copy()
    print('person_filter_dict: ',person_filter_dict)

    if len(person_filter_dict.keys())>0:
      for k,v in person_filter_dict.items():
        df = df[df[k]==v]

    input_comp_intro = company[company['comp_company_id']==comp_id]['comp_intro_encoded'].iloc[0]
    input_comp_intro = np.reshape(input_comp_intro,(1,-1))
    rest_person_intro = np.vstack(df['person_intro_encoded'].values)
    
    distances = scipy.spatial.distance.cdist(input_comp_intro,rest_person_intro, 'cosine')
    distances = 1-distances
    df['cosine_distance'] = distances[0]

    colnames = ['person_person_id','person_person_name','person_intro','person_gender','person_industry','person_city','person_country','cosine_distance']
    return df.sort_values('cosine_distance',ascending=False)[colnames].head(10).reset_index(drop=True)

  except Exception as e:
      print(str(e))
      return 'Error-> '+str(e)


def main():

  # Load the universal sentence encoder
  model = load_model()
  
  company, person = load_data()

  company, person = encode_data(model, company, person)

  comp_id = st.text_input("Enter Company ID from company file",'',key="comp_id")
  print('comp_id', comp_id)

  comp_filter_options = st.multiselect(
      'Select company or person columns which you wish to use for filtering recommendations',
      ['comp_city','comp_country','comp_industry','comp_province','comp_found_year','comp_company_type_dict'])

  person_filter_options = st.multiselect(
      'Select columns which you wish to use for filtering people recommendations',
      ['person_city', 'person_gender', 'person_province', 'person_country','person_industry'])

  filter_options_dict = {**{filt:'company' for filt in comp_filter_options}, **{filt:'person' for filt in person_filter_options}}
  print('filter_options_dict ',filter_options_dict)
  
  filter_dict = {}
  person_filter_dict = {}
  submit_button = False

  if len(filter_options_dict)>0:
    with st.form("Filter Form"):
      st.write("Set Filters")
      for col, val in filter_options_dict.items():
        if val=='company':
          filter_dict[col] = st.selectbox("Enter {}".format(col),tuple(company[col].unique()))
        else:
          person_filter_dict[col] = st.selectbox("Enter {}".format(col),tuple(person[col].unique()))

      submit_button = st.form_submit_button(label='Submit')


  if comp_id and not submit_button:
    comp_data = reco_comp_sbert(company, comp_id,**{})
    person_data = reco_com_person_sbert(person,company,comp_id, **{})
    if comp_data.shape[0]>0:
      st.dataframe(comp_data)
      st.write('Recommended People matching this company ->')
      st.dataframe(person_data)
    else:
      st.info('No data to show, something wrong')
  elif comp_id and submit_button:
    comp_data = reco_comp_sbert(company, comp_id,**filter_dict)
    person_data = reco_com_person_sbert(person,company,comp_id, **person_filter_dict)
    if comp_data.shape[0]>0:
      st.dataframe(comp_data)
      st.write('Recommended People matching this company ->')
      st.dataframe(person_data)
    else:
      st.info('No data to show, something wrong')
  else:
    st.info('No data to show, check if you have entered company_id')

if __name__ == '__main__':
	main()
