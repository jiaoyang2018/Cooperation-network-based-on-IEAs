# This file contains functions to perform the calculations in the appendix

import numpy  as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite as bi
from datetime import datetime
from imp import reload 
import os
from tqdm import tqdm
import community as com


import weighted_network as wn
import bipcm


# This function is to combine the date of ratification and date of entry into force together.
# Some treaties have the date of ratifiction but others have the date of entry into force, this function extract such information and put them together.

# Input: a dataframe containing the country-treaty relatioships after the data selection
# Output: a dataframe combining the date of ratifiction and date of entry into force together

# Parameters:
# 'df': a dataframe


def combined_date(df):
    
    parties=list(set(df['party_code']))
    dfl=[]

    for c in parties:

        df11=df[df['party_code']==c]

        df22=df11[['treaty_id','date_entry_into_force','date_ratification']]
        df33=df22[df22['date_ratification'].notnull()][['treaty_id','date_ratification']].rename(columns={'date_ratification':'date'})
        df44=df22[df22['date_ratification'].isnull()][['treaty_id','date_entry_into_force']].rename(columns={'date_entry_into_force':'date'})

        df55=pd.concat([df33,df44]).sort_values(by='date')
        df55['party_code']=c

        dfl.append(df55)

    dff=pd.concat(dfl)
    
    return dff

# This function takes into account the age of country-treaty relationships and the last new signatory of treaties and 
# is used to select country-treaty relationships till a specific year.

# Input: a dataframe of the original dataset
# Output: a dataframe containing the selected country-treaty relationships

# Parameters:
# 'df_parties_total': a dataframe, the original dataset
# # 'year': the year of interest
# 'field_id' and 'subject_id' are for the function 'data_selection'.
#  The default value of 'field_id' is None, otherwised it can be '1' for regional treaties or '2' for global treaties.
#  The default values of 'subject_id' is an empty list '[]', or it can be a list of subject ids
# 'membership': bool, True if remove old country-treaty relationships, False otherwise.
# 'last_sig': bool, True if remove treaties with no newsignatories over the past 10, False otherwise.


def data_selection_recent(df_parties_total,year,field_id,subject_ids, membership=False, last_sig=True):# This function select data points until a specific year based on the date_entry into force and date_ratification
    
    # Transform the datetime 
    list_TypeofDates=['date_entry_into_force','date_ratification','date_simple_sigNMture','date_definite_sigNMture','date_withdrawal','date_consent_to_be_bound','date_accession_approv','date_acceptance_approv','date_provisioNMl_application','date_succession','date_reservation']
    for i in list_TypeofDates:
        df_parties_total[i]=pd.to_datetime(df_parties_total[i],format='%d/%m/%Y')

        
    # Remove countries that had deceased   
    if (datetime(year,12,31)<datetime(1918,10,28))|(datetime(year,12,31)>datetime(1992,12,31)):
        df_parties_total=df_parties_total[df_parties_total['party_code']!='CS']
    if (datetime(year,12,31)<datetime(1922,12,30))|(datetime(year,12,31)>datetime(1991,12,26)):
        df_parties_total=df_parties_total[df_parties_total['party_code']!='SU']
    if (datetime(year,12,31)<datetime(1929,1,1))|(datetime(year,12,31)>datetime(2003,1,1)):
        df_parties_total=df_parties_total[df_parties_total['party_code']!='YU']
        
    # Select the type of parties-Here only keep the countries
    df_parties_cat=pd.read_csv('IEA_data/parties_cat.csv',sep=',')
    df_parties_total_cat=pd.merge(df_parties_total,df_parties_cat,left_on='party_code',right_on='code')
    df_country=df_parties_total_cat[df_parties_total_cat['type']==1]

    # Select the field of application: local==1 or global==2
    df_document=pd.read_csv('IEA_data/document.csv')
    df_document_1=df_document[['treaty_id','field_application']]
    df_parties_fields=df_country.merge(df_document_1, on='treaty_id',how='left')

    if field_id==None:
        df_field=df_parties_fields
    else:
        df_field=df_parties_fields[df_parties_fields['field_application']==field_id]

    # Select the subjects of treaties
    df_subjects=pd.read_csv('IEA_data/subject_rel.csv')
    df_field_subjects=df_field.merge(df_subjects,how='left',on='treaty_id')
    
    
    if len(subject_ids)==0:
        df_one_subject=df_field
    else:
        list_df=[]
        for i in subject_ids:

            df_one_subject_1=df_field_subjects[df_field_subjects['subject_id']==i]
            list_df.append(df_one_subject_1)
            
        df_one_subject=pd.concat(list_df)
        df_one_subject.drop_duplicates(subset=['treaty_id', 'party_code'], inplace=True)


    # Select data points according to the dates of ratification and entry into force
    if membership==True:    
        
        df_parties_1=df_one_subject[((df_one_subject['date_ratification']>datetime(year-15,12,31))&(df_one_subject['date_ratification']<=datetime(year,12,31))&(df_one_subject['date_withdrawal']>datetime(year,12,31)))|((df_one_subject['date_ratification']>datetime(year-15,12,31))&(df_one_subject['date_ratification']<=datetime(year,12,31))&(df_one_subject['date_withdrawal'].isnull()))]
        df_parties_2=df_one_subject[(df_one_subject['date_ratification'].isnull()&(df_one_subject['date_entry_into_force']>datetime(year-15,12,31))&(df_one_subject['date_entry_into_force']<=datetime(year,12,31))&(df_one_subject['date_withdrawal']>datetime(year,12,31)))|(df_one_subject['date_ratification'].isnull()&(df_one_subject['date_entry_into_force']>datetime(year-15,12,31))&(df_one_subject['date_entry_into_force']<=datetime(year,12,31))&df_one_subject['date_withdrawal'].isnull())]    
        df_parties= df_parties_1.append(df_parties_2)
    

    # Remove treaties ten years old after its last signatory
    if last_sig==True:
        
        df_com=pd.read_csv('IEA_data/combined_entry_rafi.csv')
        
        df_last=df_com.sort_values(by=['treaty_id','date'], ascending=False).drop_duplicates(subset=['treaty_id'])
        df1=df_one_subject.merge(df_last[['treaty_id','date']], how='left', on='treaty_id')
        df1['date']=pd.to_datetime(df1['date'],format='%Y-%m-%d')
        df1['obs']=datetime(year,12,31)
        
        df3=df1[(df1['date']>datetime(year,12,31))]
        df4=df1[(df1['date']<=datetime(year,12,31))&(df1['date']>datetime(year-10,12,31))]
        df2=pd.concat([df3,df4])

        df_parties_1=df2[((df2['date_ratification']<=datetime(year,12,31))&(df2['date_withdrawal']>datetime(year,12,31)))|((df2['date_ratification']<=datetime(year,12,31))&(df2['date_withdrawal'].isnull()))]
        df_parties_2=df2[(df2['date_ratification'].isnull()&(df2['date_entry_into_force']<=datetime(year,12,31))&(df2['date_withdrawal']>datetime(year,12,31)))|(df2['date_ratification'].isnull()&(df2['date_entry_into_force']<=datetime(year,12,31))&df2['date_withdrawal'].isnull())]    
        df_parties= df_parties_1.append(df_parties_2)

        

    return df_parties