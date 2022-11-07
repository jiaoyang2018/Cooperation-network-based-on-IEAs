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


# This function is used to project on a specific layer of a bipartite network by considering the importance of citations and media reports into the analysis by revising
# the formula for calculating link weights.

# Input: a bipartite network B
# Output: a one-mode projection G

#Parameters:
# 'B': a bipartite network
# 'layer':'top' for treaties or 'bottom ' for countries
# 'method': 'log', 'linear' or 'exp'
# 'citation': bool, True if consider the citations of treaties, False otherwise.
# 'media': bool, True if consider the media coverage of treaties, False otherwise.

def importance_weighted_projection(B,layer, year,method, citation=False, media=True ):
    
    r=0.02 # r is the discount rate ranging from 0.02 to 0.05
    
    nodes_top= {n for n,d in B.nodes(data=True) if d['bipartite']==0}
    nodes_bottom= set(B) - nodes_top


    if layer=='bottom':
        nodes=nodes_bottom
    else:
        nodes=nodes_top

    if B.is_directed():
            pred = B.pred
            G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n, B.nodes[n]) for n in nodes)
    
    if media==True:
        #df_media=pd.read_excel('FACTIVA_IEA/Treaty_List Main 3.xlsx')
        df_media=pd.read_excel('FACTIVA_IEA/Treaty_List Main 3.xlsx')

        col_list= list(df_media)
        cols=col_list[5:5+(year-1969+1)]
        df_media['sum'+str(year)] = df_media[cols].sum(axis=1)
        
        if method=='log':
            df_media['log']=df_media['sum'+str(year)].map(lambda x:np.log(x+1)+1)
        if method=='linear':
            df_media['log']=df_media['sum'+str(year)].map(lambda x:x+1)
        if method=='exp':
            df_media['log']=df_media['sum'+str(year)].map(lambda x:(1+r)**x)


        dic_media=dict(zip(df_media['treaty_id'],df_media['log']))
                
    if citation==True:
        df_cite=pd.read_csv('IEA_data/cites_treaty.csv')
        df_doc['date_treaty']=pd.to_datetime(df_doc['date_treaty'],format='%d/%m/%Y')

        df1=df_cite.merge(df_doc[['treaty_id','date_treaty']], how='left', on='treaty_id')
        #df2=df1.merge(df_doc[['treaty_id','date_treaty']], how='left', left_on='treaty_cited', right_on='treaty_id')

        df2=df1[df1['date_treaty']<=datetime(year,12,31)]
        dfnc=df2.groupby(by='treaty_cited').count()
        dic_cite=dict(dfnc['treaty_id'])
        
        df3=df_parties_total.sort_values(by='treaty_id').drop_duplicates(subset='treaty_id')[['treaty_id']]

        df4=pd.DataFrame({'num':dic_cite}).reset_index().merge(df3, how='outer', right_on='treaty_id', left_on='index')
        df5=df4[df4['treaty_id'].notnull()]
        df5.fillna(value=0,inplace=True)
        
        if method=='log':
        # log increase
            df5['log']=df5['num'].map(lambda x:np.log(x+1)+1)
        # linear increase
        if method=='linear':
            df5['log']=df5['num'].map(lambda x:x+1)
        if method=='exp':
            df5['log']=df5['num'].map(lambda x:(1+r)**x)


        dic_citation=dict(zip(df5['treaty_id'],df5['log']))


    for u in nodes:
        unbrs = set(B[u])# treaties signed by u
        nbrs2 = {n for nbr in unbrs for n in B[nbr] if n != u} # set of countries signing the same treaty as u
        for v in nbrs2:
            vnbrs = set(pred[v]) # treaties signed by v
            degree={n:len(B[n]) for n in unbrs & vnbrs}
        
        # when considering the date of availability using date_treaties
            if citation==True:
                
                common = ((degree[n], dic_citation[n]) for n in unbrs & vnbrs)
                weight = sum(1.0 / (deg - 1)*(1+r)**(num) for deg, num in common if deg > 1)
                G.add_edge(u, v, weight=weight)
                
        # when considering the date of the last sig using year_last;  
            if media==True:
                
                common = ((degree[n], dic_media[n]) for n in unbrs & vnbrs)
                
                weight = sum(1.0 / (deg - 1)*(1+r)**(num) for deg, num in common if deg > 1)

                G.add_edge(u, v, weight=weight)
            
            
            
    return G

# This function is to calculate the global measures of networks when considering the activity of treaties in the process of projection

# Input: a dataframe of 'parties.csv'
# Output: a dataframe containing the global measures of the network in different years

# Parameters:
# 'year_list': a list of years
# 'layer': which layer to project on, and should be 'top'(treaty layer), 'bottom' (country layer)
# 'constraint':  'Ture' if 'layer' is set to be 'bottom', or 'False' if 'layer' is set to be 'top'
# 'depository_excluded' is a list of depository ids; If it is not empty, then treaties belonging to this depository are excluded from the network 
# 'treaty_excluded' is a treaty id, if treaty_excluded is not None, then the treaty will be excluded from the dataset
# 'weighted': True or False
# 'field_id' and 'subject_id' are for the function 'data_selection'.
#  The default value of 'field_id' is None, otherwised it can be '1' for regional treaties or '2' for global treaties.
#  The default values of 'subject_id' is an empty list '[]', or it can be a list of subject ids
# 'membership': bool, True if remove old country-treaty relationships, False otherwise.
# 'last_sig': bool, True if remove treaties with no newsignatories over the past 10, False otherwise.

def significant_global_measures_bipcm_activity(year_list, field_id, subject_id, layer, constraint, depository_excluded, membership, last_sig, weighted=True,):
    
    dic_1={}
    dic_2={}
    dic_3={}
    dic_4={}
    dic_5={}
    dic_6={}
    dic_7={}
    dic_8={}
    dic_9={}
    dic_10={}
        
    
    df_parties_total=pd.read_csv("IEA_data/parties.csv",sep=",")

    list_TypeofDates=['date_entry_into_force','date_ratification','date_simple_sigNMture','date_definite_sigNMture','date_withdrawal','date_consent_to_be_bound','date_accession_approv','date_acceptance_approv','date_provisioNMl_application','date_succession','date_reservation']
    for i in list_TypeofDates:
        df_parties_total[i]=pd.to_datetime(df_parties_total[i],format='%d/%m/%Y')
    df_parties_1=df_parties_total[(df_parties_total['date_entry_into_force']<datetime(1947,12,31))|(df_parties_total['date_ratification']<datetime(1947,12,31))]

    old_treaties=set(df_parties_1['treaty_id'])
    for i in old_treaties:
        df_parties_total=df_parties_total[df_parties_total['treaty_id']!=i]
           


    df_depo_real=pd.read_csv('IEA_data/depository_rel.csv',sep=',')
    if len(depository_excluded)!=0:
        list_depo=[]
        for i in depository_excluded:
            df1=df_depo_real[df_depo_real['depository_id']==i]
            list_depo.append(df1)

        df2=pd.concat(list_depo)

        df3=df_parties_total.merge(df2,how='left', on='treaty_id')
        df_parties_2=df3[df3['depository_id'].isnull()]

            
    else:
        df_parties_2=df_parties_total
    
    
    for year in tqdm(year_list):

        # Select data points based on memeberships and the date of the last signatories.
        df_parties=data_selection_recent(df_parties_2,year,field_id,subject_id, membership, last_sig)

        B=wn.bipartite_network(df_parties)# the top nodes are the treaties
        
        G_weight=wn.projection(B,layer)# choose which layer to project on

        G_matrix=nx.to_numpy_matrix(G_weight,weight='weight').A

        num_nodes=G_weight.number_of_nodes()

        # obtain the p-values
        nodes_treaties= {n for n,d in B.nodes(data=True) if d['bipartite']==0}
        nodes_parties= set(B) - nodes_treaties
        
        if len(nodes_parties)==0 | len(nodes_treaties)==0:
            continue
        

        Sparse_Matrix_B=bi.biadjacency_matrix(B,list(nodes_parties), list(nodes_treaties))# row_order, colunm_order
        Matrix_B=Sparse_Matrix_B.A # the rows are the countries and the columns are the treaties
        B_pcm = bipcm.BiPCM(Matrix_B, constraint)# choose a null model

        p_value_countries_bipcm=B_pcm.lambda_motifs_main(bip_set=constraint, write=False)

        # test if the p-value is significant by the False discovery rate
        p_value_list=[]

        for i in (range(0,num_nodes)):
            for j in range(0,num_nodes):
                if p_value_countries_bipcm[i][j]!=0:
                    p_value_list.append(p_value_countries_bipcm[i][j])

        p_value_list.sort(reverse=True) # decrease gradually
        order=None
        for k in range(0,len(p_value_list)):
            if p_value_list[k]<=(len(p_value_list)-k)*0.01/len(p_value_list):
                significant=p_value_list[k]
                order=k # this is the number of links that should be removed
                break
        if order==None:
            continue

         # transfer the p-value matrix to link matrix
        for i in range(0,num_nodes):
            for j in range(0,num_nodes):
                if p_value_countries_bipcm[i][j]> p_value_list[order]:
                    p_value_countries_bipcm[i][j]=0

        for i in range(0,num_nodes):
            for j in range(0,num_nodes):
                if p_value_countries_bipcm[i][j]==0:
                    G_matrix[i][j]=0   

        G_bipcm_sig=nx.from_numpy_matrix(G_matrix, create_using=None)

        node_list=list(nx.nodes(G_bipcm_sig))
        if layer=='bottom':
            G_sig=nx.relabel_nodes(G_bipcm_sig, dict(zip(node_list,list(nodes_parties))))
        else:
            G_sig=nx.relabel_nodes(G_bipcm_sig, dict(zip(node_list,list(nodes_treaties))))


        degrees=dict(nx.degree(G_sig))
        for k,v in degrees.items():
            if v==0:
                G_sig.remove_node(k)

        dic_1[year]= G_sig.number_of_nodes()
        dic_2[year]= G_sig.number_of_edges()
        dic_3[year]= nx.density(G_sig)
        dic_4[year]= nx.number_connected_components(G_sig)
        dic_5[year]= wn.fraction_largest_component(G_sig)
        dic_6[year]= wn.average(dict(nx.degree(G_sig)).values())

        if weighted:
            dic_7[year]= wn.average(dict(nx.degree(G_sig, weight='weight')).values())
            dic_8[year], dic_9[year]= wn.average_shortest_path_length(G_sig)
        # dic_8 is the dict of diameter, as the function average_shortest_path_length can return diameter as well
        #dic_8[year]= wn.shortest_path_lenghth_weighted(G_sig).max().max()
            dic_10[year]= wn.global_clustering_coefficient(G_sig)
        

            global_measures=['number_of_nodes','number_of_links','density','number_of_components','fraction_of_largest_component', 'average_degree','average_strength','average_weighted_shortest_path_length','diameter','weighted_global_clustering_coefficient']
      
            list_dict=[dic_1,dic_2,dic_3,dic_4,dic_5,dic_6,dic_7,dic_8,dic_9,dic_10]
            dict_all=dict(zip(global_measures,list_dict))
            df=pd.DataFrame(dict_all)
            df_1=df.reset_index().rename(columns={'index':'year'})

        else:
            dic_7[year], dic_8[year]= wn.average_shortest_path_length_unweighted(G_sig)
            dic_9[year] = nx.transitivity(G_sig)
            global_measures=['number_of_nodes','number_of_links','density','number_of_components','fraction_of_largest_component', 'average_degree','average_shortest_path_length','diameter','global_clustering_coefficient']
            list_dict=[dic_1,dic_2,dic_3,dic_4,dic_5,dic_6,dic_7,dic_8,dic_9]
            dict_all=dict(zip(global_measures,list_dict))
            df=pd.DataFrame(dict_all)
            df_1=df.reset_index().rename(columns={'index':'year'})


    return df_1



# This function is to calculate the global measures of networks when considering the importance of treaties in the process of projection

# Input: a dataframe of 'parties.csv'
# Output: a dataframe containing the global measures of the network in different years

# Parameters:
# 'year_list': a list of years
# 'layer': which layer to project on, and should be 'top'(treaty layer), 'bottom' (country layer)
# 'constraint':  'Ture' if 'layer' is set to be 'bottom', or 'False' if 'layer' is set to be 'top'
# 'depository_excluded' is a list of depository ids; If it is not empty, then treaties belonging to this depository are excluded from the network 
# 'treaty_excluded' is a treaty id, if treaty_excluded is not None, then the treaty will be excluded from the dataset
# 'weighted': True or False
# 'field_id' and 'subject_id' are for the function 'data_selection'.
#  The default value of 'field_id' is None, otherwised it can be '1' for regional treaties or '2' for global treaties.
#  The default values of 'subject_id' is an empty list '[]', or it can be a list of subject ids
# 'projection_method': 'log', 'linear' or 'exp'; used in function 'importance_weighted_projection'

def significant_global_measures_bipcm_importance(year_list, field_id,subject_id, layer, constraint, depository_excluded, projection_method, weighted=True):
    
    dic_1={}
    dic_2={}
    dic_3={}
    dic_4={}
    dic_5={}
    dic_6={}
    dic_7={}
    dic_8={}
    dic_9={}
    dic_10={}
        
    
    df_parties_total=pd.read_csv("IEA_data/parties.csv",sep=",")

    list_TypeofDates=['date_entry_into_force','date_ratification','date_simple_sigNMture','date_definite_sigNMture','date_withdrawal','date_consent_to_be_bound','date_accession_approv','date_acceptance_approv','date_provisioNMl_application','date_succession','date_reservation']
    for i in list_TypeofDates:
        df_parties_total[i]=pd.to_datetime(df_parties_total[i],format='%d/%m/%Y')
    df_parties_1=df_parties_total[(df_parties_total['date_entry_into_force']<datetime(1947,12,31))|(df_parties_total['date_ratification']<datetime(1947,12,31))]

    old_treaties=set(df_parties_1['treaty_id'])
    for i in old_treaties:
        df_parties_total=df_parties_total[df_parties_total['treaty_id']!=i]      


    df_depo_real=pd.read_csv('IEA_data/depository_rel.csv',sep=',')
    if len(depository_excluded)!=0:
        list_depo=[]
        for i in depository_excluded:
            df1=df_depo_real[df_depo_real['depository_id']==i]
            list_depo.append(df1)

        df2=pd.concat(list_depo)

        df3=df_parties_total.merge(df2,how='left', on='treaty_id')
        df_parties_2=df3[df3['depository_id'].isnull()]

            
    else:
        df_parties_2=df_parties_total
    
    
    for year in tqdm(year_list):

        
        df_parties=wn.data_selection(df_parties_2,year,field_id,subject_id)

        B=wn.bipartite_network(df_parties)# the top nodes are the treaties
        
        G_weight=importance_weighted_projection(B,layer, year, projection_method)# choose which layer to project on

        G_matrix=nx.to_numpy_matrix(G_weight,weight='weight').A

        num_nodes=G_weight.number_of_nodes()

        # obtain the p-values
        nodes_treaties= {n for n,d in B.nodes(data=True) if d['bipartite']==0}
        nodes_parties= set(B) - nodes_treaties
        
        if len(nodes_parties)==0 | len(nodes_treaties)==0:
            continue
        

        Sparse_Matrix_B=bi.biadjacency_matrix(B,list(nodes_parties), list(nodes_treaties))# row_order, colunm_order
        Matrix_B=Sparse_Matrix_B.A # the rows are the countries and the columns are the treaties
        B_pcm = bipcm.BiPCM(Matrix_B, constraint)# choose a null model

        p_value_countries_bipcm=B_pcm.lambda_motifs_main(bip_set=constraint, write=False)

        # test if the p-value is significant by the False discovery rate
        p_value_list=[]

        for i in (range(0,num_nodes)):
            for j in range(0,num_nodes):
                if p_value_countries_bipcm[i][j]!=0:
                    p_value_list.append(p_value_countries_bipcm[i][j])

        p_value_list.sort(reverse=True) # decrease gradually
        order=None
        for k in range(0,len(p_value_list)):
            if p_value_list[k]<=(len(p_value_list)-k)*0.01/len(p_value_list):
                significant=p_value_list[k]
                order=k # this is the number of links that should be removed
                break
        if order==None:
            continue

         # transfer the p-value matrix to link matrix
        for i in range(0,num_nodes):
            for j in range(0,num_nodes):
                if p_value_countries_bipcm[i][j]> p_value_list[order]:
                    p_value_countries_bipcm[i][j]=0

        for i in range(0,num_nodes):
            for j in range(0,num_nodes):
                if p_value_countries_bipcm[i][j]==0:
                    G_matrix[i][j]=0   

        G_bipcm_sig=nx.from_numpy_matrix(G_matrix, create_using=None)

        node_list=list(nx.nodes(G_bipcm_sig))
        if layer=='bottom':
            G_sig=nx.relabel_nodes(G_bipcm_sig, dict(zip(node_list,list(nodes_parties))))
        else:
            G_sig=nx.relabel_nodes(G_bipcm_sig, dict(zip(node_list,list(nodes_treaties))))


        degrees=dict(nx.degree(G_sig))
        for k,v in degrees.items():
            if v==0:
                G_sig.remove_node(k)

        dic_1[year]= G_sig.number_of_nodes()
        dic_2[year]= G_sig.number_of_edges()
        dic_3[year]= nx.density(G_sig)
        dic_4[year]= nx.number_connected_components(G_sig)
        dic_5[year]= wn.fraction_largest_component(G_sig)
        dic_6[year]= wn.average(dict(nx.degree(G_sig)).values())

        if weighted:
            dic_7[year]= wn.average(dict(nx.degree(G_sig, weight='weight')).values())
            dic_8[year], dic_9[year]= wn.average_shortest_path_length(G_sig)
        # dic_8 is the dict of diameter, as the function average_shortest_path_length can return diameter as well
        #dic_8[year]= wn.shortest_path_lenghth_weighted(G_sig).max().max()
            dic_10[year]= wn.global_clustering_coefficient(G_sig)
        

            global_measures=['number_of_nodes','number_of_links','density','number_of_components','fraction_of_largest_component', 'average_degree','average_strength','average_weighted_shortest_path_length','diameter','weighted_global_clustering_coefficient']
      
            list_dict=[dic_1,dic_2,dic_3,dic_4,dic_5,dic_6,dic_7,dic_8,dic_9,dic_10]
            dict_all=dict(zip(global_measures,list_dict))
            df=pd.DataFrame(dict_all)
            df_1=df.reset_index().rename(columns={'index':'year'})

        else:
            dic_7[year], dic_8[year]= wn.average_shortest_path_length_unweighted(G_sig)
            dic_9[year] = nx.transitivity(G_sig)
            global_measures=['number_of_nodes','number_of_links','density','number_of_components','fraction_of_largest_component', 'average_degree','average_shortest_path_length','diameter','global_clustering_coefficient']
            list_dict=[dic_1,dic_2,dic_3,dic_4,dic_5,dic_6,dic_7,dic_8,dic_9]
            dict_all=dict(zip(global_measures,list_dict))
            df=pd.DataFrame(dict_all)
            df_1=df.reset_index().rename(columns={'index':'year'})


    return df_1


# This funtion aims to consider the role of the European Union.
# In 65 of all the treateis the EU signed the agreement ahead of the majority of member states that were part of the block at the time.
# In addition, there are 5 agreements which only have the EU, but not individual member countries, as a signatory. We
# consider these 70 treaties as driven by the EU. They are assigned to the EU as an additional
# node in the cooperation network. The remaining 52 agreements, as well as any IEAs not
# signed by the EU, are assigned to individual member countries as before.

# Input: a dataframe of 'parties.csv'
# Output: a dataframe containing the global measures of the network in different years

# Parameters:
# 'year_list': a list of years
# 'layer': which layer to project on, and should be 'top'(treaty layer), 'bottom' (country layer)
# 'constraint':  'Ture' if 'layer' is set to be 'bottom', or 'False' if 'layer' is set to be 'top'
# 'weighted': True or False
# 'field_id' and 'subject_id' are for the function 'data_selection'.
#  The default value of 'field_id' is None, otherwised it can be '1' for regional treaties or '2' for global treaties.
#  The default values of 'subject_id' is an empty list '[]', or it can be a list of subject ids
# 'projection_method': 'log', 'linear' or 'exp'; used in function 'importance_weighted_projection'


def significant_network_EU(year_list,field_id,subject_id,layer,constraint,weighted):
    
    
    df_parties_total=pd.read_csv('IEA_data/parties.csv')

    list_TypeofDates=['date_entry_into_force','date_ratification','date_simple_sigNMture','date_definite_sigNMture','date_withdrawal','date_consent_to_be_bound','date_accession_approv','date_acceptance_approv','date_provisioNMl_application','date_succession','date_reservation']
    for i in list_TypeofDates:
        df_parties_total[i]=pd.to_datetime(df_parties_total[i],format='%d/%m/%Y')
    df_parties_1=df_parties_total[(df_parties_total['date_entry_into_force']<=datetime(1947,12,31))|(df_parties_total['date_ratification']<=datetime(1947,12,31))]

    # not exclude treaties after 2015, so need to use the function wn.data_selection()
    old_treaties=set(df_parties_1['treaty_id'])
    for i in old_treaties:
        df_parties_total=df_parties_total[df_parties_total['treaty_id']!=i]

    df_parties_total=df_parties_total[(df_parties_total['date_entry_into_force'].notnull())|(df_parties_total['date_ratification'].notnull())]


    
    df_EU_treaties=pd.read_csv('IEA_data/treaties_with_EU_correct.csv')
    df_EU_treaties=df_EU_treaties.rename(columns={'index':'treaty_id'})
    EU_countries=pd.read_csv('IEA_data/EU_countries.csv')
    EU_countries['year_of_join']=pd.to_datetime(EU_countries['year_of_join'],format='%d/%m/%Y')

    
    EU_countries['EU']=True

    treaties_eu=list(df_EU_treaties[df_EU_treaties['test']>=0]['treaty_id'])

    
    G_dic={}
    
    for year in tqdm(year_list):
        
        df_parties=wn.data_selection(df_parties_total,year,field_id,subject_id)
        
        df_parties.drop(columns=['code','name','type','field_application'], inplace=True)
        
        df_parties_com=combined_date(df_parties) # including EU countries       

        
        # for each EU dominate treaty, add EU as a party and remove other countries 
        for i in treaties_eu:

            df1=df_parties_total[df_parties_total['treaty_id']==i]
            

            # Select data points according to the dates of ratification and entry into force
            df_parties_1=df1[((df1['date_ratification']<datetime(year,12,31))&(df1['date_withdrawal']>datetime(year,12,31)))|((df1['date_ratification']<datetime(year,12,31))&(df1['date_withdrawal'].isnull()))]
            df_parties_2=df1[(df1['date_ratification'].isnull()&(df1['date_entry_into_force']<datetime(year,12,31))&(df1['date_withdrawal']>datetime(year,12,31)))|(df1['date_ratification'].isnull()&(df1['date_entry_into_force']<datetime(year,12,31))&df1['date_withdrawal'].isnull())]    
            df12= df_parties_1.append(df_parties_2)
            
            if len(df12)==0:
                continue
            else:

                # get the observations for the EU

                dfl=combined_date(df12)
                df2=dfl[dfl['party_code']=='EU']

                # get the observations for European countries
                df3=dfl.merge(EU_countries, how='left', left_on='party_code', right_on='code')
                df3['interval']=df3.apply(lambda x:(x['date']-x['year_of_join']).days/356, axis=1)
                df4=df3[df3['interval']>=0].drop(columns=['country_name','code', 'year_of_join','EU','interval'])

                df_parties_com=pd.concat([df_parties_com,df2,df4,df4]).drop_duplicates(keep=False)# df_parties_com + df2 - df4


        B=wn.bipartite_network(df_parties_com)# the top nodes are the treaties
        G_weight=wn.projection(B,layer)# choose which layer to project on
        G_matrix=nx.to_numpy_matrix(G_weight,weight='weight').A

        num_nodes=G_weight.number_of_nodes()

        # obtain the p-values
        nodes_treaties= {n for n,d in B.nodes(data=True) if d['bipartite']==0}
        nodes_parties= set(B) - nodes_treaties

        Sparse_Matrix_B=bi.biadjacency_matrix(B,list(nodes_parties), list(nodes_treaties))# row_order, colunm_order
        Matrix_B=Sparse_Matrix_B.A # the rows are the countries and the columns are the treaties
        B_pcm = bipcm.BiPCM(Matrix_B, constraint)# choose a null model

        p_value_countries_bipcm=B_pcm.lambda_motifs_main(bip_set=constraint, write=False)

        # test if the p-value is significant by the False discovery rate
        p_value_list=[]

        for i in (range(0,num_nodes)):
            for j in range(0,num_nodes):
                if p_value_countries_bipcm[i][j]!=0:
                    p_value_list.append(p_value_countries_bipcm[i][j])

        p_value_list.sort(reverse=True) # decrease gradually
        order=None
        for k in range(0,len(p_value_list)):
            if p_value_list[k]<=(len(p_value_list)-k)*0.01/len(p_value_list):
                significant=p_value_list[k]
                order=k # this is the number of links that should be removed
                break
        if order==None:
            continue

         # transfer the p-value matrix to link matrix
        for i in range(0,num_nodes):
            for j in range(0,num_nodes):
                if p_value_countries_bipcm[i][j]> p_value_list[order]:
                    p_value_countries_bipcm[i][j]=0

        for i in range(0,num_nodes):
            for j in range(0,num_nodes):
                if p_value_countries_bipcm[i][j]==0:
                    G_matrix[i][j]=0   

        G_bipcm_sig=nx.from_numpy_matrix(G_matrix, create_using=None)

        node_list=list(nx.nodes(G_bipcm_sig))
        if layer=='bottom':
            G_sig=nx.relabel_nodes(G_bipcm_sig, dict(zip(node_list,list(nodes_parties))))
        else:
            G_sig=nx.relabel_nodes(G_bipcm_sig, dict(zip(node_list,list(nodes_treaties))))

        degrees=dict(nx.degree(G_sig))
        for k,v in degrees.items():
            if v==0:
                G_sig.remove_node(k)

     
        if layer=='top':
            df_1=pd.DataFrame()
            df_1['treaty_id']=list(nx.nodes(G_sig))
            df_2=pd.merge(df_1,df_depository,how='outer',on='treaty_id')
            dic_treaty_depository=dict(zip(df_2['treaty_id'],df_2['depository_id']))
            nx.set_node_attributes(G_sig, dic_treaty_depository, name='depository_id')
            
        dic_degree=dict(nx.degree(G_sig))
        dic_strength=dict(nx.degree(G_sig,weight='weight'))
        
        if weighted==True:
            dic_betweenness_centrality=wn.betweenness_centrality_weighted(G_sig)
            dic_closeness_centrality=wn.closeness_centrality_weighted(G_sig)
            #dic_local_clustering_coefficient=wn.local_clustering_coefficient(G_sig)
            #dic_eigenvector=nx.eigenvector_centrality(G_sig,max_iter=200,weight='weight')
        if weighted==False:
            dic_betweenness_centrality=nx.betweenness_centrality(G_sig)
            dic_closeness_centrality=nx.closeness_centrality(G_sig)
            
        
        nx.set_node_attributes(G_sig, year, name='year')
        nx.set_node_attributes(G_sig, dic_degree, name='degree')
        nx.set_node_attributes(G_sig, dic_strength, name='strength')
        nx.set_node_attributes(G_sig, dic_closeness_centrality, name='closeness_centrality')
        nx.set_node_attributes(G_sig, dic_betweenness_centrality, name='betweenness_centrality')
        
        
        G_dic[year]=G_sig
        
    
    return G_dic

