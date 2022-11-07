# This file contains functions used to generate statistically significant networks and 
# calculate local and global measures of networks in the main paper

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


# This function is to obatain the statistically significant projection of a bipartite network using bipcm
# This function doesn't compute measures, just to obtain the networks.

# Input: country-treay relationships in parties.csv
# Output: a dict; networks in different years


# Parameters:
# 'year_list': a list of years
# 'layer': which layer to project on, and should be 'top'(treaty layer), 'bottom' (country layer)
# 'constraint':  'Ture' if 'layer' is set to be 'bottom', or 'False' if 'layer' is set to be 'top'.
# 'depository_id' is a a depository id; If it is not zero, then treaties belonging to this depository are excluded from the network. 
# 'treaty_excluded' is a treaty id, if treaty_excluded is not None, then the treaty will be excluded from the dataset
# 'weighted': True or False
# 'field_id' and 'subject_id' are for the function 'data_selection'.
#  The default value of 'field_id' is None, otherwised it can be '1' for regional treaties or '2' for global treaties.
#  The default values of 'subject_id' is an empty list '[]', or it can be a list of subject ids


def significant_network_bipcm_Gs(year_list,field_id,subject_id,layer,constraint,depository_id,treaty_excluded,weighted):
    
    
    df_parties_total=pd.read_csv("IEA_data/parties.csv",sep=",")

    list_TypeofDates=['date_entry_into_force','date_ratification','date_simple_sigNMture','date_definite_sigNMture','date_withdrawal','date_consent_to_be_bound','date_accession_approv','date_acceptance_approv','date_provisioNMl_application','date_succession','date_reservation']
    for i in list_TypeofDates:
        df_parties_total[i]=pd.to_datetime(df_parties_total[i],format='%d/%m/%Y')

    df_parties_1=df_parties_total[(df_parties_total['date_entry_into_force']<datetime(1947,12,31))|(df_parties_total['date_ratification']<datetime(1947,12,31))]

    old_treaties=set(df_parties_1['treaty_id'])
    for i in old_treaties:
        df_parties_total=df_parties_total[df_parties_total['treaty_id']!=i]
    
    df_depository=pd.read_csv('IEA_data/depository_rel.csv',sep=',')
    if depository_id!=0:
            df_treaty_id=df_depository[df_depository['depository_id']==depository_id]    
            df_parties_1=pd.merge(df_parties_total,df_treaty_id,how='left')
            df_parties_2=df_parties_1[df_parties_1['depository_id']!=depository_id]
    else:
        df_parties_2=df_parties_total
        
    if treaty_excluded==None:
        df_parties_3=df_parties_2
    else: 
        df_parties_3=df_parties_2[df_parties_2['treaty_id']!=treaty_excluded]
    
    
    G_dic={}
    
    for year in tqdm(year_list):
        
        df_parties=wn.data_selection(df_parties_3,year,field_id,subject_id)
        B=wn.bipartite_network(df_parties)# the top nodes are the treaties
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
        # The lower triangular part (including the diagonal) of the returned matrix is set to zero.

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

        G_dic[year]=G_sig
        
    
    return G_dic



# This function is to obatain the statistically significant projection of a bipartite network using bipcm.
# This function is used to obtain the networks and local measures of nodes in the network.

# Input: country-treay relationships in parties.csv
# Output: a dict; networks in different years with local measures


# Parameters:
# 'year_list': a list of years
# 'layer': which layer to project on, and should be 'top'(treaty layer), 'bottom' (country layer)
# 'constraint':  'Ture' if 'layer' is set to be 'bottom', or 'False' if 'layer' is set to be 'top'.
# 'depository_excluded' is a a depository id; If it is not zero, then treaties belonging to this depository are excluded from the network. 
# 'treaty_excluded' is a treaty id, if treaty_excluded is not None, then the treaty will be excluded from the dataset
# 'weighted': True or False
# 'field_id' and 'subject_id' are for the function 'data_selection'.
#  The default value of 'field_id' is None, otherwised it can be '1' for regional treaties or '2' for global treaties.
#  The default values of 'subject_id' is an empty list '[]', or it can be a list of subject ids

def significant_network_bipcm(year_list,field_id,subject_id,layer,constraint,subject_excluded,depository_excluded,treaty_excluded,weighted):
    
    
    df_parties_total=pd.read_csv("IEA_data/parties.csv",sep=",")
    df_depository=pd.read_csv('IEA_data/depository_rel.csv',sep=',')

    # keep post-war treaties
    df_parties_1=df_parties_total[(df_parties_total['date_entry_into_force']<datetime(1947,12,31))|(df_parties_total['date_ratification']<datetime(1947,12,31))]

    old_treaties=set(df_parties_1['treaty_id'])
    for i in old_treaties:
        df_parties_total=df_parties_total[df_parties_total['treaty_id']!=i]


    
    if depository_excluded!=0:
            df_treaty_id=df_depository[df_depository['depository_id']==depository_excluded]    
            df_parties_1=pd.merge(df_parties_total,df_treaty_id,how='left')
            df_parties_2=df_parties_1[df_parties_1['depository_id']!=depository_excluded]
    else:
        df_parties_2=df_parties_total
        
    if treaty_excluded==None:
        df_parties_3=df_parties_2
    else: 
        df_parties_3=df_parties_2[df_parties_2['treaty_id']!=treaty_excluded]
    
    
    G_dic={}
    
    for year in tqdm(year_list):
        
        df_parties=wn.data_selection(df_parties_3,year,field_id,subject_id)
        B=wn.bipartite_network(df_parties)# the top nodes are the treaties
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

        #ratio_links_reduced=(G_weight.number_of_edges()-G_sig.number_of_edges())/G_weight.number_of_edges()

        #dic_num_treaties=dict(country_degrees)
        
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
            dic_local_clustering_coefficient=wn.local_clustering_coefficient(G_sig)
            #dic_eigenvector=nx.eigenvector_centrality(G_sig,max_iter=200,weight='weight')
        if weighted==False:
            dic_betweenness_centrality=nx.betweenness_centrality(G_sig)
            dic_closeness_centrality=nx.closeness_centrality(G_sig)
            dic_local_clustering_coefficient=nx.clustering(G_sig)
            #dic_eigenvector=nx.eigenvector_centrality(G_sig)

  
        nx.set_node_attributes(G_sig, year, name='year')
        #nx.set_node_attributes(G_sig, dic_num_treaties, name='number_of_treaties') # no point in bipcm
        nx.set_node_attributes(G_sig, dic_degree, name='degree')
        nx.set_node_attributes(G_sig, dic_strength, name='strength')
        #nx.set_node_attributes(G_sig, dic_eigenvector, name='eigenvector_centrality')
        nx.set_node_attributes(G_sig, dic_closeness_centrality, name='closeness_centrality')
        nx.set_node_attributes(G_sig, dic_betweenness_centrality, name='betweenness_centrality')
        nx.set_node_attributes(G_sig, dic_local_clustering_coefficient, name='local_clustering_coefficient')
        
        #nx.set_node_attributes(G_weight, dic_lattitude, name='lattitude')
        #nx.set_node_attributes(G_weight, dic_longitude, name='longitude')
        
        G_dic[year]=G_sig
        
    
    return G_dic

# This function is to extract the local measures of nodes obatained using function 'significant_network_bipcm'.

# Input: a dict of networks in different years,
# Output: a dataframe containing the local meausres of nodes in different years

# Parameters:
# 'layer': which layer to project on, and should be 'top'(treaty layer), 'bottom' (country layer)


def significant_local_measures_bipcm(G_dic,layer):
    year_list=list(G_dic.keys())
    
    node_attributes=['year','degree','strength','closeness_centrality','betweenness_centrality','local_clustering_coefficient']
    dic={}
    list_df=[]
    

    for year in tqdm(year_list):
        for i in node_attributes:
            dic[i]= nx.get_node_attributes(G_dic[year],i)

        df=pd.DataFrame(dic)
        if layer=='top':
            df_3=df.reset_index().rename(columns={'index':'treaty'})
        else:
            df_1=df.reset_index().rename(columns={'index':'country'})
            df_country_codes=pd.read_csv('IEA_data/countries_codes_a2_a3_final.csv')
            country_codes=['country','country_iso_a3','country_name']
            df_2=pd.merge(df_1,df_country_codes, how='left',left_on='country',right_on='country')
            df_3=df_2[country_codes+node_attributes]
            
       
        list_df.append(df_3)
        
    df_all=pd.concat(list_df)
    if layer=='top':
        df_all.sort_values(by=['treaty','year'],inplace=True)
    else:
        df_all.sort_values(by=['country','year'],inplace=True)
    
    df_all.reset_index(drop=True,inplace=True)
    
    return df_all

# This function is used to calculate global measure of networks, inlcuding 'number_of_nodes','number_of_links','density','number_of_components',
#'fraction_of_largest_component', 'average_degree','average_strength','average_weighted_shortest_path_length','diameter','weighted_global_clustering_coefficient'.

# Input: country-treay relationships in parties.csv
# Output: a dataframe containing global measures for networks in different years


# Parameters:
# 'year_list': a list of years
# 'layer': which layer to project on, and should be 'top'(treaty layer), 'bottom' (country layer)
# 'constraint':  'Ture' if 'layer' is set to be 'bottom', or 'False' if 'layer' is set to be 'top'.
# 'depository_excluded' is a a depository id; If it is not zero, then treaties belonging to this depository are excluded from the network. 
# 'treaty_excluded' is a treaty id, if treaty_excluded is not None, then the treaty will be excluded from the dataset
# 'weighted': True or False
# 'field_id' and 'subject_id' are for the function 'data_selection'.
#  The default value of 'field_id' is None, otherwised it can be '1' for regional treaties or '2' for global treaties.
#  The default values of 'subject_id' is an empty list '[]', or it can be a list of subject ids


def significant_global_measures_bipcm_depository(year_list, field_id,subject_id, layer, constraint, depository_excluded, subject_excluded, weighted=True):
    
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


    df_depository=pd.read_csv('IEA_data/depository_rel.csv',sep=',')
    if depository_excluded!=0:
            df_treaty_id=df_depository[df_depository['depository_id']==depository_excluded]    
            df_parties_1=pd.merge(df_parties_total,df_treaty_id,how='left')
            df_parties_2=df_parties_1[df_parties_1['depository_id']!=depository_excluded]
    else:
        df_parties_2=df_parties_total
    

    
    for year in tqdm(year_list):
        
        df_parties=wn.data_selection(df_parties_2,year,field_id,subject_id)
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

# This function is to calculate the local measures for both the bottom and top nodes for a bipartite network

# Input: country-treay relationships in parties.csv
# Output: two dataframes containing local measures for countries and treaties in the bipartite network, respectively.


# Parameters:
# 'year_list': a list of years
# 'depository_excluded' is a a depository id; If it is not zero, then treaties belonging to this depository are excluded from the network. 
# 'field_id' and 'subject_id' are for the function 'data_selection'.
#  The default value of 'field_id' is None, otherwised it can be '1' for regional treaties or '2' for global treaties.
#  The default values of 'subject_id' is an empty list '[]', or it can be a list of subject ids


def bipartite_local_measures(year_list, field_id, subject_id, depository_excluded):
    
    df_parties_total=pd.read_csv("IEA_data/parties.csv",sep=",")
    list_TypeofDates=['date_entry_into_force','date_ratification','date_simple_sigNMture','date_definite_sigNMture','date_withdrawal','date_consent_to_be_bound','date_accession_approv','date_acceptance_approv','date_provisioNMl_application','date_succession','date_reservation']
    for i in list_TypeofDates:
        df_parties_total[i]=pd.to_datetime(df_parties_total[i],format='%d/%m/%Y')
    df_parties_1=df_parties_total[(df_parties_total['date_entry_into_force']<datetime(1947,12,31))|(df_parties_total['date_ratification']<datetime(1947,12,31))]

    old_treaties=set(df_parties_1['treaty_id'])
    for i in old_treaties:
        df_parties_total=df_parties_total[df_parties_total['treaty_id']!=i]


    df_depository=pd.read_csv('IEA_data/depository_rel.csv',sep=',')
    
    if depository_excluded!=None:
        
        for i in depository_excluded:
            df_treaty_id=df_depository[df_depository['depository_id']==i]    
            df_parties1=pd.merge(df_parties_total,df_treaty_id,how='left',on='treaty_id')
            df_parties2=df_parties1[df_parties1['depository_id']!=i]
            df_parties_total=df_parties2.drop(columns=['depository_id'])

    B_dic={}
    
    for year in tqdm(year_list):
        
        df_parties=wn.data_selection(df_parties_total,year,field_id,subject_id)
        B=wn.bipartite_network(df_parties)

        nodes_treaties= {n for n,d in B.nodes(data=True) if d['bipartite']==0}
        nodes_parties= set(B) - nodes_treaties
        country_degrees,treaty_degrees=bi.degrees(B,nodes_treaties)

        dic_country_degree=dict(country_degrees)
        dic_treaty_degree=dict(treaty_degrees)
        dic_degree=dic_country_degree.copy()
        dic_degree.update(dic_treaty_degree)
        
        nx.set_node_attributes(B, year, name='year')
        nx.set_node_attributes(B, dic_degree, name='degree')
        
        B_dic[year]=B
        
    list_df=[]
    node_attributes=['year','bipartite','degree']
    dic={}
    for year in year_list:
        for i in node_attributes:
            dic[i]= nx.get_node_attributes(B_dic[year],i)
        df=pd.DataFrame(dic)
        df_0=df.reset_index().rename(columns={'index':'country/treaty'})
        list_df.append(df_0)
        
    df_all=pd.concat(list_df)
    df_all.sort_values(by=['country/treaty','year'],inplace=True)
    df_all.reset_index(drop=True,inplace=True)
    
    df_treaties=df_all[df_all['bipartite']==0].rename(columns={'country/treaty':'treaty'})
    
    df_country_codes=pd.read_csv('IEA_data/countries_codes_a2_a3_final.csv')
    df_1=df_all[df_all['bipartite']==1]
    df_2=df_1.merge(df_country_codes,how='left',left_on='country/treaty',right_on='country')
    df_countries=df_2[['country/treaty','country_iso_a3','country_name','year','bipartite','degree',]].rename(columns={'country/treaty':'country'})
    
    return df_countries, df_treaties



# This function is to calcualte the global measures for both the bottom and top nodes in a bipartite network

# Input: country-treay relationships in parties.csv
# Output: two dataframes containing global measures for countries and treaties in the bipartite network, respectively.

# Parameters:
# 'year_list': a list of years
# 'depository_excluded' is a a depository id; If it is not zero, then treaties belonging to this depository are excluded from the network. 
# 'field_id' and 'subject_id' are for the function 'data_selection'.
#  The default value of 'field_id' is None, otherwised it can be '1' for regional treaties or '2' for global treaties.
#  The default values of 'subject_id' is an empty list '[]', or it can be a list of subject ids


def bipartite_global_measures(year_list, field_id, subject_id, depository_excluded):
    
    df_local_measures_countries,df_local_measures_treaties=bipartite_local_measures(year_list,field_id, subject_id, depository_excluded)
   
    dic_country_global_all={}
    dic_treaty_global_all={}
    dic_country_global={}
    dic_treaty_global={}
    
    local_measures=['degree']
    global_measures=['average_degree']
    
    for i in local_measures:
        
        for year in year_list:
            dic_country_global[year]=df_local_measures_countries[df_local_measures_countries['year']==year].mean()[i]
            dic_treaty_global[year]=df_local_measures_treaties[df_local_measures_treaties['year']==year].mean()[i]
            
        dic_country_global_all[i]=dic_country_global
        dic_treaty_global_all[i]=dic_treaty_global
    
    df_global_measures_country=pd.DataFrame(dic_country_global_all).rename(columns=dict(zip(local_measures,global_measures)))
    df_global_measures_treaty =pd.DataFrame(dic_treaty_global_all).rename(columns=dict(zip(local_measures,global_measures)))
    
    df_countries=df_global_measures_country.reset_index().rename(columns={'index':'year'})
    df_treaties=df_global_measures_treaty.reset_index().rename(columns={'index':'year'})


    return df_countries, df_treaties

# This function is used to generate gephi files to plot graphs in the Gephi;

# Input: country-treay relationships in parties.csv
# Output: gephi files containing the edge lists and local measures of nodes

# Parameters:
# 'year_list': a list of years
# 'field_id' and 'subject_id' are for the function 'data_selection'.
#  The default value of 'field_id' is None, otherwised it can be '1' for regional treaties or '2' for global treaties.
#  The default values of 'subject_id' is an empty list '[]', or it can be a list of subject ids
# 'layer': which layer to project on, and should be 'top'(treaty layer), 'bottom' (country layer)
# 'constraint':  'Ture' if 'layer' is set to be 'bottom', or 'False' if 'layer' is set to be 'top'.
# 'file_output': The path of the output file


# Node attributes are calcuated and added to the file

def gephi_images_bipcm(year_list, field_id, subject_id, layer, constraint, file_output):
    
    df_parties_total=pd.read_csv("IEA_data/parties.csv",sep=",")
    df_depository=pd.read_csv('IEA_data/depository_rel.csv',sep=',')
    df_document=pd.read_csv('IEA_data/document.csv',sep=',')
    
    for year in tqdm(year_list):

        df_parties= wn.data_selection(df_parties_total, year, field_id, subject_id)
        B=wn.bipartite_network(df_parties)
        G_weight=wn.projection(B,layer)
        G_matrix=nx.to_numpy_matrix(G_weight,weight='weight').A

        
        num_nodes=G_weight.number_of_nodes()
        
            # obtain the p-values
        nodes_treaties= {n for n,d in B.nodes(data=True) if d['bipartite']==0}
        nodes_parties= set(B) - nodes_treaties

        Sparse_Matrix_B=bi.biadjacency_matrix(B,list(nodes_parties), list(nodes_treaties))
        Matrix_B=Sparse_Matrix_B.A # the rows are the countries and the columns are the treaties
        B_pcm = bipcm.BiPCM(Matrix_B, constraint)

        p_value_countries_bipcm=B_pcm.lambda_motifs_main(bip_set=constraint, write=False)

        # test if the p-value is significant by the False discovery rate
        p_value_list=[]

        for i in (range(0,num_nodes)):
            for j in range(0,num_nodes):
                if p_value_countries_bipcm[i][j]!=0:
                    p_value_list.append(p_value_countries_bipcm[i][j])

        p_value_list.sort(reverse=True) # decreasing gradually
        for k in range(0,len(p_value_list)):
            if p_value_list[k]<=k*0.01/len(p_value_list):
                significant=p_value_list[k]
                order=k # this is the number of links that should be removed
                break

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

        dic_strength=dict(nx.degree(G_sig,weight='weight'))
        dic_local_clustering_coefficient=wn.local_clustering_coefficient(G_sig)
        dic_betweenness_centrality=wn.betweenness_centrality_weighted(G_sig)
        dic_closeness_centrality=wn.closeness_centrality_weighted(G_sig)


        if layer=='bottom':
            df_geography=pd.read_csv("IEA_data/geograpical information of countries.csv",sep=',',encoding='latin-1')

            node_list_sig=list(G_sig.nodes())
            num_nodes_sig=G_sig.number_of_nodes()
            df_party_code=pd.DataFrame({'code':dict(zip(range(0,num_nodes_sig),list(node_list_sig)))})
            df=pd.merge(df_party_code, df_geography, how='left', left_on='code',right_on='Id')  
            

            dic_lattitude=dict(zip(list(df['code']),list(df['latitude'])))
            dic_longitude=dict(zip(list(df['code']),list(df['longitude'])))
            
            
            nx.set_node_attributes(G_sig, dic_lattitude, name='lattitude')
            nx.set_node_attributes(G_sig, dic_longitude, name='longitude')
            
        if layer=='top':
            df_1=pd.DataFrame()
            df_1['treaty_id']=list(nx.nodes(G_sig))
            df_2=pd.merge(df_1,df_depository,how='outer',on='treaty_id')
            df_2.fillna(0,inplace=True)
            dic_treaty_depository=dict(zip(df_2['treaty_id'],df_2['depository_id']))
            nx.set_node_attributes(G_sig, dic_treaty_depository, name='depository_id')
            

        nx.set_node_attributes(G_sig, dic_strength, name='strength')
        nx.set_node_attributes(G_sig, dic_local_clustering_coefficient, name='local_clustering_coefficient')
        nx.set_node_attributes(G_sig, dic_betweenness_centrality, name='weighted_betweenness_centrality')
        nx.set_node_attributes(G_sig, dic_closeness_centrality, name='weighted_closeness_centrality')
        
        dic_partitions=com.best_partition(G_sig, random_state=1)# default is 'weighted'
        nx.set_node_attributes(G_sig, dic_partitions, name='community')
        
        if layer=='top':
            treaty_titles=dict(zip(list(df_document['treaty_id']),list(df_document['titleOfText'])))
            G_final=nx.relabel_nodes(G_sig, treaty_titles)
            

        file_name=file_output+'_'+str(year)+'.gexf'
        
        nx.write_gexf(G_final,file_name)