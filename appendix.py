import pandas as pd
from datetime import datetime
import numpy  as np
import networkx as nx
from networkx.algorithms import bipartite as bi

import copy

# This function selects data points until a specific year based on the date_entry into force and date_ratification

# Input: the original data in the format of dataframe 
# Output: a dataframe of selected data points

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


def data_selection(df_parties_total,year,field_id,subject_ids):# This function select data points until a specific year based on the date_entry into force and date_ratification
    
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
    df_parties_1=df_one_subject[((df_one_subject['date_ratification']<datetime(year,12,31))&(df_one_subject['date_withdrawal']>datetime(year,12,31)))|((df_one_subject['date_ratification']<datetime(year,12,31))&(df_one_subject['date_withdrawal'].isnull()))]
    df_parties_2=df_one_subject[(df_one_subject['date_ratification'].isnull()&(df_one_subject['date_entry_into_force']<datetime(year,12,31))&(df_one_subject['date_withdrawal']>datetime(year,12,31)))|(df_one_subject['date_ratification'].isnull()&(df_one_subject['date_entry_into_force']<datetime(year,12,31))&df_one_subject['date_withdrawal'].isnull())]    
    df_parties= df_parties_1.append(df_parties_2)
    
    return df_parties

# This funciton is used to generate a bipartite network after the data selection. 

# Input: a dataframe containing the edgelist of a bipartite network
# Output: a bipartite netwrok defined in networkx

# Parameters:
# 'df_parties': a dataframe containing the edgelist of a bipartite network

def bipartite_network(df_parties):
    
    B = nx.Graph()
    B.add_nodes_from(df_parties['treaty_id'], bipartite=0)
    B.add_nodes_from(df_parties['party_code'], bipartite=1)
    B.add_edges_from([(row['treaty_id'], row['party_code']) for idx, row in df_parties.iterrows()])
    
    return B

# This funciton is used to project a bipartite network to a one-mode network. 

# Input is a bipartite network  and  the output is the one-mode network. 
# Project on the nodes with attribute as '1'

# Parameters:
# 'B': a bipartit network defined in networkx
# 'nodes_type': 'top' for treaties or 'bottom ' for countries

def projection(B,nodes_type):
    
    nodes_top= {n for n,d in B.nodes(data=True) if d['bipartite']==0}
    nodes_bottom= set(B) - nodes_top
    
    if nodes_type=='top':
        G=bi.collaboration_weighted_projected_graph(B, list(nodes_top))# treaties
    if nodes_type=='bottom':
        G=bi.collaboration_weighted_projected_graph(B, list(nodes_bottom))# countries
    
    return G  
    

# This function is to obatin a graph with the distance as the weight to use the functions in networkX

# Input: a dataframe containing the edge list and link weights 'w'
# Output: a network with the distance '1/w' as link weights

# Parameters:
# G: a one-mode network

def network_distance(G):
    df=nx.to_pandas_edgelist(G)
    df['distance']=df['weight'].map(lambda x: 1/x) 
    df_distance=df[['source','target','distance']]
    G_distance=nx.from_pandas_edgelist(df_distance,edge_attr=True)
    return G_distance

# This function is  to calculate the shortest path length according to 1/w(this can be changed)

# Input: a dataframe containing the edge list and link weights 'w'
# Output: a dataframe containing the distance between any pairs of nodes

# Parameters:
# 'G': a one-mode network

def shortest_path_lenghth_weighted(G):
    df=nx.to_pandas_edgelist(G)
    df['distance']=df['weight'].map(lambda x: 1/x) 
    df_distance=df[['source','target','distance']]
    G_distance=nx.from_pandas_edgelist(df_distance,edge_attr=True)
    dict_ShortestPathLength=dict(nx.shortest_path_length(G_distance,weight='distance'))
    df_shortest_path_length=pd.DataFrame(dict_ShortestPathLength)

    return df_shortest_path_length


# This function is to calculate the frequency of the shortest path length 

# Input: a one-mode network with link weights 'w'
# Output: the frequency of distance between any pairs of nodes

# Parameters:
# 'G': a one-mode network
# 'minimum': the minimum value 
# 'maximum': the maximum value 
# 'num_intervels': the number of intervals

def frequency_shortest_path_length(G,minimum,maximum,num_intervels):
    df=nx.to_pandas_edgelist(G)
    df['distance']=df['weight'].map(lambda x: 1/x) 
    df_distance=df[['source','target','distance']]
    G_distance=nx.from_pandas_edgelist(df_distance,edge_attr=True)
    dict_ShortestPathLength=dict(nx.shortest_path_length(G_distance,weight='distance'))
    df_shortest_path_length=pd.DataFrame(dict_ShortestPathLength)
    
    column_names=df_shortest_path_length.columns.values.tolist()
    list_1=[]
    for m in column_names:
        list_1.extend(df_shortest_path_length[m])
        
    list_2=[x for x in list_1 if x!=0]
    
    dic_frequency={}
    for i in range(0,num_intervels):
        j=(maximum-minimum)/num_intervels*i+minimum
        k=(maximum-minimum)/num_intervels*(i+1)+minimum
        frequency=len([v for v in list_2 if j<=v< k])/len(list_2)
        dic_frequency.update({(j+k)/2:frequency})
        
    return dic_frequency


# This function is to obtain the weighted average shortest path length of networks using the 1/w as distance between ndoes

# Input: a one-mode netwokr
# Output: the weighted average shortest path length and the diameter of the network 

# Parameters:
# 'G': a one-mode network


def average_shortest_path_length(G):
    df=nx.to_pandas_edgelist(G)
    df['distance']=df['weight'].map(lambda x: 1/x) 
    df_distance=df[['source','target','distance']]
    G_distance=nx.from_pandas_edgelist(df_distance,edge_attr=True)
    largest_cc = max(nx.connected_components(G_distance), key=len)

    length=[]
    for i in largest_cc:
        for j in largest_cc:
            dd=nx.shortest_path_length(G_distance, source=i, target=j, weight='distance')
            length.append(dd)
    #average_shortest_path_length_weight=average(length)
    diameter=max(length)
    average_shortest_path_length_weight=nx.average_shortest_path_length(G_distance.subgraph(largest_cc).copy(), weight='distance')


    return average_shortest_path_length_weight, diameter


# This function is to obtain the unweighted average shortest path length of networks

# Input: a one-mode netwokr
# Output: the average shortest path length and the diameter of the network 

# Parameters:
# 'G': a one-mode network

def average_shortest_path_length_unweighted(G):
    
    largest_cc = max(nx.connected_components(G), key=len)

    average_shortest_path_length_unweight=nx.average_shortest_path_length(G.subgraph(largest_cc).copy())
    diameter=nx.diameter(G.subgraph(largest_cc).copy())

    return average_shortest_path_length_unweight, diameter


# This function is to compute the fraction of the largest component 
# Input: a one-mode network
# Output: the fraction of the largest component 

# Parameters:
# 'G': a one-mode network

def fraction_largest_component(G):

    largest_cc = max(nx.connected_components(G), key=len)
    number_nodes= nx.number_of_nodes(G)
    fraction=len(largest_cc)/number_nodes
    return fraction



# This function is to calculate the closeness centrality with 1/w (this can be changed) as the distance between nodes

# Input: a one-mode network
# Output: a dict of weighted closeness centrality between any pairs of nodes

# Parameters:
# 'G': a one-mode network


def closeness_centrality_weighted(G):
    df=nx.to_pandas_edgelist(G)
    df['distance']=df['weight'].map(lambda x: 1/x) 
    df_distance=df[['source','target','distance']]
    G_distance=nx.from_pandas_edgelist(df_distance,edge_attr=True)
    closeness_centrality_distance=nx.closeness_centrality(G_distance, distance = 'distance')
    return closeness_centrality_distance


#This function is designed to calculate the betweenness centrality with 1/w (this can be changed) as the distance

# Input: a one-mode network
# Output: a dict of weighted betweenness centrality between any pairs of nodes

# Parameters:
# 'G': a one-mode network

def betweenness_centrality_weighted(G):
    df=nx.to_pandas_edgelist(G)
    df['distance']=df['weight'].map(lambda x: 1/x) 
    df_distance=df[['source','target','distance']]
    G_distance=nx.from_pandas_edgelist(df_distance,edge_attr=True)
    betweenness_centrality_distance=nx.betweenness_centrality(G_distance, weight = 'distance')   
    return betweenness_centrality_distance


# Get the tops ones in a dict
# The input is a dict and the number of items you want to select
def top_ones(dic,top_number):
    sorted_items=sorted(dic.items(), key=lambda dic: dic[1],reverse=True)
    return dict(map(lambda x:x, sorted_items[:top_number]))

# This function is to get the average of a list of vlaues

# Input: a list of vlaues
# Output: the mean value of the list of values

def average(seq, total=0.0): 
    num = 0
    for item in seq: 
        total += item 
        num += 1
    return total / num

# This funciton is used to computed the local clustering coefficient using the method probosed by Barrat. 
# Input: a graph
# Output: a dict with local clustering coefficient for all nodes

# Parameters:
# 'G': a one-mode network

def local_clustering_coefficient(G):
    local_clustering = {}
    for node in G:
        value_closed_triplets=0
        degree=0
        strength=0
        neighboursOfNode=list(G.neighbors(node))
        neighboursOfNode.sort()
        degree=len(neighboursOfNode)
                              
        for i in range(len(neighboursOfNode)):
            neighbour_1=neighboursOfNode[i]
            strength+=G[node][neighbour_1]['weight']
            for j in range(i+1,len(neighboursOfNode)):
                neighbour_2=neighboursOfNode[j]
                if neighbour_1 in G.neighbors(neighbour_2):
                    value_closed_triplets += (G[node][neighbour_1]['weight']+G[node][neighbour_2]['weight'])
                    
        if (degree-1)!=0:
            if strength!=0:
                local_clustering[node]=value_closed_triplets/strength/(degree-1)
        else:
            local_clustering[node]=0
    return  local_clustering
                   

# This funciton is used to computed the global clustering coefficient using the method probosed by Tore Opshal. 
# Input: a graph
# Output: the global clustering coefficient   

# Parameters:
# 'G': a one-mode network

def global_clustering_coefficient(G):
    value_closed_triplets=0
    value_triplets=0
    for node in G:
        neighboursOfNode=list(G.neighbors(node))
        neighboursOfNode.sort()
        for i in range(len(neighboursOfNode)):
                 for j in range(i+1,len(neighboursOfNode)):
                        neighbour_1=neighboursOfNode[i]
                        neighbour_2=neighboursOfNode[j]
                        value_triplets += (G[node][neighbour_1]['weight']+G[node][neighbour_2]['weight'])/2
                        if neighbour_1 in G.neighbors(neighbour_2):
                            value_closed_triplets += (G[node][neighbour_1]['weight']+G[node][neighbour_2]['weight'])/2

    if value_triplets==0:
        global_clustering=0
    else:
        global_clustering=value_closed_triplets/value_triplets
        
    return global_clustering


# this function is the same as above; it is uesed to test the middle steps of the calculation

def global_clustering_coefficient_test(G):
    value_closed_triplets=0
    value_triplets=0
    for node in G:
        neighboursOfNode=list(G.neighbors(node))
        neighboursOfNode.sort()
        for i in range(len(neighboursOfNode)):
                 for j in range(i+1,len(neighboursOfNode)):
                        neighbour_1=neighboursOfNode[i]
                        neighbour_2=neighboursOfNode[j]
                        value_triplets += (G[node][neighbour_1]['weight']+G[node][neighbour_2]['weight'])/2
                        if neighbour_1 in G.neighbors(neighbour_2):
                            value_closed_triplets += (G[node][neighbour_1]['weight']+G[node][neighbour_2]['weight'])/2

    if value_triplets==0:
        global_clustering=0
    else:
        global_clustering=value_closed_triplets/value_triplets
        
    return value_closed_triplets, value_triplets, global_clustering


# This function is to get a list of countries' names according to countries's codes

# Input: a list of countries' codes
# Output: a list of countries' names and a text file to copy

# Parameters:
# 'code_list': a list of country codes

def country_list(code_list):
    
    list_name=[]
    df=pd.read_csv('IEA_data/parties_cat.csv', sep=',')
    df_code=pd.read_csv('IEA_data/countries_codes_a2_a3_final.csv', sep=',')

    code_list_new=[]
    if len(code_list[0])==3:
        for i in code_list:
            one=list(df_code[df_code['country_iso_a3']==i]['country']) 
            code_list_new.append(one[0])
    else:
        code_list_new=code_list
        

    for i in code_list_new:
        one=list(df[df['code']==i]['name']) 
        list_name.append(one[0])
    
    file = open('country_names','w')

    for name in list_name:
        file.write(name + ', ')

    file.close()
        
    return list_name
        
# This function is to get a list of treaties' names according to treaties' ids

# Input: a list of treateis' ids
# Output: a list of treaties' names and a text file to copy

# Parameters:
# 'code_list': a list of treaty ids

def treaty_title(code_list):

    list_name=[]
    df=pd.read_csv('IEA_data/document.csv', sep=',')
    for i in code_list:
        one=list(df[df['treaty_id']==i]['titleOfText']) 
        list_name.append(one[0])

    file = open('treaty_titles','w')

    #for name in list_name:
     #   file.write(name + ', ')

    #file.close()

    return list_name