***** This file contains the codes for the regression in Tables D.1 and F.1 in the appendix *****

clear all
set more off

cd "/Users/jianjiangao/Documents/Research project_co-treaty network/local-repository-network-analysis-of-climate-treaties/significant_country_network/RobustTest/Regression"

capture log close
log using regression.log, replace

*** Past treaty membership and speed of ratification of future IEAs

import delimited "regression_citations_media.csv", encoding(ISO-8859-1) clear


encode treaty_id, gen(treaty_id_code)
encode party_code, gen(party_code_code)

gen spanlog=ln(span)

reg spanlog mean_reports  i.treaty_id_code i.party_code_code i.year_rati, vce(cluster party_code_code)

estimates store ols1

reg spanlog  mean_citations  i.treaty_id_code i.party_code_code i.year_rati, vce(cluster party_code_code)
estimates store ols2


esttab ols1 ols2   using citation_media.tex, replace se ar2 nogaps title(Regression) mtitle(span logspan) keep(mean_reports mean_citations)


*** Centrality ranking and speed of ratification of future IEAs

import delimited "regression_centrality_cut.csv", encoding(ISO-8859-1) clear


encode treaty_id, gen(treaty_id_code)
encode party_code, gen(party_code_code)


gen spanlog=ln(span)

reg spanlog rank_str  i.treaty_id_code i.party_code_code i.year_rati, vce(cluster party_code_code)

estimates store ols1

reg spanlog rank_bet  i.treaty_id_code i.party_code_code i.year_rati, vce(cluster party_code_code)

estimates store ols2

reg spanlog rank_clo  i.treaty_id_code i.party_code_code i.year_rati, vce(cluster party_code_code)

estimates store ols3

esttab ols1 ols2 ols3 using centrality.tex, replace se ar2 nogaps title(Regression ranking) mtitle(strength betweenness_centrality closeness_centrality) keep(rank_str rank_bet rank_clo) star(* 0.10 ** 0.05 *** 0.01)



clear

log close
