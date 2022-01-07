#! /usr/bin/env python

import yaml
from prettytable import PrettyTable

ERA = "2018"
REGION = "WtoMN"
EXTR_REGION = "WtoMN"
pred_file_name = "plots/Efficiency/v5_calo_AOD_"+ERA+"_"+REGION+"/BkgPredResults_"+ERA+"_"+REGION+"_vs_eta_"

slices = ["A","B","C","D"]
results = {}
for s in slices:
    print pred_file_name+s+".yaml"
    with open(pred_file_name+s+".yaml","r") as f:
        print "Info: opening dictionary in file "+pred_file_name+s+".yaml"
        print "Extrapolation region: ", REGION
        results[s] = yaml.load(f, Loader=yaml.Loader)
        f.close()

#print results
data_set = results[slices[0]][EXTR_REGION].keys()[0]
key_set = results[slices[0]][EXTR_REGION][data_set].keys()
dicty = {}

for k in key_set:
    dicty[k] = 0.
    for s in slices:
        #print s
        #print results[s][EXTR_REGION][data_set]
        #print k, results[s][EXTR_REGION][data_set][k]
        dicty[k]+=results[s][EXTR_REGION][data_set][k]

print dicty

print "=== Prediction of "+REGION+ " extrapolating from "+EXTR_REGION+" ==="
table1 =  PrettyTable(['Dataset', 'Bin 1 Yield', 'Bin 1 Pred', 'Discr. %'])
row1 = [data_set+" "+ERA,round(dicty["y_1"],0),round(dicty["pred_1"],2), round(100*(dicty["y_1"]-dicty["pred_1"])/dicty["y_1"],2)]
table1.add_row(row1)
print table1

table2 = PrettyTable(['Bin 2 Yield', 'Bin 2 Pred from 0', 'Pred from 0 Discr. %', 'Bin 2 Pred from 1', 'Pred from 1 Discr. %', 'Syst. Unc. %'])
row2 = [round(dicty["y_2"],0) , round(dicty["pred_2"],4), round(100*(dicty["y_2"]-dicty["pred_2"])/dicty["y_2"],2), round(dicty["pred_2_from_1"],4),  round(100*(dicty["y_2"]-dicty["pred_2_from_1"])/dicty["y_2"],2), round(100*abs(dicty["pred_2"]-dicty["pred_2_from_1"])/((dicty["pred_2"]+dicty["pred_2_from_1"])/2),2) ]
table2.add_row(row2)
print table2
