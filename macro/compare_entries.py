#!/usr/bin/env python
import os
#import ROOT as ROOT
import yaml
from prettytable import PrettyTable

ERA = "2017"
cutflow_dir_base = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/fig/cutflow_%s/"
#"Dict_cutflow_SUSY_mh127_ctau500_HH.yaml"
yield_dir_base = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_SR_CWR/SUSY/BR_h%s_z%s/datacards/"
#"SUSY_mh127_ctau500.yaml"

def compare(era,sign,ch):
    C_DIR = cutflow_dir_base % (era)
    br = 0
    if ch=="HH":
        br=100
    if ch=="ZZ":
        br=0
    if ch=="HZ":
        br=50
    Y_DIR = yield_dir_base % (era,str(br),str(100-br))
    print "Reading yield dir: ", Y_DIR

    table = PrettyTable([
        'signal',
        'b2 cutflow',
        'b2 yield',
        'diff',
        'stat unc',
        'warning?',
    ])


    for s in sign:
        c_name = "Dict_cutflow_"+s+"_"+ch+".yaml"
        y_name = s+".yaml"
        if era=="2016":
            y_name = s+"_G-H.yaml"
            c_name = "Dict_cutflow_"+s+"_"+ch+"_G-H.yaml"

        if not os.path.isfile(C_DIR+c_name):
            print C_DIR+c_name, " not found! "
            exit()
        if not os.path.isfile(Y_DIR+y_name):
            print Y_DIR+y_name, " not found! "
            exit()


        with open(Y_DIR+y_name) as y:
            r_y = yaml.load(y, Loader=yaml.Loader)
            y.close()

        with open(C_DIR+c_name) as c:
            r_c = yaml.load(c, Loader=yaml.Loader)
            c.close()

        #print "Comparing bin2 entries: "
        #print C_DIR+c_name
        #print r_c['b2']

        #print "\n"
        #print Y_DIR+y_name
        #print r_y['y_entries']

        row = [s,r_c['b2'],r_y['y_entries'],round( (100*(r_y['y_entries']-r_c['b2']) / ( 0.5 * (r_y['y_entries']+r_c['b2'])  ) ) , 2), round(100*r_c['b2_stat_unc']/r_c['b2'],2), "!" if ( abs(100*(r_y['y_entries']-r_c['b2']) / ( 0.5 * (r_y['y_entries']+r_c['b2'])  ) ) > abs(100*r_c['b2_stat_unc']/r_c['b2'])  )  else "" ]
        table.add_row(row)
    print table
    
    #root_files = [x for x in os.listdir(DATADIR) if os.path.isfile(os.path.join(DATADIR, x))]
    #for r in root_files:
    #    if "_HH.root" not in r and "_HZ.root" not in r and "_ZZ.root" not in r:
    #        os.system("echo mv "+DATADIR+r+" "+DATADIR+r.replace(".root","_HH.root"))
    #        os.system("mv "+DATADIR+r+" "+DATADIR+r.replace(".root","_HH.root"))
    print "\n"

eras = ["2016"]
sign = [
    "SUSY_mh127_ctau500","SUSY_mh127_ctau3000",
    "SUSY_mh150_ctau500","SUSY_mh150_ctau3000",
    "SUSY_mh175_ctau500","SUSY_mh175_ctau3000",
    "SUSY_mh200_ctau500","SUSY_mh200_ctau3000",
    "SUSY_mh250_ctau500","SUSY_mh250_ctau3000",
    "SUSY_mh300_ctau500","SUSY_mh300_ctau3000",
    "SUSY_mh400_ctau500","SUSY_mh400_ctau3000",
    "SUSY_mh600_ctau500","SUSY_mh600_ctau3000",
    "SUSY_mh800_ctau500","SUSY_mh800_ctau3000",
    "SUSY_mh1000_ctau500","SUSY_mh1000_ctau3000",
    "SUSY_mh1250_ctau500","SUSY_mh1250_ctau3000",
    "SUSY_mh1500_ctau500","SUSY_mh1500_ctau3000",
    "SUSY_mh1800_ctau500","SUSY_mh1800_ctau3000",
]
for e in eras:
    print " - - ", e , " - - "
    compare(e,sign,"ZZ")
    print "\n"
