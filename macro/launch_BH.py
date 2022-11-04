#!/usr/bin/env python
import os

var = [
#"min_dPhi_jets_0p7",
#"min_dPhi_jets_0p9",
#"min_dPhi_jets_0p996",
#"min_dR_jets",
#"nCSCSegments",
##"DT_fit_res",
#"dt_ecal_dist",
#"Jets.eta",
#"Jets.phi",
#"Jets.nRecHitsEB",
#"Jets.FracCal",
#" '(-Jets.ecalE + Jets.hcalE)/(Jets.hcalE + Jets.ecalE)' ",
#"JetsCaloAdd.eFracEB2CSC",
#"JetsCaloAdd.eFracEB2CSC0p1",
#"JetsCaloAdd.eFracEB2CSC0p04",

#"JetsCaloAdd.meanPhiEB",
##"JetsCaloAdd.meanEtaEB",
##"JetsCaloAdd.meanATLASEB",
#"JetsCaloAdd.meanXEB",
#"JetsCaloAdd.meanYEB",
#"JetsCaloAdd.meanZEB",
#"JetsCaloAdd.meanREB",

##"JetsCaloAdd.spreadPhiEB",
##"JetsCaloAdd.spreadEtaEB",
##"JetsCaloAdd.spreadATLASEB",
#"JetsCaloAdd.spreadXEB",
#"JetsCaloAdd.spreadYEB",
#"JetsCaloAdd.spreadZEB",
#"JetsCaloAdd.spreadREB",
#"Jets.sigprob",
#"nCosmicMuons",
#"nCosmicMuonsOneLeg",
#"JetsCaloAdd.eFracEB2CSC0p1"


#"MEt.pt",
#"MEt.phi",

#"JetsNegative.eFracRecHitsEB",
##"JetsNegative.energyRecHitsEB",
#"JetsNegative.nRecHitsEB",
#"JetsNegative.timeRMSRecHitsEB",
#"JetsNegative.timeRecHitsEB",
#"JetsNegative.pt",
##"JetsNegative.eta",
"JetsNegative.phi",
#"JetsNegative.sigprob",

#"Jets.timeRMSRecHitsEB",
#"Jets.timeRecHitsEB",
#"Jets.nRecHitsEB",
#"Jets.eFracRecHitsEB",
#"Jets.eta",
#"Jets.phi",
#"Jets.pt",
#"Jets.sigprob",
]


cuts=[
    #"SR_veto_bin2_cosmics",
    #"SR_veto_bin2_cosmics_nj",
    #"SR_veto_bin2_cosmics_tag_0p9",
    #"SR_veto_bin2_cosmics_tag_0p9_largeRMS",

    "SR_veto_bin2_cosmics_nj_time",

    #"SR_veto_bin2_cosmics_tag",
    #"SR_veto_bin2_cosmics_tag",
    #"SR_veto_bin2_cosmics_tag_0p9",
    #"SR",
    #"SR_2cosmic",
    #"SR_1cosmiconeleg",
    #"SR_dnn_0p7",
    #"SR_dnn_0p7_closePhi",
    #"SR_dnn_0p7_closePhi_less10hits",
    #"SR_dnn_0p7_closePhi_less10hits_also_sign",
    #"SR_closePhi",
    #"SR_closePhi_less10hits",
    #"SR_less10hits",
    #"SR_veto_bin2",
]

regs = ["calo"]

for a in var:
    for b in cuts:
        os.system('echo python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s '+ str(b) + '_match -v ' + str(a) + ' -b \n')
        os.system('python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s '+ str(b) + '_match -v ' + str(a) + ' -b \n')
        #os.system('echo python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch -v ' + str(a) + ' -b \n')
        #os.system('python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch -v ' + str(a) + ' -b \n')
        #os.system('echo python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch_less10hits -v ' + str(a) + ' -b \n')
        #os.system('python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch_less10hits -v ' + str(a) + ' -b \n')
