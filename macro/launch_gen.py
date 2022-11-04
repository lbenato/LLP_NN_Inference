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


#2016 missing
#"JetsNegative.eFracRecHitsEB",
#"JetsNegative.energyRecHitsEB",
#"JetsNegative.nRecHitsEB",
#"JetsNegative.timeRMSRecHitsEB",
#"JetsNegative.timeRecHitsEB",
#"JetsNegative.pt",
#"JetsNegative.eta",
#"JetsNegative.phi",

#"Jets.timeRMSRecHitsEB",
#"Jets.timeRecHitsEB",
#"nCosmicMuons",
#"nCosmicMuonsOneLeg",
#"JetsCaloAdd.eFracEB2CSC0p1"

#"MEt.pt",
#"MEt.phi",
"Jets[0].eta",
"Jets[0].phi",
"Jets[0].pt",
"Jets[0].sigprob",
"Jets[0].energyRecHitsEB",
"Jets[0].eFracRecHitsEB",
"Jets[0].nRecHitsEB",
"Jets[0].sig1EB",
"Jets[0].sig2EB",
"Jets[0].ptDEB",
"Jets[0].cHadEFrac",
"Jets[0].nHadEFrac",
"Jets[0].eleEFrac",
"Jets[0].photonEFrac",
"Jets[0].ptAllTracks",
"Jets[0].ptAllPVTracks",
"Jets[0].alphaMax",
"Jets[0].betaMax",
"Jets[0].gammaMax",
"Jets[0].gammaMaxEM",
"Jets[0].gammaMaxHadronic",
"Jets[0].gammaMaxET",
"Jets[0].minDeltaRPVTracks",
"Jets[0].minDeltaRAllTracks",
#"MinJetMetDPhi",
#"Jets.dPhi_met",
"Jets[0].nRecHitsEB",
"Jets[0].timeRecHitsEB",
"Jets[0].nConstituents",
"Jets[0].nTrackConstituents",
"Jets[0].timeRMSRecHitsEB",
#"nTagJets_0p996_JJ",
#"nCHSJetsAcceptanceCalo",
#"GenHiggs[0].pt",
#"GenHiggs[1].pt",
#"GenLLPs[0].pt",
#"GenLLPs[1].pt",


#"GenGravitinos[0].pt",
#"GenGravitinos[1].pt",
#"DiGravitino_eta",
#"DiGravitino_pt",
#"DiGravitino_phi",
#"dPhi_DiGravitino_GenMet",
#"dPhi_DiGravitino_Met",
#"dPhi_Gravitino_GenMet_0",
#"dPhi_Gravitino_GenMet_1",
#"dPhi_Gravitino_Met_0",
#"dPhi_Gravitino_Met_1",
#"dPhi_Higgs_Met_0",
#"dPhi_Higgs_Met_1",
#"dPhi_Higgs_GenMet_0",
#"dPhi_Higgs_GenMet_1",
#"dPhi_LLP_GenMet_0",
#"dPhi_LLP_GenMet_1",
#"dPhi_LLP_Met_0",
#"dPhi_LLP_Met_1",
#"perc_met_held_by_gravitinos",

#"GenHiggs[0].dRdaughters",
#"GenHiggs[1].dRdaughters",
]


cuts=[
    #"ctau500",
    #"METctau500",
    "ctau500match",
    "METctau500match",
    #"ctau3000",
    #"METctau3000",
]

regs = ["calo"]

for a in var:
    for b in cuts:
        os.system('echo python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s '+ str(b) + ' -v ' + str(a) + ' -B -b \n')
        os.system('python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s '+ str(b) + ' -v ' + str(a) + ' -B -b \n')
        #os.system('echo python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch -v ' + str(a) + ' -b \n')
        #os.system('python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch -v ' + str(a) + ' -b \n')
        #os.system('echo python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch_less10hits -v ' + str(a) + ' -b \n')
        #os.system('python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch_less10hits -v ' + str(a) + ' -b \n')
