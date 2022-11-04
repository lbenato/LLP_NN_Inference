#!/usr/bin/env python
import os

var = [
"Jets.eta",
"Jets.phi",
"Jets.pt",
"Jets.sigprob",
"Jets.energyRecHitsEB",
"Jets.eFracRecHitsEB",
"Jets.nRecHitsEB",
"Jets.sig1EB",
"Jets.sig2EB",
"Jets.ptDEB",
"Jets.cHadEFrac",
"Jets.nHadEFrac",
"Jets.eleEFrac",
"Jets.photonEFrac",
"Jets.ptAllTracks",
"Jets.ptAllPVTracks",
"Jets.alphaMax",
"Jets.betaMax",
"Jets.gammaMax",
"Jets.gammaMaxEM",
"Jets.gammaMaxHadronic",
"Jets.gammaMaxET",
"Jets.minDeltaRPVTracks",
"Jets.minDeltaRAllTracks",
"Jets.nRecHitsEB",
"Jets.timeRecHitsEB",
"Jets.nConstituents",
"Jets.nTrackConstituents",
"Jets.timeRMSRecHitsEB",
"Jets.radiusLLPCaloCorr"
]


cuts=[
    "SR_match_compare",
    #"SR_match_low_score",
]

cuts_b=[
    "SR_match_compare_low_pt",
    #"SR_match_low_score",
]

regs = ["calo"]

for a in var:
    for n, b in enumerate(cuts):
        os.system('echo python macro/plot_from_tree_skim_v6.py -c ' + str(cuts_b[n]) + ' -s '+ str(b) + ' -v ' + str(a) + ' -B -b \n')
        os.system('python macro/plot_from_tree_skim_v6.py -c ' + str(cuts_b[n]) + ' -s '+ str(b) + ' -v ' + str(a) + ' -B -b \n')
        #os.system('echo python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch -v ' + str(a) + ' -b \n')
        #os.system('python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch -v ' + str(a) + ' -b \n')
        #os.system('echo python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch_less10hits -v ' + str(a) + ' -b \n')
        #os.system('python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch_less10hits -v ' + str(a) + ' -b \n')
