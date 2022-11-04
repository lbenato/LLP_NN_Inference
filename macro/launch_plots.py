#!/usr/bin/env python
import os

var = [
#"number_of_b_matched_to_CHSJets",
#"Z_mass",
#"MEt.pt",
#"MEt.phi",
#"HT",
#"Jets.pt",
###############33
#"JetsNegative[0].timeRecHitsEB",
#"JetsNegative[0].timeRMSRecHitsEB",

#"JetsNegative[0].CSV",
#"JetsNegative[0].pt",

#"Jets[0].CSV",
#"Jets[0].eta",
#"Jets[0].pt",
#"Jets[0].phi",    
#"Jets[0].nConstituents",
#"Jets[0].nTrackConstituents",
#"Jets[0].nSelectedTracks",
#"Jets[0].timeRecHitsEB",
#"Jets[0].timeRMSRecHitsEB",
#"Jets[0].energyRecHitsEB",
#"Jets[0].eFracRecHitsEB",
#"Jets[0].nRecHitsEB",
#"Jets[0].sig1EB",
#"Jets[0].sig2EB",
#"Jets[0].ptDEB",
#"Jets[0].cHadEFrac",
#"Jets[0].nHadEFrac",
#"Jets[0].eleEFrac",
#"Jets[0].photonEFrac",
#"Jets[0].ptAllTracks",
#"Jets[0].ptAllPVTracks",
#"Jets[0].alphaMax",
#"Jets[0].betaMax",
#"Jets[0].gammaMax",
#"Jets[0].gammaMaxEM",
#"Jets[0].gammaMaxHadronic",
#"Jets[0].gammaMaxET",
#"Jets[0].minDeltaRPVTracks",
#"Jets[0].minDeltaRAllTracks",
#"Jets[0].sigprob",

#"nTagJets_0p996_JJ",
"MEt.pt",
"HT",
#"MT",
#"nPV",
#"nTagJets_0p996_JJ",
#"nElectronsPassing",
#"nCHSJetsAcceptanceCalo",
#"Electrons.pt",
#"Electrons.eta",
"JetsNegative.pt",
"JetsNegative.eta",
"JetsNegative.phi",    
"JetsNegative.nConstituents",
"JetsNegative.nTrackConstituents",
"JetsNegative.nSelectedTracks",
"JetsNegative.timeRecHitsEB",
"JetsNegative.timeRMSRecHitsEB",
"JetsNegative.energyRecHitsEB",
"JetsNegative.eFracRecHitsEB",
"JetsNegative.nRecHitsEB",
"JetsNegative.sig1EB",
"JetsNegative.sig2EB",
"JetsNegative.ptDEB",
"JetsNegative.cHadEFrac",
"JetsNegative.nHadEFrac",
"JetsNegative.eleEFrac",
"JetsNegative.photonEFrac",
"JetsNegative.ptAllTracks",
"JetsNegative.ptAllPVTracks",
"JetsNegative.alphaMax",
"JetsNegative.betaMax",
"JetsNegative.gammaMax",
"JetsNegative.gammaMaxEM",
"JetsNegative.gammaMaxHadronic",
"JetsNegative.gammaMaxET",
"JetsNegative.minDeltaRPVTracks",
"JetsNegative.minDeltaRAllTracks",
"JetsNegative.sigprob",
#"Jets.FracCal",
"JetsNegative.ecalE",
"JetsNegative.hcalE",

#"dR_Higgs",
#"dR_LLPs",
#"dR_Gravitinos",
#"dR_Gravitino_Higgs_0",
#"dR_Gravitino_Higgs_1",
#"dR_Gravitino_GenMet_0",
#"dR_Gravitino_GenMet_1",
#"dPhi_Gravitino_Met_0",
#"dPhi_Gravitino_Met_1",
#"dPhi_Gravitino_GenMet_0",
#"dPhi_Gravitino_GenMet_1",
#"dR_LLP_GenMet_0",
#"dR_LLP_GenMet_1",
#"dPhi_LLP_Met_0",
#"dPhi_LLP_Met_1",
#"dPhi_LLP_GenMet_0",
#"dPhi_LLP_GenMet_1",
#"dR_Higgs_GenMet_0",
#"dR_Higgs_GenMet_1",
#"dPhi_Higgs_Met_0",
#"dPhi_Higgs_Met_1",
#"dPhi_Higgs_GenMet_0",
#"dPhi_Higgs_GenMet_1",
#"DiGravitino_pt",
#"DiGravitino_mass",
#"DiGravitino_eta",
#"DiGravitino_phi",
#"dR_DiGravitino_GenMet",
#"dPhi_DiGravitino_Met",
#"dPhi_DiGravitino_GenMet",
#"inv_mass_high_DNN",
#"inv_mass_tag_DNN",
#"inv_mass_reco_higgs_0",
#"inv_mass_reco_higgs_1",
#"perc_met_held_by_gravitinos",
]


var = [
    "dt_ecal_no_tag_dist",
]
#var = [
#"Jets.sigprob",
#"MEt.pt",
#"MEt.ptGen",
#"",
#"",
#"GenGravitinos[0].pt",
#"GenGravitinos[1].pt",
#"GenLLPs[0].pt",
#"GenLLPs[1].pt",
#"GenHiggs[0].pt",
#"GenHiggs[1].pt",
#"GenBquarks[0].pt",
#"GenBquarks[2].pt",
#"MinJetMetDPhi",
#"perc_met_held_by_gravitinos",
#"inv_mass_high_DNN",
#"inv_mass_tag_DNN",
#"inv_mass_reco_higgs_0",
#"inv_mass_reco_higgs_1",
#]
#cuts = [
#"none",]


cuts=[
#"none",
#"SRsel",
#"LLP0_calo",
#"LLP1_calo",
#"LLP0_prompt",
#"LLP1_prompt",
#"LLP0_calo_MET",
#"LLP1_calo_MET",
#"2j_h0",
#"2j_h1",
#"2j_h0_prompt",
#"2j_h1_prompt",
#"2j_h0_calo",
#"2j_h1_calo",
#"SR_veto_bin2",
#"ZtoMM",
#"ZtoMMPho",
#"SR_accumulation",
#"SR_no_accumulation",
#"SRPositiveWeights",
#"SRNegativeWeights",
#"WtoMN_noMT",
#"TtoEM",
#"TtoEM_CSV_j0",
#"SRmatch",
#"SRmatch_tag",

#"elePUCaltech"
"elePUCMSFinal",#METPt70VetoPho"
#"ele_met50",
#"ele_jets50",
#"ele_metjets50",
]

cuts = ["SR_boost"]
cuts = ["SR_negative"]
cuts = ["SR_positive"]

'''
var = [
"Jets.nTrackConstituents",
"Jets.timeRecHitsEB",
"Jets.nHadEFrac",
"Jets.cHadEFrac",
"Jets.photonEFrac",
"Jets.photonEFrac + Jets.nHadEFrac",
"Jets.pt",
"Jets.nRecHitsEB",
"Jets.eFracRecHitsEB",
"Jets.hcalE",
"Jets.ecalE",
"Jets.FracCal",
"Jets.sigprob",
"Jets.eta",
"Jets.phi",
#"(-Jets.ecalE + Jets.hcalE)/(Jets.hcalE + Jets.ecalE)",
]
'''
'''
cuts = [
#"WtoMN_tag",
#"WtoMN_no_tag",
#"WtoMN_low_nRecHitsEB",
#"SR_low_nRecHitsEB",
#"",#HBHE
#"tag",
#"match",
"SR_fail_HBHE",
"SR_fail_HBHE_match",
]
'''
regs = ["calo"]

for a in var:
    for b in cuts:
        #for ele CR and SF:
        #os.system('echo python macro/plot_from_tree_skim_v6_on_v5_ntuples_ele_2017.py -c ' + str(b) + ' -d ' + str(b)+' -v ' + str(a) + ' -s signPUCMSFinal  -b \n')
        #os.system('python macro/plot_from_tree_skim_v6_on_v5_ntuples_ele_2017.py -c ' + str(b) + ' -d ' + str(b)+' -v ' + str(a) + ' -b -s signPUCMSFinal \n')

        #os.system('echo python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch -v ' + str(a) + ' -b -B\n')
        #os.system('python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SRmatch -v ' + str(a) + ' -b -B \n')
        os.system('echo python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s ' + str(b)+' -v ' + str(a) + ' -b \n')
        os.system('python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s ' + str(b)+' -v ' + str(a) + ' -b \n')

        #os.system('echo python macro/plot_from_tree_skim_v5.py -c ' + str(b) + ' -s ' + str(b)+' -v ' + str(a) + ' -b -B \n')
        #os.system('python macro/plot_from_tree_skim_v5.py -c ' + str(b) + ' -s ' + str(b)+' -v ' + str(a) + ' -b -B \n')

        #os.system('echo python macro/plot_from_tree_skim_v6_chris_hill_2018.py -c ' + str(b) + ' -s ' + str(b)+' -v ' + str(a) + ' -b \n')
        #os.system('python macro/plot_from_tree_skim_v6_chris_hill_2018.py -c ' + str(b) + ' -s ' + str(b)+' -v ' + str(a) + ' -b \n')


        #os.system('echo python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SR_match -v ' + str(a) + ' -b \n')
        #os.system('python macro/plot_from_tree_skim_v6.py -c ' + str(b) + ' -s SR_match -v ' + str(a) + ' -b \n')
        '''
        if "Jets" in a:
            if "JetsNegative" in a:
                b = cuts[0]+"_1nj"
                os.system('echo python macro/plot_from_tree_skim_v6_on_v5_ntuples.py -c ' + str(b) + ' -d ' + str(b)+' -v ' + str(a) + ' -s sig_1nj -b \n')
                os.system('python macro/plot_from_tree_skim_v6_on_v5_ntuples.py -c ' + str(b) + ' -d ' + str(b)+' -v ' + str(a) + ' -s sig_1nj -b \n')
            else:
                b = cuts[0]+"_1j"
                os.system('echo python macro/plot_from_tree_skim_v6_on_v5_ntuples.py -c ' + str(b) + ' -d ' + str(b)+' -v ' + str(a) + ' -s sig_1j -b \n')
                os.system('python macro/plot_from_tree_skim_v6_on_v5_ntuples.py -c ' + str(b) + ' -d ' + str(b)+' -v ' + str(a) + ' -s sig_1j -b \n')
        else:
            os.system('echo python macro/plot_from_tree_skim_v6_on_v5_ntuples.py -c ' + str(b) + ' -d ' + str(b)+' -v ' + str(a) + ' -s "isMC" -b \n')
            os.system('python macro/plot_from_tree_skim_v6_on_v5_ntuples.py -c ' + str(b) + ' -d ' + str(b)+' -v ' + str(a) + ' -s "isMC" -b \n')
        '''
