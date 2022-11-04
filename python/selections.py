#! /usr/bin/env python

long_string = "("
long_string += "Jets.Jets[0].isGenMatched" #new version from v3
long_string += ")"

selection = {
    "none" : "",
    "ele" : "isE && HT>200",

    "elePUCMSFinal" : "isE && HT>200 && MEt.pt> 70 && nCHSJetsAcceptanceCalo==1 && nPhotons==0 && Electrons[0].pt>70 && Jets[0].pt>70",
    "elePUCMSFinalNegative" : "isE && HT>200 && MEt.pt> 70 && nCHSJetsNegativeAcceptanceCalo==1 && nPhotons==0 && Electrons[0].pt>70 && JetsNegative[0].pt>70",

    "signPUCMSFinal" : "HT>200 && nCHSJetsAcceptanceCalo==1 && Jets[0].pt>70 && abs(Jets[0].eta)<1.",
    "signPUCMSFinalNegative" : "HT>200 && nCHSJetsAcceptanceCalo==1 && JetsNegative[0].pt>70 && abs(JetsNegative[0].eta)<1.",

    "elePUCaltech" : "isE && HT>200 && nCHSJetsAcceptanceCalo==1",

    "elePUCMS" : "isE && HT>200 && nCHSJetsAcceptanceCalo==1",
    "elePUCaltechVetoPho" : "isE && HT>200 && nCHSJetsAcceptanceCalo==1 && nPhotons==0",
    "elePUCMSVetoPho" : "isE && HT>200 && nCHSJetsAcceptanceCalo==1 && nPhotons==0",

    "elePUCMSMETPtVetoPho" : "isE && HT>200 && MEt.pt> 70 && nCHSJetsAcceptanceCalo==1 && nPhotons==0 && Electrons[0].pt>50 && Jets[0].pt>50",

    "elePUCMSMETPt70VetoPho" : "isE && HT>200 && MEt.pt> 70 && nCHSJetsAcceptanceCalo==1 && nPhotons==0 && Electrons[0].pt>70 && Jets[0].pt>70",
    "elePUCaltechMETPt70VetoPho" : "isE && HT>200 && MEt.pt> 70 && nCHSJetsAcceptanceCalo==1 && nPhotons==0 && Electrons[0].pt>70 && Jets[0].pt>70",

    "elePUCMSPtVetoPho" : "isE && HT>200 && nCHSJetsAcceptanceCalo==1 && nPhotons==0 && Electrons[0].pt>50 && Jets[0].pt>50",

    "ele_met50" : "isE && HT>200 && MEt.pt>50",
    "ele_jets50" : "isE && HT>200 && Jets.pt>50",
    "ele_metjets50" : "isE && HT>200 && Jets.pt>50 && MEt.pt>50",
    "ele_no_top" : "isE && HT>200",
    "isMC" : "isMC",
    "ctau500" : "",
    "ctau500match" : "Jets[0].isGenMatchedCaloCorrLLPAccept",
    "ctau3000" : "",
    "METctau500" : "MEt.pt>200",
    "METctau500match" : "MEt.pt>200 && Jets[0].isGenMatchedCaloCorrLLPAccept",
    "METctau3000" : "MEt.pt>200",
    "MET" : "MEt.pt>200",
    "match" : "Jets.isGenMatchedCaloCorrLLPAccept",
    "blind" : "nTagJets_0p996_JJ<2",
    "bin0" : "nTagJets_0p996_JJ==0",
    "tag" : "isSR && Jets.sigprob>0.996",
    "largeRMS" : "isSR && Jets.timeRMSRecHitsEB>5",
    "largeRMS_smallnEB" : "isSR && Jets.timeRMSRecHitsEB>5 && Jets.nRecHitsEB<5",
    "bin1" : "nTagJets_0p996_JJ==1",
    "bin2" : "nTagJets_0p996_JJ>1",
    "bin2_tagged" : "nTagJets_0p996_JJ>1 && Jets.sigprob>0.996",
    "bin2_tagged_eta_phi" : "nTagJets_0p996_JJ>1 && Jets.sigprob>0.996 && abs(Jets.eta)<1. && Jets.phi<2.5",
    "ZtoMM" : "isZtoMM",
    "TtoEM" : "isTtoEM && MEt.pt>200",
    "TtoEM_bin2" : "isTtoEM && MEt.pt>200 && abs(Jets.eta)<1 && nTagJets_0p996_JJ>1",
    "TtoEM_CSV" : "isTtoEM && Jets.CSV>0.9",
    "TtoEM_CSV_j0" : "isTtoEM && Jets[0].CSV>0.8 && MEt.pt>200",
    "TtoEM_CSV_nj0" : "isTtoEM && JetsNegative[0].CSV>0.8 && MEt.pt>200",
    "SR_negative" : "isSR && JetsNegative.timeRecHitsEB<-1 && MEt.pt>200",
    "SR_positive" : "isSR && JetsNegative.timeRecHitsEB>-1 && MEt.pt>200",
    #"DiJetMET" : "isDiJetMET && nCHSJetsAcceptanceCalo==2  && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v)",
    #"DiJetMET" : "isDiJetMET && nCHSJetsAcceptanceCalo==2 && MinLeadingJetMetDPhi<0.4 && MEt.pt<100  && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) ",
    "DiJetMET" : "isDiJetMET && nCHSJetsAcceptanceCalo==2 && MinLeadingJetMetDPhi<0.4 && MEt.pt<100 && (HLT_PFJet500_v) ",
    "DiJetMET140" : "isDiJetMET && nCHSJetsAcceptanceCalo==2 && MinSubLeadingJetMetDPhi<0.4 && MEt.pt<100 &&  (HLT_PFJet140_v)",
    "JetHT140" : "isJetHT && (HLT_PFJet140_v)",
    "JetHT260" : "isJetHT && (HLT_PFJet260_v)",
    "JetHT320" : "isJetHT && (HLT_PFJet320_v)",
    "JetHT400" : "isJetHT && (HLT_PFJet400_v)",
    "JetHT500" : "isJetHT && (HLT_PFJet500_v)",
    "JetHT" : "isJetHT && (HLT_PFJet500_v)",
    "JetHTLessMET" : "isJetHT && (HLT_PFJet500_v) && MEt.pt<25",
    "JetHTLessMETBlind" : "isJetHT && (HLT_PFJet500_v) && MEt.pt<25 && nTagJets_0p996_JJ<2",
    "JetMET" : "isJetMET && MEt.pt>100 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v)",
    "JetMET_500" : "isJetMET && MEt.pt>100 && (HLT_PFJet500_v)",
    "JetMET_Lep" : "nLeptons>0 && isJetMET && MEt.pt>100 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v)",

    "JetMET_low_dPhi_CR" : "isJetMET && MEt.pt>100 && MinLeadingJetMetDPhi>=0.5 && MinLeadingJetMetDPhi<=2 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v)",

    "JetMET_low_dPhi_500" : "isJetMET && MEt.pt>100 && MinLeadingJetMetDPhi>=0.5 && MinLeadingJetMetDPhi<=2 && (HLT_PFJet500_v)",

    "JetMET_low_dPhi_CR_blind" : "isJetMET && MEt.pt>100 && MinLeadingJetMetDPhi>=0.5 && MinLeadingJetMetDPhi<=2 && nTagJets_0p996_JJ<2  && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v)",
    "JetMET_low_dPhi_CR_bin2" : "isJetMET && MEt.pt>100 && MinLeadingJetMetDPhi>=0.5 && MinLeadingJetMetDPhi<=2 && nTagJets_0p996_JJ>1  && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v)",

    "JetMET_Lep_blind" : "nTagJets_0p996_JJ<2 && nLeptons>0 && isJetMET && MEt.pt>100 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v)",
    "JetMET_Lep_bin2" : "nTagJets_0p996_JJ>1 && nLeptons>0 && isJetMET && MEt.pt>100 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v)",


    "JetMET_low_dPhi_low_MET_CR" : "isJetMET && MEt.pt>100 && MEt.pt<200 && MinLeadingJetMetDPhi>=0.5 && MinLeadingJetMetDPhi<=2 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v)",
    "JetMET_low_dPhi" : "isJetMET && MEt.pt>100 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v)",
    "JetMET_low_dPhi_MET_200" : "isJetMET && MEt.pt>200 && MinLeadingJetMetDPhi>=0.5 && MinLeadingJetMetDPhi<=2 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v)",
    "JetMETUnprTr" : "isJetMET && HLT_PFJet500_v",
    "JetMETDiJetTr" : "isJetMET && (HLT_DiPFJetAve40_v || HLT_DiPFJetAve60_v || HLT_DiPFJetAve80_v || HLT_DiPFJetAve200_v || HLT_DiPFJetAve500_v)",
    "JetMETSingleJetTr" : "isJetMET && ( HLT_PFJet40_v || HLT_PFJet60_v || HLT_PFJet80_v || HLT_PFJet140_v || HLT_PFJet200_v || HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v || HLT_PFJet550_v)",
    "ZtoMM_HEM" : "isZtoMM && (isMC ? 1 : ( RunNumber>=319077 ? (nCHSJets_in_HEM==0) : 1))",
    #"ZtoMM_HEM" : "isZtoMM && (isMC ? 1 : ( RunNumber>=319077 ? (nCHSJets_in_HEM==0 && nElectrons_in_HEM && nPhotons_in_HEM) : 1))",
    "ZtoMM_RunC"  : "isZtoMM",
    "ZtoMM_bin0" : "isZtoMM && nTagJets_0p996_JJ==0 && nCHSJetsAcceptanceCalo>=1",
    "ZtoMM_bin1" : "isZtoMM && nTagJets_0p996_JJ==1",
    "ZtoMMPho"  : "fabs(Z_mass-91.2)<10.",# && Z_pho_tight",
    "ZtoMMPho_1j"  : "fabs(Z_mass-91.2)<10. && fabs(Jets[0].eta)<1",# && Z_pho_tight",
    "ZtoMMPho_1nj"  : "fabs(Z_mass-91.2)<10. && fabs(JetsNegative[0].eta)<1",# && Z_pho_tight",

    "sig_1j"  : "fabs(Jets[0].eta)<1 && Jets[0].isGenMatchedCaloCorrLLPAccept",
    "sig_1nj"  : "fabs(JetsNegative[0].eta)<1 && JetsNegative[0].isGenMatchedCaloCorrLLPAccept",

    "ZtoEEPho"  : "fabs(Z_mass-91.2)<10. && Z_pho_tight",
    "ZtoEE" : "isZtoEE",
    "ZtoEE_HEM" : "isZtoEE && (isMC ? 1 : ( RunNumber>=319077 ? (nCHSJets_in_HEM==0) : 1))",
    "MR" : "isMR && nLeptons==1 && MT<100",
    "MN" : "isMN && nLeptons==1",
    "EN" : "isEN && nLeptons==1",
    "SR_low_nRecHitsEB" : "isSR && Jets.nRecHitsEB<5",
    "WtoMN" : "isWtoMN && MT<100",
    "WtoMN_low_nRecHitsEB" : "isWtoMN && MT<100 && Jets.nRecHitsEB<10",
    "WtoMN_tag" : "isWtoMN && MT<100 && Jets.sigprob>0.996",
    "WtoMN_no_tag" : "isWtoMN && MT<100 && Jets.sigprob<=0.996",
    "WtoMN_noMT" : "isWtoMN",
    "WtoMN_noMT_yes_BH" : "isWtoMN && !Flag2_globalSuperTightHalo2016Filter",
    "WtoMN_noMT_bin1" : "isWtoMN && nTagJets_0p996_JJ==1",
    "WtoMN_noMT_bin1_BH" : "isWtoMN && nTagJets_0p996_JJ==1 && Flag2_globalSuperTightHalo2016Filter",
    "WtoMN_noMT_bin2" : "isWtoMN && nTagJets_0p996_JJ>1",
    "WtoMN_noMT_bin2_BH" : "isWtoMN && nTagJets_0p996_JJ>1 && Flag2_globalSuperTightHalo2016Filter",
    "WtoMN_noMT_MET100" : "isWtoMN && MEt.pt>100",
    "WtoMN_noMT_MET200" : "isWtoMN && MEt.pt>200",
    "WtoMNbin2" : "isWtoMN && MT<100 && nTagJets_0p996_JJ>1",
    "WtoEN" : "isWtoEN && MT<100",
    "WtoEN_noMT" : "isWtoEN",
    "WtoEN_bin0" : "isWtoEN && nTagJets_0p996_JJ==0",
    "WtoMN_bin0" : "isWtoMN && nTagJets_0p996_JJ==0",
    "WtoMN_HEM" : "isWtoMN && MT<100 && (isMC ? 1 : ( RunNumber>=319077 ? (nCHSJets_in_HEM==0) : 1))",
    "WtoEN_HEM" : "isWtoEN && MT<100 && (isMC ? 1 : ( RunNumber>=319077 ? (nCHSJets_in_HEM==0) : 1))",
    #"SR"  : "isSR && Flag2_globalSuperTightHalo2016Filter",
    "SR"  : "isSR",
    "SR_boost"  : "isSR && HT>250 && MEt.pt>250 && nTagJets_0p996_JJ<2",

    "SR_kill_qcd_met300"  : "isSR && MEt.pt>300 && MinJetMetDPhi>0.5 && HT>300",
    "SR_kill_qcd_met250"  : "isSR && MEt.pt>250 && MinJetMetDPhi>0.5 && HT>300",
    "SR_kill_qcd_met200"  : "isSR && MEt.pt>200 && MinJetMetDPhi>0.5 && HT>300",
    "SR_kill_qcd_ht200"  : "isSR && MinJetMetDPhi>0.5 && HT>200",
    "SR_kill_qcd_ht300"  : "isSR && MinJetMetDPhi>0.5 && HT>300 && MEt.pt>240",
    "SR_met250"  : "isSR && MEt.pt>300 && HT>300",
    "SR_met250"  : "isSR && MEt.pt>250 && HT>300",
    "SR_met200"  : "isSR && MEt.pt>200 && HT>300",
    "SR_ht200"  : "isSR && HT>200",
    "SR_ht300"  : "isSR && HT>300 && MEt.pt>240",
    "SR"  : "isSR",
    "SR_kill_qcd"  : "isSR && MinJetMetDPhi>0.5",

    "SR_compare"  : "isSR && MinJetMetDPhi>0.5",
    "SR_compare_low_pt"  : "isSR && MinJetMetDPhi>0.5 && Jets.pt<70",
    "SR_match_compare"  : "isSR && Jets.isGenMatchedCaloCorrLLPAccept && MinJetMetDPhi>0.5",
    "SR_match_compare_low_pt"  : "isSR && Jets.isGenMatchedCaloCorrLLPAccept && MinJetMetDPhi>0.5 && Jets.pt<70",

    "SR_match"  : "isSR && Jets.isGenMatchedCaloCorrLLPAccept",
    "SR_match_low_score"  : "isSR && Jets.isGenMatchedCaloCorrLLPAccept && Jets.sigprob<0.05",

    "SR_2cosmic"  : "isSR && nCosmicMuons>=2",
    "SR_1cosmiconeleg"  : "isSR && nCosmicMuonsOneLeg>=1",
    "SRPositiveWeights"  : "isSR && EventWeight>=0.",
    "SRNegativeWeights"  : "isSR && EventWeight<0.",
    "SR_clean"  : "isSR && dt_ecal_dist>0.5 && min_dPhi_jets_0p996>0.05",
    "bin2_clean"  : "isSR && dt_ecal_dist>0.5 && min_dPhi_jets_0p996>0.05 && nTagJets_0p996_JJ>1",
    #"SR_clean"  : "isSR && nTagJets_0p996_JJ>1",
    "SRmatch"  : "isSR && (isMC ? Jets.isGenMatchedCaloCorrLLPAccept : 1)",
    "SRmatch_tag"  : "isSR && (isMC ? Jets.isGenMatchedCaloCorrLLPAccept : 1) && Jets.sigprob>0.996",
    "SRmatch_less10hits"  : "isSR && (isMC ? Jets.isGenMatchedCaloCorrLLPAccept : 1) && Jets.nRecHitsEB<10",
    "SR_accumulation" : "isSR && nTagJets_0p996_JJ<2 && Jets.sigprob>0.6478790 && Jets.sigprob<0.6478792",
    "SR_no_accumulation" : "isSR && nTagJets_0p996_JJ<2 && (Jets.sigprob<=0.6478790 || Jets.sigprob>=0.6478792)",
    "SR_bin0"  : "isSR && nTagJets_0p996_JJ==0",
    "SR_bin1"  : "isSR && nTagJets_0p996_JJ==1",
    "SR_bin2"  : "isSR && nTagJets_0p996_JJ>1",
    #"SR_cosmic"  : "isSR && isDT_fit",
    "SR_cosmic_blinded"  : "isSR && isDT_fit && nTagJets_0p996_JJ<2",
    "SR_dnn_0p5"  : "isSR && Jets.sigprob>0.5 && Jets.phi<2.5",
    #"SR_dnn_0p7"  : "isSR && Jets.sigprob>0.7 && Jets.phi<2.5",
    "SR_dnn_0p7"  : "isSR && Jets.sigprob>0.7 && Jets.phi<2.5 && Jets.sigprob<0.996",
    "SR_dnn_0p7_CSC"  : "isSR && Jets.sigprob>0.7 && Jets.phi<2.5 && Jets.sigprob<0.996 && JetsCaloAdd.eFracEB2CSC==1.",
    "SR_dnn_0p7_closePhi"  : "isSR && Jets.sigprob>0.7 && Jets.phi<2.5 && Jets.sigprob<0.996 && min_dPhi_jets<0.05",
    "SR_dnn_0p7_closePhi_less10hits"  : "isSR && Jets.sigprob>0.7 && Jets.phi<2.5 && Jets.sigprob<0.996 && min_dPhi_jets<0.05 && Jets.nRecHitsEB<10",
    "SR_less10hits"  : "isSR && Jets.phi<2.5 && Jets.sigprob<0.996 && Jets.nRecHitsEB<10",
    "SR_closePhi"  : "isSR && Jets.phi<2.5 && Jets.sigprob<0.996 && min_dPhi_jets<0.05",
    "SR_largePhi"  : "isSR && min_dPhi_jets>0.05",
    #"SR_closePhi"  : "isSR && nTagJets_0p996_JJ<2 && min_dPhi_jets<0.05",
    "SR_closePhi_less10hits"  : "isSR && Jets.phi<2.5 && Jets.sigprob<0.996 && Jets.nRecHitsEB<10 && min_dPhi_jets<0.05",
    #"SR_dnn_0p7_closePhi_less10hits_also_sign"  : "isSR && Jets.sigprob>0.7 && Jets.phi<2.5 && Jets.sigprob<0.996 && min_dPhi_jets<0.05 && Jets.nRecHitsEB<10",
    "SR_dnn_0p7_closePhi_narrow"  : "isSR && Jets.sigprob>0.7 && Jets.phi<2.5 && Jets.sigprob<0.996 && min_dPhi_jets<0.05 && JetsCaloAdd.spreadXEB<1.",
    "SR_dnn_0p996"  : "isSR && Jets.sigprob>0.996 && Jets.phi<2.5",
    #"SR_dnn_0p9"  : "isSR && Jets.sigprob>0.9 && Jets.sigprob<0.996",
    "SR_dnn_0p9"  : "isSR && Jets.sigprob>0.9 && Jets.phi<2.5 && Jets.sigprob<0.996",
    #"SR_veto_bin2"  : "isSR && nTagJets_0p996_JJ<2",
    "SR_veto_bin2_cosmics"  : "isSR && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_veto_bin2_cosmics_match"  : "isSR && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_veto_bin2_cosmics_j"  : "isSR && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? Jets.phi<2.7 : 1) && ( RunNumber>=315252 ? (Jets.phi<0.4 || Jets.phi>0.9) : 1 )",
    "SR_veto_bin2_cosmics_nj"  : "isSR && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? JetsNegative.phi<2.7 : 1) && ( RunNumber>=315252 ? (JetsNegative.phi<0.4 || JetsNegative.phi>0.9) : 1 )",
    "SR_veto_bin2_cosmics_nj_match"  : "isSR && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? JetsNegative.phi<2.7 : 1) && ( RunNumber>=315252 ? (JetsNegative.phi<0.4 || JetsNegative.phi>0.9) : 1 ) && JetsNegative.isGenMatchedCaloCorrLLPAccept",

    "SR_veto_bin2_cosmics_nj_time"  : "isSR && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? JetsNegative.phi<2.7 : 1) && ( RunNumber>=315252 ? (JetsNegative.phi<0.4 || JetsNegative.phi>0.9) : 1 ) && JetsNegative.timeRecHitsEB<-1.",
    "SR_veto_bin2_cosmics_nj_time_match"  : "isSR && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? JetsNegative.phi<2.7 : 1) && ( RunNumber>=315252 ? (JetsNegative.phi<0.4 || JetsNegative.phi>0.9) : 1 ) && JetsNegative.timeRecHitsEB<-1. && JetsNegative.isGenMatchedCaloCorrLLPAccept",


    "SR_veto_bin2_cosmics_tag_nj"  : "isSR && JetsNegative.sigprob>0.996 && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? JetsNegative.phi<2.7 : 1) && ( RunNumber>=315252 ? (JetsNegative.phi<0.4 || JetsNegative.phi>0.9) : 1 )",
    "SR_veto_bin2_cosmics_tag_nj_0p9"  : "isSR && JetsNegative.sigprob>0.9 && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? JetsNegative.phi<2.7 : 1) && ( RunNumber>=315252 ? (JetsNegative.phi<0.4 || JetsNegative.phi>0.9) : 1 )",

    "SR_veto_bin2_cosmics_tag"  : "isSR && Jets.sigprob>0.996 && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? Jets.phi<2.7 : 1) && ( RunNumber>=315252 ? (Jets.phi<0.4 || Jets.phi>0.9) : 1 )",
    "SR_veto_bin2_cosmics_tag_match"  : "isSR && Jets.sigprob>0.996 && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? Jets.phi<2.7 : 1) && ( RunNumber>=315252 ? (Jets.phi<0.4 || Jets.phi>0.9) : 1 ) && Jets.isGenMatchedCaloCorrLLPAccept",
    "SR_veto_bin2_cosmics_tag_0p9"  : "isSR && Jets.sigprob>0.9 && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? Jets.phi<2.7 : 1) && ( RunNumber>=315252 ? (Jets.phi<0.4 || Jets.phi>0.9) : 1 )",
    "SR_veto_bin2_cosmics_tag_0p9_match"  : "isSR && Jets.sigprob>0.9 && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? Jets.phi<2.7 : 1) && ( RunNumber>=315252 ? (Jets.phi<0.4 || Jets.phi>0.9) : 1 ) && Jets.isGenMatchedCaloCorrLLPAccept",
    "SR_veto_bin2_cosmics_tag_0p9_largeRMS"  : "isSR && Jets.sigprob>0.9 && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? Jets.phi<2.7 : 1) && ( RunNumber>=315252 ? (Jets.phi<0.4 || Jets.phi>0.9) : 1 ) && Jets.timeRMSRecHitsEB>8",
    "SR_veto_bin2_cosmics_tag_0p9_largeRMS_match"  : "isSR && Jets.sigprob>0.9 && nTagJets_0p996_JJ<2 && (isMC ? 1 : (isDT_fit ? dt_ecal_dist>=0.5 : 1) ) && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && ( (RunNumber>=297020 && RunNumber<=306462) ? Jets.phi<2.7 : 1) && ( RunNumber>=315252 ? (Jets.phi<0.4 || Jets.phi>0.9) : 1 ) && Jets.isGenMatchedCaloCorrLLPAccept && Jets.timeRMSRecHitsEB>8",



    "SR_veto_bin2_and_1_jet"  : "isSR && nTagJets_0p996_JJ<2 && nCHSJetsAcceptanceCalo>0",
    "SR_HT"  : "isSR && Flag2_globalSuperTightHalo2016Filter && HT>200",
    "SR_eta_1p0"  : "isSR && Flag2_globalSuperTightHalo2016Filter && (nCHSJetsAcceptanceCalo==0 || fabs(Jets.eta)<1.)",
    "SR_0p996"  : "isSR && Flag2_globalSuperTightHalo2016Filter && Jets.sigprob>0.996",
    "SR_MinDPhi"  : "isSR && Flag2_globalSuperTightHalo2016Filter && MinJetMetDPhi>0.5",
    "SR_MinDPhi_ntuple"  : "isSR && Flag2_globalSuperTightHalo2016Filter && MinJetMetDPhi_ntuple>0.5",

    "SR_yes_BH"  : "isSR && !Flag2_globalSuperTightHalo2016Filter",
    "SR_no_BH"  : "isSR && Flag2_globalSuperTightHalo2016Filter",
    "SR_no_BH_bin1"  : "isSR && Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==1",
    "SR_no_BH_bin0"  : "isSR && Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==0",
    "SR_no_BH_MinDPhi"  : "isSR && Flag2_globalSuperTightHalo2016Filter && MinJetMetDPhi>0.5",
    "SR_no_BH_bin1_MinDPhi"  : "isSR && Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==1 && MinJetMetDPhi>0.5",
    "SR_no_BH_bin0_MinDPhi"  : "isSR && Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==0 && MinJetMetDPhi>0.5",

    "SR_noHEM"  : "isSR && Flag2_globalSuperTightHalo2016Filter && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_no_BH_noHEM"  : "isSR && Flag2_globalSuperTightHalo2016Filter && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_no_BH_noHEM_pt20"  : "isSR && Flag2_globalSuperTightHalo2016Filter && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_20_all_eta==0) : 1)",
    "SR_no_BH_bin1_noHEM"  : "isSR && Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==1 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_no_BH_bin0_noHEM"  : "isSR && Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==0 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_no_BH_MinDPhi_noHEM"  : "isSR && Flag2_globalSuperTightHalo2016Filter && MinJetMetDPhi>0.5 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_no_BH_MinDPhi_noHEM_pt20"  : "isSR && Flag2_globalSuperTightHalo2016Filter && MinJetMetDPhi>0.5 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_20_all_eta==0) : 1)",
    "SR_no_BH_bin1_MinDPhi_noHEM"  : "isSR && Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==1 && MinJetMetDPhi>0.5 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_no_BH_bin0_MinDPhi_noHEM"  : "isSR && Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==0 && MinJetMetDPhi>0.5 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",


    "SR_yes_BH"  : "isSR && !Flag2_globalSuperTightHalo2016Filter",
    "SR_yes_BH_bin1"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==1",
    "SR_yes_BH_bin0"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==0",
    "SR_yes_BH_MinDPhi"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && MinJetMetDPhi>0.5",
    "SR_yes_BH_bin1_MinDPhi"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==1 && MinJetMetDPhi>0.5",
    "SR_yes_BH_bin0_MinDPhi"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==0 && MinJetMetDPhi>0.5",

    "SR_yes_BH_noHEM"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_yes_BH_MinDPhi_noHEM"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && MinJetMetDPhi>0.5 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_yes_BH_bin1_noHEM"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==1 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_yes_BH_bin1_MinDPhi_noHEM"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && nTagJets_0p996_JJ==1 && MinJetMetDPhi>0.5 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",


    "SR_no_BH_tagged"  : "isSR && Flag2_globalSuperTightHalo2016Filter && Jets.sigprob>0.996",
    "SR_no_BH_MinDPhi_tagged"  : "isSR && Flag2_globalSuperTightHalo2016Filter && Jets.sigprob>0.996 && MinJetMetDPhi>0.5",

    "SR_no_BH_j0_tagged"  : "isSR && Flag2_globalSuperTightHalo2016Filter && Jets[0].sigprob>0.996",
    "SR_no_BH_MinDPhi_j0_tagged"  : "isSR && Flag2_globalSuperTightHalo2016Filter && Jets[0].sigprob>0.996 && MinJetMetDPhi>0.5",
    "SR_yes_BH_j0_tagged"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && Jets[0].sigprob>0.996",
    "SR_yes_BH_MinDPhi_j0_tagged"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && Jets[0].sigprob>0.996 && MinJetMetDPhi>0.5",

    "SR_no_BH_noHEM_tagged"  : "isSR && Flag2_globalSuperTightHalo2016Filter && Jets.sigprob>0.996 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_no_BH_MinDPhi_noHEM_tagged"  : "isSR && Flag2_globalSuperTightHalo2016Filter && Jets.sigprob>0.996 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && MinJetMetDPhi>0.5",

    "SR_yes_BH_tagged"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && Jets.sigprob>0.996",
    "SR_yes_BH_MinDPhi_tagged"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && Jets.sigprob>0.996 && MinJetMetDPhi>0.5",
    "SR_yes_BH_noHEM_tagged"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && Jets.sigprob>0.996 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1)",
    "SR_yes_BH_MinDPhi_noHEM_tagged"  : "isSR && !Flag2_globalSuperTightHalo2016Filter && Jets.sigprob>0.996 && ( RunNumber>=319077 ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && MinJetMetDPhi>0.5",



    "SRLeadDPhi"  : "isSR && MinLeadingJetMetDPhi>0.5",
    "SRDPhiBarr"  : "isSR && MinJetMetDPhiBarrel>0.5",
    "SRDPhi"  : "isSR && MinJetMetDPhi>0.5",
    "SRDPhi_ntuple"  : "isSR && MinJetMetDPhi_ntuple>0.5",
    "SRDPhiBoth"  : "isSR && MinJetMetDPhiBarrel>0.5 && MinLeadingJetMetDPhi>0.5",
    "SRHT"  : "HT>200 && MEt.pt>250",
    "SRHT_RunA"  : "HT>200 && MEt.pt>250",
    "SRHT_RunA_HEM"  : "HT>200 && MEt.pt>250 && (isMC ? 1 : ( RunNumber>=319077 ? nCHSJets_in_HEM==0 : 1))",
    "SRHT_RunC"  : "HT>200 && MEt.pt>250",
    "SRHT_RunC_HEM"  : "HT>200 && MEt.pt>250 && (isMC ? 1 : ( RunNumber>=319077 ? nCHSJets_in_HEM==0 : 1))",
    "SRHT_RunD"  : "HT>200 && MEt.pt>250",
    "SRHT_RunD_HEM"  : "HT>200 && MEt.pt>250 && (isMC ? 1 : ( RunNumber>=319077 ? nCHSJets_in_HEM==0 : 1))",

    "SR_RunD"  : "MEt.pt>200",
    "SR_RunD_HEM"  : "MEt.pt>200 && (isMC ? 1 : ( RunNumber>=319077 ? nCHSJets_in_HEM==0 : 1))",

    "SRHT_RunB"  : "HT>200 && MEt.pt>250",
    "SRHT_RunB_HEM"  : "HT>200 && MEt.pt>250 && (isMC ? 1 : ( RunNumber>=319077 ? nCHSJets_in_HEM==0 : 1))",
    "SR_RunA"  : "MEt.pt>200",
    "SR_bin0"  : "isSR && nCHSJetsAcceptanceCalo>=1 && nTagJets_0p996_JJ==0",
    "WtoMN_RunC" : "isWtoMN && MT<100",
    "WtoEN_RunC" : "isWtoEN && MT<100",
    "ZtoEE_RunC"  : "isZtoEE",
    #"TtoEM" : "isTtoEM",
    #"TtoEM_RunC" : "isTtoEM",
    "bin0" : "nTagJets_0p996_JJ==0 && nCHSJetsAcceptanceCalo>=1",
    "bin1" : "nTagJets_0p996_JJ==1",
    "bin2" : "nTagJets_0p996_JJ>1",
    "isMC" : "isMC",
    "ResolvedSUSY" : "isMC && GenHiggs[0].dRdaughters>0.4",
    "ResolvedHH" : "isMC && GenLLPs[0].dRdaughters>0.4",
    "BoostedSUSY" : "isMC && GenHiggs[0].dRdaughters<0.8",
    "BoostedHH" : "isMC && GenLLPs[0].dRdaughters<0.8",
    "OverlapSUSY" : "isMC && GenHiggs[0].dRdaughters>0.4 && GenHiggs[0].dRdaughters<0.8",
    "OverlapHH" : "isMC && GenLLPs[0].dRdaughters>0.4 && GenLLPs[0].dRdaughters<0.8",
    "VBF" : "isVBF",
    #Gen-level
    "SRsel" : "(HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v) && Flag2_globalSuperTightHalo2016Filter && Flag2_goodVertices && Flag2_EcalDeadCellTriggerPrimitiveFilter && Flag2_HBHENoiseFilter && Flag2_HBHEIsoNoiseFilter && Flag2_ecalBadCalibFilter && Flag2_eeBadScFilter && Flag2_BadPFMuonFilter && MEt.pt>200 && nMuonsPassing==0 && nElectronsPassing==0 && nTausPassing==0 && nPhotonsPassing==0",
    "SRsel" : "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v && MEt.pt>200 && nMuonsPassing==0 && nElectronsPassing==0 && nTausPassing==0 && nPhotonsPassing==0",

    "SR_fail_HBHE" : "(HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v) && !Flag2_HBHEIsoNoiseFilter && MEt.pt>200 && nMuonsPassing==0 && nElectronsPassing==0 && nTausPassing==0 && nPhotonsPassing==0",
    "SR_fail_HBHE_match" : "(HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v) && !Flag2_HBHEIsoNoiseFilter && MEt.pt>200 && nMuonsPassing==0 && nElectronsPassing==0 && nTausPassing==0 && nPhotonsPassing==0 && Jets.isGenMatchedCaloCorrLLPAccept",
    "SR_gen_match" : "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v && MEt.pt>200 && nMuonsPassing==0 && nElectronsPassing==0 && nTausPassing==0 && nPhotonsPassing==0 && Jets.isGenMatchedCaloCorrLLPAccept",


    "1j_h0" : "n_j_higgs_0==1",
    "1j_h1" : "n_j_higgs_1==1",
    "2j_h0" : "n_j_higgs_0==2",
    "2j_h1" : "n_j_higgs_1==2",
    "LLP0_prompt" : "fabs(GenLLPs[0].travelRadius)<30",
    "LLP1_prompt" : "fabs(GenLLPs[1].travelRadius)<30",
    "LLP0_calo" : "fabs(GenLLPs[0].travelRadius)>30 && fabs(GenLLPs[0].travelRadius)<184.",
    "LLP1_calo" : "fabs(GenLLPs[1].travelRadius)>30 && fabs(GenLLPs[1].travelRadius)<184.",

    "LLP0_calo_MET" : "fabs(GenLLPs[0].travelRadius)>30 && fabs(GenLLPs[0].travelRadius)<184. && MEt.pt>200",
    "LLP1_calo_MET" : "fabs(GenLLPs[1].travelRadius)>30 && fabs(GenLLPs[1].travelRadius)<184. && MEt.pt>200",

    "2j_h0_prompt" : "n_j_higgs_0==2 && fabs(GenHiggs[0].travelRadiusLLP)<30",
    "2j_h1_prompt" : "n_j_higgs_1==2 && fabs(GenHiggs[1].travelRadiusLLP)<30",
    "2j_h0_calo" : "n_j_higgs_0==2 && fabs(GenHiggs[0].travelRadiusLLP)<184. && fabs(GenHiggs[0].travelRadiusLLP)>30",
    "2j_h1_calo" : "n_j_higgs_1==2 && fabs(GenHiggs[1].travelRadiusLLP)<184. && fabs(GenHiggs[1].travelRadiusLLP)>30",

    "MET" : "MEt.pt>200",

}
selection["PreselSkim"] = "MEt.pt>200"# && EventNumber%2!=0"
selection["PreselSkimMatchAK8"] = "MEt.pt>200 && FatJets[0].nMatchedGenBquarksCaloCorr==2"# && EventNumber%2!=0"
selection["PreselSkimAK4JJ"] = "MEt.pt>200 && nCHSJetsAcceptanceCalo>=1"# && EventNumber%2!=0"
selection["PreselSkimAK4JJCleanQCD"] = "MinJetMetDPhiBarrel>0.5 && MEt.pt>200 && nCHSJetsAcceptanceCalo>=1"# && EventNumber%2!=0"
selection["PreselSkimAK4JJHTCleanQCD"] = "HT>100 && MinJetMetDPhiBarrel>0.5 && MEt.pt>200 && nCHSJetsAcceptanceCalo>=1"# && EventNumber%2!=0"
selection["PreselSkimAK4JJHT"] = "HT>100 && MEt.pt>200 && nCHSJetsAcceptanceCalo>=1"# && EventNumber%2!=0"

selection["PreselSkimAK4JJSR"] = "nTagJets_0p994_JJ>=2 && HT>100 && MEt.pt>200 && nCHSJetsAcceptanceCalo>=1"# && EventNumber%2!=0"

selection["check"] = "MEt.pt>200 && nCHSJetsAcceptanceCalo>=1 && isTagAK4_0p994 "


selection["PreselSkimAK4"] = "HT>200 && MEt.pt>200 && nCHSJetsAcceptanceCalo>=1"# && EventNumber%2!=0"
selection["PreselSkimAK4Odd"] = "HT>200 && MEt.pt>200 && nCHSJetsAcceptanceCalo>=1 && EventNumber%2!=0"
#selection["PreselSkimAK4New"] = "HT>200 && MEt.pt>250 && MEt.pt<700 && nCHSJets>=1"# && EventNumber%2!=0"
selection["PreselSkimAK4Match"] = "MEt.pt>200 && nCHSJets>=1 && Jets[0].isGenMatchedCaloCorrLLPAccept"# && EventNumber%2!=0"
selection["PreselSkimAK8"] = "HT>200 && MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1"# && EventNumber%2!=0"
selection["PreselSkimAK8JJ"] = "MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1"# && EventNumber%2!=0"

selection["PreselSkimAK8170"] = "MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1 && isTagAK8_0p999999_170"
selection["PreselSkimAK8200"] = "MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1 && isTagAK8_0p999999_200"
selection["PreselSkimAK8250"] = "MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1 && isTagAK8_0p999999_250"
selection["PreselSkimAK8300"] = "MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1 && isTagAK8_0p999999_300"
selection["PreselSkimAK8350"] = "MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1 && isTagAK8_0p999999_350"




selection["PreselSkimAK8JJHT"] = "HT>100 & MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1"# && EventNumber%2!=0"
selection["PreselSkimAK8JJHT250"] = "isAK8_JJ_250 && HT>100 & MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1"# && EventNumber%2!=0"
selection["PreselSkimAK8JJHT300"] = "FatJets[0].pt>300 && HT>100 & MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1"# && EventNumber%2!=0"
selection["PreselSkimAK8JJHT350"] = "FatJets[0].pt>350 && HT>100 & MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1"# && EventNumber%2!=0"

selection["PreselSkimAK8PtCut"] = "HT>200 && MEt.pt>200 && FatJets[0].pt>200 && nCHSFatJets>=1"# && EventNumber%2!=0"
selection["PreselSkimAK8Match"] = "MEt.pt>200 && nCHSFatJets>=1 && FatJets[0].isGenMatched"# && EventNumber%2!=0"
selection["PreselTrainDNNSkim"] = "MEt_pt>200 && nCHSJets>=1"# && EventNumber%2!=0"
selection["PreselTrainDNNSkimAK8"] = "MEt_pt>200 && nCHSFatJets>=1"# && EventNumber%2!=0"
