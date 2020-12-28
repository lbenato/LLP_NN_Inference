#! /usr/bin/env python

long_string = "("
long_string += "Jets.Jets[0].isGenMatched" #new version from v3
long_string += ")"

selection = {
    "none" : "",
    "isMC" : "isMC",
    "ResolvedSUSY" : "isMC && GenHiggs[0].dRdaughters>0.4",
    "ResolvedHH" : "isMC && GenLLPs[0].dRdaughters>0.4",
    "BoostedSUSY" : "isMC && GenHiggs[0].dRdaughters<0.8",
    "BoostedHH" : "isMC && GenLLPs[0].dRdaughters<0.8",
    "OverlapSUSY" : "isMC && GenHiggs[0].dRdaughters>0.4 && GenHiggs[0].dRdaughters<0.8",
    "OverlapHH" : "isMC && GenLLPs[0].dRdaughters>0.4 && GenLLPs[0].dRdaughters<0.8",
    "VBF" : "isVBF",
}
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
selection["PreselSkimAK8JJHT"] = "HT>100 & MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1"# && EventNumber%2!=0"
selection["PreselSkimAK8JJHT250"] = "isAK8_JJ_250 && HT>100 & MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1"# && EventNumber%2!=0"
selection["PreselSkimAK8JJHT300"] = "FatJets[0].pt>300 && HT>100 & MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1"# && EventNumber%2!=0"
selection["PreselSkimAK8JJHT350"] = "FatJets[0].pt>350 && HT>100 & MEt.pt>200 && nCHSFatJetsAcceptanceCalo>=1"# && EventNumber%2!=0"

selection["PreselSkimAK8PtCut"] = "HT>200 && MEt.pt>200 && FatJets[0].pt>200 && nCHSFatJets>=1"# && EventNumber%2!=0"
selection["PreselSkimAK8Match"] = "MEt.pt>200 && nCHSFatJets>=1 && FatJets[0].isGenMatched"# && EventNumber%2!=0"
selection["PreselTrainDNNSkim"] = "MEt_pt>200 && nCHSJets>=1"# && EventNumber%2!=0"
selection["PreselTrainDNNSkimAK8"] = "MEt_pt>200 && nCHSFatJets>=1"# && EventNumber%2!=0"
