#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
from ROOT import TLorentzVector
from collections import defaultdict

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
(options, args) = parser.parse_args()


gROOT.SetBatch(True)
gStyle.SetOptStat(0)

from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *


masses = [127,150,175,300]
ctaus = [500,3000]

f = defaultdict(dict)
H_pt = defaultdict(dict)
H_pt_met200 = defaultdict(dict)

LLP_pt = defaultdict(dict)
LLP_pt_met200 = defaultdict(dict)

LLP_boost = defaultdict(dict)
LLP_boost_met200 = defaultdict(dict)

g_met_pt_1LLP = defaultdict(dict)
g_met_pt_met200_1LLP = defaultdict(dict)
met_pt_1LLP = defaultdict(dict)

dH_pt = defaultdict(dict)
dH_pt_met200 = defaultdict(dict)

met_pt = defaultdict(dict)
met_pt_met200 = defaultdict(dict)


G_pt = defaultdict(dict)
G_pt_met200 = defaultdict(dict)
diG_pt = defaultdict(dict)
diG_pt_met200 = defaultdict(dict)


for m in masses:
    for ct in ctaus:
        g_met_pt_1LLP[m][ct] = TH1F("g_met_pt_m"+str(m)+"_ct"+str(ct),"",50,0,1000)
        g_met_pt_met200_1LLP[m][ct] = TH1F("g_met_pt_met200_m"+str(m)+"_ct"+str(ct),"",50,0,1000)
        met_pt_1LLP[m][ct] = TH1F("met_pt_m"+str(m)+"_ct"+str(ct),"",50,0,1000)
        g_met_pt_1LLP[m][ct].Sumw2()
        g_met_pt_met200_1LLP[m][ct].Sumw2()
        met_pt_1LLP[m][ct].Sumw2()

        LLP_boost[m][ct] = TH1F("LLP_boost_m"+str(m)+"_ct"+str(ct),"",50,0,1000)
        LLP_boost_met200[m][ct] = TH1F("LLP_boost_met200_m"+str(m)+"_ct"+str(ct),"",50,0,1000)

'''
for m in masses:
    for ct in ctaus:
        f[m][ct] = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_Gen/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m"+str(m)+"_ctau"+str(ct)+"_HH.root"
        tmp_file = TFile(f[m][ct],"READ")
        tree = tmp_file.Get("tree")
        tree_weight = tree.GetWeight()
        print "file ", f[m][ct]
        for e in range(0,tree.GetEntries()):
            if(e % 5000==0):
                print "event ", e
            tree.GetEntry(e)
            gravitino = getattr(tree,"GenGravitinos")
            higgs = getattr(tree,"GenHiggs")
            llp = getattr(tree,"GenLLPs")
            met = getattr(tree,"MEt")
            llp0 = TLorentzVector(llp[0].px, llp[0].py, llp[0].pz, llp[0].energy) 
            llp1 = TLorentzVector(llp[1].px, llp[1].py, llp[1].pz, llp[1].energy) 

            boost = (llp0 + llp1).Pt()
            LLP_boost[m][ct].Fill( boost )
            if met.pt>200:
                LLP_boost_met200[m][ct].Fill( boost )
            g0 = TLorentzVector(gravitino[0].px, gravitino[0].py, gravitino[0].pz, gravitino[0].energy)
            g1 = TLorentzVector(gravitino[1].px, gravitino[1].py, gravitino[1].pz, gravitino[1].energy)
            h = TLorentzVector()
            if tree.nLLPInCalo==1:
                if llp[0].isLLPInCaloAcceptance and llp[0].travelRadius>100:
                    h = TLorentzVector(higgs[0].px, higgs[0].py, higgs[0].pz, higgs[0].energy)
                if llp[1].isLLPInCaloAcceptance and llp[1].travelRadius>100:
                    h = TLorentzVector(higgs[1].px, higgs[1].py, higgs[1].pz, higgs[1].energy)

                gen_met = (g0 + g1 + h).Pt()
                g_met_pt_1LLP[m][ct].Fill( gen_met  )
                met_pt_1LLP[m][ct].Fill( met.pt  )

                if met.pt>200:
                    g_met_pt_met200_1LLP[m][ct].Fill( gen_met  )
            
        print "filled ", m, ct
        LLP_boost[m][ct].Scale( tree_weight )
        LLP_boost_met200[m][ct].Scale( tree_weight )
        g_met_pt_1LLP[m][ct].Scale( tree_weight )
        met_pt_1LLP[m][ct].Scale( tree_weight )
        g_met_pt_met200_1LLP[m][ct].Scale( tree_weight )
'''

for m in masses:
    for ct in ctaus:
        f[m][ct] = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_Gen/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m"+str(m)+"_ctau"+str(ct)+"_HH.root"
        H_pt[m][ct] = TH1F("H_pt_m"+str(m)+"_ct"+str(ct),"",50,0,1000)
        H_pt_met200[m][ct] = TH1F("H_pt_met200_m"+str(m)+"_ct"+str(ct),"",50,0,1000)
        H_pt[m][ct].Sumw2()
        H_pt_met200[m][ct].Sumw2()

        LLP_pt[m][ct] = TH1F("LLP_pt_m"+str(m)+"_ct"+str(ct),"",50,0,2000)
        LLP_pt_met200[m][ct] = TH1F("LLP_pt_met200_m"+str(m)+"_ct"+str(ct),"",50,0,2000)
        LLP_pt[m][ct].Sumw2()
        LLP_pt_met200[m][ct].Sumw2()
 
        G_pt[m][ct] = TH1F("G_pt_m"+str(m)+"_ct"+str(ct),"",50,0,250)
        G_pt_met200[m][ct] = TH1F("G_pt_met200_m"+str(m)+"_ct"+str(ct),"",50,0,250)
        G_pt[m][ct].Sumw2()
        G_pt_met200[m][ct].Sumw2()

        diG_pt[m][ct] = TH1F("diG_pt_m"+str(m)+"_ct"+str(ct),"",50,0,250)
        diG_pt_met200[m][ct] = TH1F("diG_pt_met200_m"+str(m)+"_ct"+str(ct),"",50,0,250)
        diG_pt[m][ct].Sumw2()
        diG_pt_met200[m][ct].Sumw2()

        dH_pt[m][ct] = TH1F("dH_pt_m"+str(m)+"_ct"+str(ct),"",50,0,1000)
        dH_pt_met200[m][ct] = TH1F("dH_pt_met200_m"+str(m)+"_ct"+str(ct),"",25,0,500)
        dH_pt[m][ct].Sumw2()
        dH_pt_met200[m][ct].Sumw2()

        met_pt[m][ct] = TH1F("met_pt_m"+str(m)+"_ct"+str(ct),"",10,150,550)
        met_pt_met200[m][ct] = TH1F("met_pt_met200_m"+str(m)+"_ct"+str(ct),"",10,150,550)
        met_pt[m][ct].Sumw2()
        met_pt_met200[m][ct].Sumw2()

        chain = TChain("tree")
        chain.Add(f[m][ct])
        chain.Project("H_pt_m"+str(m)+"_ct"+str(ct), "GenHiggs.pt", "")
        chain.Project("LLP_pt_m"+str(m)+"_ct"+str(ct), "GenLLPs.pt", "")
        chain.Project("G_pt_m"+str(m)+"_ct"+str(ct), "GenGravitinos.pt", "")
        chain.Project("diG_pt_m"+str(m)+"_ct"+str(ct), "DiGravitino_pt", "")
        chain.Project("met_pt_m"+str(m)+"_ct"+str(ct), "MEt.pt", "")

        chain.Project("H_pt_met200_m"+str(m)+"_ct"+str(ct), "GenHiggs.pt", "MEt.pt>200")
        chain.Project("LLP_pt_met200_m"+str(m)+"_ct"+str(ct), "GenLLPs.pt", "MEt.pt>200")
        chain.Project("G_pt_met200_m"+str(m)+"_ct"+str(ct), "GenGravitinos.pt", "MEt.pt>200")
        chain.Project("diG_pt_met200_m"+str(m)+"_ct"+str(ct), "DiGravitino_pt", "MEt.pt>200")
        chain.Project("met_pt_met200_m"+str(m)+"_ct"+str(ct), "MEt.pt", "MEt.pt>150")

        chain.Project("dH_pt_m"+str(m)+"_ct"+str(ct), "GenHiggs[0].pt", "GenLLPs[0].travelRadius>100")
        chain.Project("dH_pt_met200_m"+str(m)+"_ct"+str(ct), "GenHiggs[0].pt", "GenLLPs[0].travelRadius>100 && MEt.pt>200")


can = TCanvas("can","can",1000,800)
can.SetRightMargin(0.05)
can.SetLeftMargin(0.12)
can.SetBottomMargin(0.1)
can.cd()
leg = TLegend(0.6, 0.6, 0.99, 0.99)
leg.SetTextSize(0.035)
for m in masses:
    for ct in [3000,500]:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        H_pt[m][ct].SetLineColor(samples[s]['linecolor'])
        H_pt[m][ct].SetLineStyle(samples[s]['linestyle'])
        H_pt[m][ct].SetLineWidth(3)
        H_pt_met200[m][ct].SetLineColor(samples[s]['linecolor'])
        H_pt_met200[m][ct].SetLineStyle(samples[s]['linestyle'])
        H_pt_met200[m][ct].SetMarkerColor(samples[s]['linecolor'])
        H_pt_met200[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        H_pt_met200[m][ct].SetMarkerSize(1.2)
        H_pt_met200[m][ct].SetLineWidth(3)
        H_pt[m][ct].SetMinimum(1)
        H_pt_met200[m][ct].GetXaxis().SetTitle("Gen Higgs p_{T} (GeV)")
        #H_pt[m][ct].Draw("HIST,sames")
        H_pt_met200[m][ct].Draw("PE,sames")
        #leg.AddEntry(H_pt[m][ct],samples[s]['label'],"FL")
        #leg.AddEntry(H_pt_met200[m][ct],"+ MET>200 GeV","P")
        leg.AddEntry(H_pt_met200[m][ct],samples[s]['label'],"PL")

leg.Draw("sames")
can.SetLogy()
drawCMS_simple(-1, "Simulation", onTop=True)
can.Print('GenHiggs_pt_compare.png')
can.Print('GenHiggs_pt_compare.pdf')

can.Close()


can = TCanvas("can","can",1000,800)
can.SetRightMargin(0.05)
can.SetLeftMargin(0.12)
can.SetBottomMargin(0.1)
can.cd()
for m in masses:
    for ct in [3000,500]:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        LLP_pt[m][ct].SetLineColor(samples[s]['linecolor'])
        LLP_pt[m][ct].SetLineStyle(samples[s]['linestyle'])
        LLP_pt[m][ct].SetLineWidth(3)
        LLP_pt_met200[m][ct].SetLineColor(samples[s]['linecolor'])
        LLP_pt_met200[m][ct].SetLineStyle(samples[s]['linestyle'])
        LLP_pt_met200[m][ct].SetMarkerColor(samples[s]['linecolor'])
        LLP_pt_met200[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        LLP_pt_met200[m][ct].SetMarkerSize(1.2)
        LLP_pt_met200[m][ct].SetLineWidth(3)
        LLP_pt[m][ct].SetMinimum(1)
        LLP_pt_met200[m][ct].GetXaxis().SetTitle("Gen LLP p_{T} (GeV)")
        #LLP_pt[m][ct].Draw("HIST,sames")
        LLP_pt_met200[m][ct].Draw("PE,sames")

leg.Draw("sames")
can.SetLogy()
drawCMS_simple(-1, "Simulation", onTop=True)
can.Print('GenLLPs_pt_compare.png')
can.Print('GenLLPs_pt_compare.pdf')

can.Close()


can = TCanvas("can","can",1000,800)
can.SetRightMargin(0.05)
can.SetLeftMargin(0.12)
can.SetBottomMargin(0.1)
can.cd()
for m in list(reversed(masses)):
    for ct in [3000,500]:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        LLP_boost[m][ct].SetLineColor(samples[s]['linecolor'])
        LLP_boost[m][ct].SetLineStyle(samples[s]['linestyle'])
        LLP_boost[m][ct].SetLineWidth(3)
        LLP_boost_met200[m][ct].SetLineColor(samples[s]['linecolor'])
        LLP_boost_met200[m][ct].SetLineStyle(samples[s]['linestyle'])
        LLP_boost_met200[m][ct].SetMarkerColor(samples[s]['linecolor'])
        LLP_boost_met200[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        LLP_boost_met200[m][ct].SetMarkerSize(1.2)
        LLP_boost_met200[m][ct].SetLineWidth(3)
        LLP_boost[m][ct].SetMinimum(1)
        LLP_boost_met200[m][ct].GetXaxis().SetTitle("Gen di-LLP p_{T} (GeV)")
        #LLP_boost[m][ct].Draw("HIST,sames")
        LLP_boost_met200[m][ct].Draw("PE,sames")

leg.Draw("sames")
can.SetLogy()
drawCMS_simple(-1, "Simulation", onTop=True)
can.Print('GenLLPs_boost_compare.png')
can.Print('GenLLPs_boost_compare.pdf')

can.Close()


can = TCanvas("can","can",1000,800)
can.SetRightMargin(0.05)
can.SetLeftMargin(0.12)
can.SetBottomMargin(0.1)
can.cd()
for m in masses:
    for ct in [3000,500]:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        dH_pt[m][ct].SetLineColor(samples[s]['linecolor'])
        dH_pt[m][ct].SetLineStyle(samples[s]['linestyle'])
        dH_pt[m][ct].SetLineWidth(3)
        dH_pt_met200[m][ct].SetLineColor(samples[s]['linecolor'])
        dH_pt_met200[m][ct].SetLineStyle(samples[s]['linestyle'])
        dH_pt_met200[m][ct].SetMarkerColor(samples[s]['linecolor'])
        dH_pt_met200[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        dH_pt_met200[m][ct].SetMarkerSize(1.2)
        dH_pt_met200[m][ct].SetLineWidth(3)
        dH_pt[m][ct].SetMinimum(1)
        dH_pt_met200[m][ct].GetXaxis().SetTitle("Displaced Gen Higgs p_{T} (GeV)")
        #dH_pt[m][ct].Draw("HIST,sames")
        dH_pt_met200[m][ct].Draw("PE,sames")

leg.Draw("sames")
can.SetLogy()
drawCMS_simple(-1, "Simulation", onTop=True)
can.Print('Displaced_GenHiggs_pt_compare.png')
can.Print('Displaced_GenHiggs_pt_compare.pdf')

can.Close()




leg = TLegend(0.6, 0.6, 0.99, 0.99)
leg.SetTextSize(0.035)
can = TCanvas("can","can",1000,800)
can.SetRightMargin(0.05)
can.SetLeftMargin(0.12)
can.SetBottomMargin(0.1)
can.cd()
for m in list(reversed(masses)):
    for ct in [500]:#,3000]:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        met_pt[m][ct].SetLineColor(samples[s]['linecolor'])
        met_pt[m][ct].SetLineStyle(samples[s]['linestyle'])
        met_pt[m][ct].SetLineWidth(3)
        met_pt_met200[m][ct].SetLineColor(samples[s]['linecolor'])
        met_pt_met200[m][ct].SetLineStyle(samples[s]['linestyle'])
        met_pt_met200[m][ct].SetMarkerColor(samples[s]['linecolor'])
        met_pt_met200[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        met_pt_met200[m][ct].SetMarkerSize(1.2)
        met_pt_met200[m][ct].SetLineWidth(3)
        met_pt[m][ct].SetMinimum(1)
        met_pt_met200[m][ct].Scale(1./met_pt_met200[m][ct].Integral())
        met_pt_met200[m][ct].GetXaxis().SetTitle("Reco E_{T}^{miss} (GeV)")
        #met_pt[m][ct].Draw("HIST,sames")
        leg.AddEntry(met_pt_met200[m][ct],samples[s]['label'],"PL")
        met_pt_met200[m][ct].Draw("PE,sames")

leg.Draw("sames")
can.SetLogy()
drawCMS_simple(-1, "Simulation", onTop=True)
can.Print('Met_pt_compare_500.png')
can.Print('Met_pt_compare_500.pdf')

can.Close()


leg = TLegend(0.6, 0.6, 0.99, 0.99)
leg.SetTextSize(0.035)
can = TCanvas("can","can",1000,800)
can.SetRightMargin(0.05)
can.SetLeftMargin(0.12)
can.SetBottomMargin(0.1)
can.cd()
for m in list(reversed(masses)):
    for ct in [3000]:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        met_pt[m][ct].SetLineColor(samples[s]['linecolor'])
        met_pt[m][ct].SetLineStyle(samples[s]['linestyle'])
        met_pt[m][ct].SetLineWidth(3)
        met_pt_met200[m][ct].SetLineColor(samples[s]['linecolor'])
        met_pt_met200[m][ct].SetLineStyle(samples[s]['linestyle'])
        met_pt_met200[m][ct].SetMarkerColor(samples[s]['linecolor'])
        met_pt_met200[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        met_pt_met200[m][ct].SetMarkerSize(1.2)
        met_pt_met200[m][ct].SetLineWidth(3)
        met_pt[m][ct].SetMinimum(1)
        met_pt_met200[m][ct].Scale(1./met_pt_met200[m][ct].Integral())
        met_pt_met200[m][ct].GetXaxis().SetTitle("Reco E_{T}^{miss} (GeV)")
        #met_pt[m][ct].Draw("HIST,sames")
        leg.AddEntry(met_pt_met200[m][ct],samples[s]['label'],"PL")
        met_pt_met200[m][ct].Draw("PE,sames")

leg.Draw("sames")
can.SetLogy()
drawCMS_simple(-1, "Simulation", onTop=True)
can.Print('Met_pt_compare_3000.png')
can.Print('Met_pt_compare_3000.pdf')

can.Close()



can = TCanvas("can","can",1000,800)
can.SetRightMargin(0.05)
can.SetLeftMargin(0.12)
can.SetBottomMargin(0.1)
can.cd()
for m in masses:
    for ct in ctaus:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        G_pt[m][ct].SetLineColor(samples[s]['linecolor'])
        G_pt[m][ct].SetLineStyle(samples[s]['linestyle'])
        G_pt[m][ct].SetLineWidth(3)
        G_pt_met200[m][ct].SetLineColor(samples[s]['linecolor'])
        G_pt_met200[m][ct].SetLineStyle(samples[s]['linestyle'])
        G_pt_met200[m][ct].SetMarkerColor(samples[s]['linecolor'])
        G_pt_met200[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        G_pt_met200[m][ct].SetMarkerSize(1.2)
        G_pt_met200[m][ct].SetLineWidth(3)
        G_pt[m][ct].SetMinimum(1)
        G_pt_met200[m][ct].GetXaxis().SetTitle("Gen Gravitino p_{T} (GeV)")
        #G_pt[m][ct].Draw("HIST,sames")
        G_pt_met200[m][ct].Draw("PE,sames")

leg.Draw("sames")
can.SetLogy()
drawCMS_simple(-1, "Simulation", onTop=True)
can.Print('GenGravitino_pt_compare.png')
can.Print('GenGravitino_pt_compare.pdf')
can.Close()

can = TCanvas("can","can",1000,800)
can.SetRightMargin(0.05)
can.SetLeftMargin(0.12)
can.SetBottomMargin(0.1)
can.cd()
for m in masses:
    for ct in ctaus:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        diG_pt[m][ct].SetLineColor(samples[s]['linecolor'])
        diG_pt[m][ct].SetLineStyle(samples[s]['linestyle'])
        diG_pt[m][ct].SetLineWidth(3)
        diG_pt_met200[m][ct].SetLineColor(samples[s]['linecolor'])
        diG_pt_met200[m][ct].SetLineStyle(samples[s]['linestyle'])
        diG_pt_met200[m][ct].SetMarkerColor(samples[s]['linecolor'])
        diG_pt_met200[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        diG_pt_met200[m][ct].SetMarkerSize(1.2)
        diG_pt_met200[m][ct].SetLineWidth(3)
        diG_pt[m][ct].SetMinimum(0.1)
        diG_pt_met200[m][ct].GetXaxis().SetTitle("Gen DiGravitino p_{T} (GeV)")
        #diG_pt[m][ct].Draw("HIST,sames")
        diG_pt_met200[m][ct].Draw("PE,sames")

leg.Draw("sames")
can.SetLogy()
drawCMS_simple(-1, "Simulation", onTop=True)
can.Print('DiGravitino_pt_compare.png')
can.Print('DiGravitino_pt_compare.pdf')
can.Close()

for m in masses:
    leg = TLegend(0.5, 0.6, 0.99, 0.99)
    leg.SetTextSize(0.035)
    can = TCanvas("can","can",1000,800)
    can.cd()
    for ct in ctaus:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        met_pt_1LLP[m][ct].SetLineColor(samples[s]['linecolor'])
        met_pt_1LLP[m][ct].SetLineStyle(samples[s]['linestyle'])
        met_pt_1LLP[m][ct].SetLineWidth(3)
        #g_met_pt_met200_1LLP[m][ct].SetLineColor(samples[s]['linecolor'])
        #g_met_pt_met200_1LLP[m][ct].SetLineStyle(samples[s]['linestyle'])
        #g_met_pt_met200_1LLP[m][ct].SetMarkerColor(samples[s]['linecolor'])
        #g_met_pt_met200_1LLP[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        #g_met_pt_met200_1LLP[m][ct].SetMarkerSize(1.2)
        #g_met_pt_met200_1LLP[m][ct].SetLineWidth(3)
        g_met_pt_1LLP[m][ct].SetMinimum(0.1)
        met_pt_1LLP[m][ct].SetMinimum(0.1)
        g_met_pt_1LLP[m][ct].GetXaxis().SetTitle("Reconstructed Gen MET [gravitinos + 1 displaced Higgs] (GeV)")
        met_pt_1LLP[m][ct].GetXaxis().SetTitle("Reconstructed Gen MET [gravitinos + 1 displaced Higgs] (GeV)")
        g_met_pt_1LLP[m][ct].SetLineColor(samples[s]['linecolor'])
        g_met_pt_1LLP[m][ct].SetMarkerColor(samples[s]['linecolor'])
        g_met_pt_1LLP[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        g_met_pt_1LLP[m][ct].SetMarkerSize(1.2)
        g_met_pt_1LLP[m][ct].SetLineStyle(samples[s]['linestyle'])
        #met_pt_1LLP[m][ct].SetFillColorAlpha(samples[s]['fillcolor'],0.2)
        g_met_pt_1LLP[m][ct].SetLineWidth(3)
        met_pt_1LLP[m][ct].Draw("HIST,sames")
        g_met_pt_1LLP[m][ct].Draw("PE,sames")
        #g_met_pt_met200_1LLP[m][ct].Draw("PE,sames")
        leg.AddEntry(g_met_pt_1LLP[m][ct],samples[s]['label'],"P")
        leg.AddEntry(met_pt_1LLP[m][ct],"True reco MET","L")

    leg.Draw("sames")
    can.SetLogy()
    drawCMS_simple(-1, "Simulation", onTop=True)
    can.Print('GenMet_compare_m'+str(m)+'.png')
    can.Print('GenMet_compare_m'+str(m)+'.pdf')
    can.Close()



for ct in ctaus:
    leg = TLegend(0.5, 0.6, 0.99, 0.99)
    leg.SetTextSize(0.035)
    can = TCanvas("can","can",1000,800)
    can.cd()
    for m in masses:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        met_pt_1LLP[m][ct].SetLineColor(samples[s]['linecolor'])
        met_pt_1LLP[m][ct].SetLineStyle(samples[s]['linestyle'])
        met_pt_1LLP[m][ct].SetLineWidth(3)
        #g_met_pt_met200_1LLP[m][ct].SetLineColor(samples[s]['linecolor'])
        #g_met_pt_met200_1LLP[m][ct].SetLineStyle(samples[s]['linestyle'])
        #g_met_pt_met200_1LLP[m][ct].SetMarkerColor(samples[s]['linecolor'])
        #g_met_pt_met200_1LLP[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        #g_met_pt_met200_1LLP[m][ct].SetMarkerSize(1.2)
        #g_met_pt_met200_1LLP[m][ct].SetLineWidth(3)
        g_met_pt_1LLP[m][ct].SetMinimum(0.1)
        met_pt_1LLP[m][ct].SetMinimum(0.1)
        g_met_pt_1LLP[m][ct].GetXaxis().SetTitle("Reconstructed Gen MET [gravitinos + 1 displaced Higgs] (GeV)")
        met_pt_1LLP[m][ct].GetXaxis().SetTitle("Reconstructed Gen MET [gravitinos + 1 displaced Higgs] (GeV)")
        g_met_pt_1LLP[m][ct].SetLineColor(samples[s]['linecolor'])
        g_met_pt_1LLP[m][ct].SetMarkerColor(samples[s]['linecolor'])
        g_met_pt_1LLP[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        g_met_pt_1LLP[m][ct].SetMarkerSize(1.2)
        g_met_pt_1LLP[m][ct].SetLineStyle(samples[s]['linestyle'])
        #met_pt_1LLP[m][ct].SetFillColorAlpha(samples[s]['fillcolor'],0.2)
        g_met_pt_1LLP[m][ct].SetLineWidth(3)
        met_pt_1LLP[m][ct].Draw("HIST,sames")
        g_met_pt_1LLP[m][ct].Draw("PE,sames")
        #g_met_pt_met200_1LLP[m][ct].Draw("PE,sames")
        leg.AddEntry(g_met_pt_1LLP[m][ct],samples[s]['label'],"P")
        leg.AddEntry(met_pt_1LLP[m][ct],"True reco MET","L")

    leg.Draw("sames")
    can.SetLogy()
    drawCMS_simple(-1, "Simulation", onTop=True)
    can.Print('GenMet_compare_ct'+str(ct)+'.png')
    can.Print('GenMet_compare_ct'+str(ct)+'.pdf')
    can.Close()



leg = TLegend(0.5, 0.6, 0.99, 0.99)
leg.SetTextSize(0.035)
can = TCanvas("can","can",1000,800)
can.cd()

for m in masses:
    for ct in ctaus:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        met_pt_1LLP[m][ct].SetLineColor(samples[s]['linecolor'])
        met_pt_1LLP[m][ct].SetLineStyle(samples[s]['linestyle'])
        met_pt_1LLP[m][ct].SetLineWidth(3)
        met_pt_1LLP[m][ct].SetMinimum(0.1)
        met_pt_1LLP[m][ct].GetXaxis().SetTitle("True reco MET (GeV)")
        met_pt_1LLP[m][ct].Draw("HIST,sames")
        leg.AddEntry(met_pt_1LLP[m][ct],samples[s]['label'],"L")

leg.Draw("sames")
can.SetLogy()
drawCMS_simple(-1, "Simulation", onTop=True)
can.Print('TrueRecoMet_compare.png')
can.Print('TrueRecoMet_compare.pdf')
can.Close()



leg = TLegend(0.5, 0.6, 0.99, 0.99)
leg.SetTextSize(0.035)
can = TCanvas("can","can",1000,800)
can.cd()

for m in masses:
    for ct in ctaus:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        g_met_pt_1LLP[m][ct].SetMinimum(0.1)
        g_met_pt_1LLP[m][ct].GetXaxis().SetTitle("Reconstructed Gen MET [gravitinos + 1 displaced Higgs] (GeV)")
        g_met_pt_1LLP[m][ct].SetLineColor(samples[s]['linecolor'])
        g_met_pt_1LLP[m][ct].SetMarkerColor(samples[s]['linecolor'])
        g_met_pt_1LLP[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        g_met_pt_1LLP[m][ct].SetMarkerSize(1.2)
        g_met_pt_1LLP[m][ct].SetLineStyle(samples[s]['linestyle'])
        g_met_pt_1LLP[m][ct].SetLineWidth(3)
        g_met_pt_1LLP[m][ct].Draw("PE,sames")
        leg.AddEntry(g_met_pt_1LLP[m][ct],samples[s]['label'],"PL")

leg.Draw("sames")
can.SetLogy()
drawCMS_simple(-1, "Simulation", onTop=True)
can.Print('GenRecoMet_compare.png')
can.Print('GenRecoMet_compare.pdf')
can.Close()


leg = TLegend(0.5, 0.6, 0.99, 0.99)
leg.SetTextSize(0.035)
can = TCanvas("can","can",1000,800)
can.cd()

for m in masses:
    for ct in ctaus:
        s = "SUSY_mh"+str(m)+"_ctau"+str(ct)+"_HH"
        g_met_pt_met200_1LLP[m][ct].SetMinimum(0.1)
        g_met_pt_met200_1LLP[m][ct].GetXaxis().SetTitle("Reconstructed Gen MET [gravitinos + 1 displaced Higgs] (GeV), MET>200 GeV")
        g_met_pt_met200_1LLP[m][ct].SetLineColor(samples[s]['linecolor'])
        g_met_pt_met200_1LLP[m][ct].SetMarkerColor(samples[s]['linecolor'])
        g_met_pt_met200_1LLP[m][ct].SetMarkerStyle(20 if ct==500 else 25)
        g_met_pt_met200_1LLP[m][ct].SetMarkerSize(1.2)
        g_met_pt_met200_1LLP[m][ct].SetLineStyle(samples[s]['linestyle'])
        g_met_pt_met200_1LLP[m][ct].SetLineWidth(3)
        g_met_pt_met200_1LLP[m][ct].Draw("PE,sames")
        leg.AddEntry(g_met_pt_1LLP[m][ct],samples[s]['label'],"PL")

leg.Draw("sames")
can.SetLogy()
drawCMS_simple(-1, "Simulation", onTop=True)
can.Print('GenRecoMet_met200_compare.png')
can.Print('GenRecoMet_met200_compare.pdf')
can.Close()


exit()

jet.SetLineColor(1)
jet_b.SetLineColor(8)
f_jet.SetLineColor(4)
f_jet_b.SetLineColor(2)

jet.SetLineWidth(3)
jet_b.SetLineWidth(3)
f_jet.SetLineWidth(3)
f_jet_b.SetLineWidth(3)

jet.SetLineStyle(2)
f_jet.SetLineStyle(2)
jet.SetTitle("")
jet_b.SetTitle("")
f_jet.SetTitle("")
f_jet_b.SetTitle("")

jet.GetXaxis().SetTitle("min #Delta #varphi (jet,MET)")
jet_b.GetXaxis().SetTitle("min #Delta #varphi (jet,MET)")
f_jet.GetXaxis().SetTitle("min #Delta #varphi (jet,MET)")
f_jet_b.GetXaxis().SetTitle("min #Delta #varphi (jet,MET)")

leg = TLegend(0.3, 0.8, .6, 1.)
leg.SetHeader(sampl)
leg.AddEntry(jet,"AK4","L")
leg.AddEntry(jet_b,"AK4 barrel","L")
leg.AddEntry(f_jet,"AK8","L")
leg.AddEntry(f_jet_b,"AK8 barrel","L")

if sampl=="QCD":
    jet.Draw("HISTO")
f_jet.Draw("HISTO,same")
f_jet_b.Draw("HISTO,same")
f_jet.Draw("HISTO,same")
jet_b.Draw("HISTO,same")
jet.Draw("HISTO,same")
leg.Draw()
