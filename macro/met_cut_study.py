#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import uproot
import root_numpy
import numpy as np
import yaml
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot, TMultiGraph
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare
#import matplotlib.pyplot as plt
from collections import defaultdict

from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples, sample
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
#from Analyzer.LLP2017.drawUtils import *

#import optparse
#usage = "usage: %prog [options]"
#parser = optparse.OptionParser(usage)
#parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
#(options, args) = parser.parse_args()
gROOT.SetBatch(True)
gStyle.SetOptStat(0000)

data = [
    ##central
    'SUSY_mh127_ctau500_HH', 'SUSY_mh127_ctau3000_HH',
    'SUSY_mh150_ctau500_HH', 'SUSY_mh150_ctau3000_HH',
    'SUSY_mh175_ctau500_HH', 'SUSY_mh175_ctau3000_HH',
    'SUSY_mh200_ctau500_HH', 'SUSY_mh200_ctau3000_HH',
    'SUSY_mh250_ctau500_HH', 'SUSY_mh250_ctau3000_HH',
    'SUSY_mh300_ctau500_HH', 'SUSY_mh300_ctau3000_HH',
    'SUSY_mh400_ctau500_HH', 'SUSY_mh400_ctau3000_HH',
    'SUSY_mh600_ctau500_HH', 'SUSY_mh600_ctau3000_HH',
    'SUSY_mh800_ctau500_HH', 'SUSY_mh800_ctau3000_HH',
    'SUSY_mh1000_ctau500_HH','SUSY_mh1000_ctau3000_HH',
    'SUSY_mh1250_ctau500_HH','SUSY_mh1250_ctau3000_HH',
    'SUSY_mh1500_ctau500_HH','SUSY_mh1500_ctau3000_HH',
    'SUSY_mh1800_ctau500_HH','SUSY_mh1800_ctau3000_HH',
]
MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_Gen/"
OUT = "plots/v6_calo_AOD_2017_Gen/"

def count_gen_central(force_HH=True):
    print "\n"
    print "Ntuples: ", MAIN
    print "\n"
    for d in data:
        for j, ss in enumerate(samples[d]['files']):
            tmp_file = TFile(MAIN+ss+".root","READ")
            tree = tmp_file.Get("tree")
            nentries = tree.GetEntries()
        if "_HZ" not in d and "_ZZ" not in d and force_HH:
            print "    '"+samples[d]['files'][0]+"_HH' : {"
        else:
            print "    '"+samples[d]['files'][0]+"' : {"
        print "        'nevents' : "+str(nentries)+","
        if "_HZ" in d:
            print "        'xsec'    : "+str( sample[ samples[d]['files'][0].replace("_HZ","_HH")  ]['xsec'] )+","
            print "        'BR'      : "+ "0.5824*0.69911"  +","
        elif "_ZZ" in d:
            print "        'xsec'    : "+str( sample[ samples[d]['files'][0].replace("_ZZ","_HH")  ]['xsec'] )+","
            print "        'BR'      : "+"0.69911*0.69911"+","
        else:
            print "        'xsec'    : "+str( sample[ samples[d]['files'][0]+"_HH" ]['xsec'] / (0.5824*0.5824))+","#remove HH BR
            print "        'BR'      : "+"0.5824*0.5824"+","
        print "        'matcheff': 1.,"
        print "        'kfactor' : 1.,"
        print "    },"
        print "\n"

def count(label,do_eta=False):
    eff = defaultdict(dict)
    for d in data:
        list_of_variables = ["nTagJets_0p996_JJ","EventNumber","LumiNumber","RunNumber",
                             "isDT_fit","dt_ecal_dist","dt_fit_chi2_reduced",
                             "Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight",
                             "Jets.timeRecHitsEB","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB",
                             "TriggerWeight","PUWeight",
                             "PUReWeight",
                             "Jets.isGenMatchedCaloCorrLLPAccept","isSR","isWtoMN","isZtoMM","isMC",
                             "Z_pt",
                             "pt",
                             "min_dPhi_jets_*",
                         ]
        print "\n"
        print "\n"
        print "opening: ", MAIN
        list_files = samples[d]["files"]
        if "SUSY" in d:
            m = samples[d]["mass"]
            eff[d]['m'] = m
            c = samples[d]["ctau"]
            eff[d]['c'] = c
            print m, c
        new_list = []
        gen_events = 0
        for l in list_files:
            new_list.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        if "SMS" in samples[d]["files"][0]:
            name = samples[d]["files"][0]
            gen_events = sample[ name ]['nevents']
        eff[d]['gen'] = gen_events
        print new_list
        gen = uproot.iterate(new_list,"tree",list_of_variables)
        pr = np.array([])
        post = np.array([])

        for arrays in gen:
            ################################################################
            #Study effect of vetos
            cut_mask = arrays["isMC"]>0
            pr = np.concatenate((pr, np.multiply(cut_mask[cut_mask],1.) ))

            cut_mask = np.logical_and(cut_mask, arrays["pt"]>200.)
            post = np.concatenate((post,np.multiply(cut_mask,1.) ))


        print "gen_events: ", gen_events
        print "events passing preselections: ", pr.sum()
        print ("%d (%.2f%s) ") % (pr.sum(), 100.*float(pr.sum()/gen_events),'\%')

        
        print "************************************"
        print "events passing met cut: ", post.sum()
        print ("%d (%.2f%s) ") % (post.sum(), 100.*float(post.sum()/gen_events),'\%')
        
        eff[d]['pr'] = pr.sum()
        eff[d]['post'] = post.sum()

    print eff
    with open(MAIN+"Dict"+label+".yaml","w") as f:
        yaml.dump(eff, f)
        f.close()
        print "Info: dictionary written in file "+MAIN+"Dict"+label+".yaml"

def draw(label):
    with open(MAIN+"Dict"+label+".yaml","r") as f:
        print "\n"
        eff = yaml.load(f, Loader=yaml.Loader)
        f.close()
    #print eff
    #print eff.keys()
    #print data
    mass = []#np.array([])
    ctau = []#np.array([])

    eff_pr_m = defaultdict(dict)
    eff_pr_c = defaultdict(dict)
    ratio = defaultdict(dict)
    eff_b2_pre_m = defaultdict(dict)
    eff_b2_pre_c = defaultdict(dict)
    eff_b2_post_m = defaultdict(dict)
    eff_b2_post_c = defaultdict(dict)
    eff_b1_pre_m = defaultdict(dict)
    eff_b1_pre_c = defaultdict(dict)
    eff_b1_post_m = defaultdict(dict)
    eff_b1_post_c = defaultdict(dict)

    for d in data:
        mass.append(eff[d]['m'])
        ctau.append(eff[d]['c'])

    mass = np.unique(np.array(mass))
    ctau = np.unique(np.array(ctau))

    colors = [1,2,418,801,856,602,920,881]
    colors = [2,418,801,856,602,920,881]
        
    count_c = 0

    c1 = TCanvas("c1", "c1", 800, 600)
    c1.SetTopMargin(0.06)
    c1.SetRightMargin(0.05)
    c1.SetTicks(1, 1)

    mg = TMultiGraph()
    mg_ratio = TMultiGraph()
    leg1 = TLegend(0.1, 0.68, 0.3, 0.94)

    for c in ctau:
        eff_pr_m[c] = TGraph()
        eff_b2_pre_m[c] = TGraph()
        eff_b2_post_m[c] = TGraph()
        eff_b1_pre_m[c] = TGraph()
        eff_b1_post_m[c] = TGraph()
        ratio[c] = TGraph()
        n = 0
        for m in mass:
            for d in data:
                if eff[d]['m']==m and eff[d]['c']==c:                    
                    print m, c, d, eff[d]['pr']
                    print "point n. ", n
                    eff_pr_m[c].SetPoint(n,m,100*float(eff[d]['pr']/eff[d]['gen']))
                    eff_b2_pre_m[c].SetPoint(n,m,100*float(eff[d]['b2_pre']/eff[d]['gen']))
                    eff_b2_post_m[c].SetPoint(n,m,100*float(eff[d]['b2_post']/eff[d]['gen']))
                    eff_b1_pre_m[c].SetPoint(n,m,100*float(eff[d]['b1_pre']/eff[d]['gen']))
                    eff_b1_post_m[c].SetPoint(n,m,100*float(eff[d]['b1_post']/eff[d]['gen']))
                    ratio[c].SetPoint(n,m,float(eff[d]['b2_post']/eff[d]['b2_pre']))
                    n+=1
        
        print c
        print eff_pr_m[c].Print()
        print eff_b2_pre_m[c].Print()
        print eff_b2_post_m[c].Print()
        eff_pr_m[c].SetLineColor(colors[count_c])
        eff_b2_pre_m[c].SetLineColor(colors[count_c])
        eff_b2_post_m[c].SetLineColor(colors[count_c])
        eff_b1_pre_m[c].SetLineColor(colors[count_c])
        eff_b1_post_m[c].SetLineColor(colors[count_c])
        eff_pr_m[c].SetLineWidth(3)
        eff_b2_pre_m[c].SetLineWidth(3)
        eff_b2_post_m[c].SetLineWidth(3)
        eff_b1_pre_m[c].SetLineWidth(1)
        eff_b1_post_m[c].SetLineWidth(1)
        eff_pr_m[c].SetLineStyle(1)
        eff_b2_pre_m[c].SetLineStyle(2)
        eff_b2_post_m[c].SetLineStyle(3)
        eff_b1_pre_m[c].SetLineStyle(2)
        eff_b1_post_m[c].SetLineStyle(2)
        eff_b1_pre_m[c].SetMarkerStyle(21)
        eff_b1_post_m[c].SetMarkerStyle(25)
        eff_b1_pre_m[c].SetMarkerColor(colors[count_c])
        eff_b1_post_m[c].SetMarkerColor(colors[count_c])

        leg1.AddEntry(eff_pr_m[c],"c_{#tau} = "+str(c/1000.)+" m","F")
        eff_b2_pre_m[c].GetXaxis().SetTitle("m_{\chi} (GeV)")
        eff_b2_pre_m[c].GetYaxis().SetTitle("Efficiency (%)")
        eff_b2_pre_m[c].GetYaxis().SetTitleSize(0.05)
        eff_b2_pre_m[c].SetMinimum(0)
        eff_b2_pre_m[c].SetMaximum(101)
        mg.Add(eff_pr_m[c])
        mg.Add(eff_b2_pre_m[c])
        #mg.Add(eff_b2_post_m[c])
        #mg.Add(eff_b1_pre_m[c])
        #mg.Add(eff_b1_post_m[c])

        ratio[c].SetMarkerStyle(20)
        ratio[c].SetMarkerColor(colors[count_c])
        ratio[c].GetXaxis().SetTitle("m_{\chi} (GeV)")
        ratio[c].GetXaxis().SetTitleOffset(1.2)
        ratio[c].GetXaxis().SetTitleSize(0.15)
        ratio[c].GetXaxis().SetLabelSize(0.15)
        ratio[c].GetYaxis().SetTitle("ratio")
        ratio[c].GetYaxis().SetTitleSize(0.15)
        ratio[c].GetYaxis().SetTitleOffset(1.)
        ratio[c].GetYaxis().SetLabelSize(0.1)
        #if count_c==0:
        #    ratio[c].Draw("AP")
        #else:
        #    ratio[c].Draw("P,sames")
        mg_ratio.Add(ratio[c])
        count_c+=1

    c1.cd(1)
    leg1.AddEntry(eff_pr_m[ctau[0]],"Presel.","L")
    leg1.AddEntry(eff_b2_pre_m[ctau[0]],"+ MinDPhi","L")
    #leg1.AddEntry(eff_b2_post_m[ctau[0]],"bin2 ","L")
    #leg1.AddEntry(eff_b1_pre_m[ctau[0]],"bin1","L")
    #leg1.AddEntry(eff_b1_post_m[ctau[0]],"bin1 + BH","L")
    mg.GetXaxis().SetTitle("m_{\chi} (GeV)")
    mg.GetYaxis().SetTitle("Efficiency (%)")
    mg.GetYaxis().SetTitleSize(0.05)
    mg.SetMinimum(0)
    mg.SetMaximum(101)
    #c1.SetLogy()
    c1.SetGrid()
    mg.Draw("AL")
    leg1.Draw()
    LUMI = 1.
    drawCMS_simple(LUMI, "Simulation", ERA="2017", onTop=True)
    '''
    c1.cd(2)
    mg_ratio.GetXaxis().SetTitle("m_{\chi} (GeV)")
    mg_ratio.GetXaxis().SetTitleOffset(1.2)
    mg_ratio.GetXaxis().SetTitleSize(0.15)
    mg_ratio.GetXaxis().SetLabelSize(0.15)
    mg_ratio.GetYaxis().SetTitle("ratio")
    mg_ratio.GetYaxis().SetTitleSize(0.15)
    mg_ratio.GetYaxis().SetTitleOffset(1.)
    mg_ratio.GetYaxis().SetLabelSize(0.1)
    mg_ratio.Draw("AP")
    '''
    #c1.Print(OUT+"SignEff_BeamHalo"+label+".pdf")
    #c1.Print(OUT+"SignEff_BeamHalo"+label+".png")
    c1.Print(OUT+"SignEff_MinDPhi"+label+".pdf")
    c1.Print(OUT+"SignEff_MinDPhi"+label+".png")
    c1.Clear()


    #Print table
    print "$c \\tau$ (m) & $m_{\chi}$ (GeV) & difference (\%)"+"\\"+"\\"
    for m in mass:
        string = ""
        nt = 0
        for c in ctau:
            for d in data:
                if "mh"+str(m)+"_ctau"+str(c) in d:
                    string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % ( 100*(eff[d]['b2_pre']-eff[d]['b2_post'])/ eff[d]['b2_pre'] ) )
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"


    '''
    count_m = 0
    for m in mass:
        eff_pr_c[m] = TGraph()
        eff_b2_pre_c[m] = TGraph()
        eff_b2_post_c[m] = TGraph()
        n = 0
        for c in ctau:
            for d in data:
                if eff[d]['m']==m and eff[d]['c']==c:                    
                    print m, c, d, eff[d]['pr']
                    print "point n. ", n
                    eff_pr_c[m].SetPoint(n,c,100*float(eff[d]['pr']/eff[d]['gen']))
                    eff_b2_pre_c[m].SetPoint(n,c,100*float(eff[d]['b2_pre']/eff[d]['gen']))
                    eff_b2_post_c[m].SetPoint(n,c,100*float(eff[d]['b2_post']/eff[d]['gen']))
                    n+=1
        print eff_pr_c[m].Print()
        print eff_b2_pre_c[m].Print()
        print eff_b2_post_c[m].Print()
        count_m+=1
    '''

def calc_eff(label,do_eta=False):
    eff = defaultdict(dict)
    for d in data:
        list_of_variables = ["nTagJets_0p996_JJ","EventNumber","LumiNumber","RunNumber",
                             #"nDTSegments","nCosmicMuons","nCosmicMuonsOneLeg","DTSegments.x","DTSegments.y","DTSegments.z","CSCSegments.x","CSCSegments.y","CSCSegments.z","DTSegments.time","CSCSegments.time","skimmedEcalRecHitsAK4.x","skimmedEcalRecHitsAK4.y","skimmedEcalRecHitsAK4.z","taggedEcalRecHitsAK4.x","taggedEcalRecHitsAK4.y","taggedEcalRecHitsAK4.z","DTSegments.eta","DTSegments.phi","CSCSegments.eta","CSCSegments.phi","taggedEcalRecHitsAK4.eta","taggedEcalRecHitsAK4.phi","skimmedEcalRecHitsAK4.eta","skimmedEcalRecHitsAK4.phi","min_dPhi_jets","min_dPhi_jets_0p996","JetsCaloAdd.eFracEB2CSC0p1","JetsCaloAdd.spreadXEB",
                             "isDT_fit","dt_ecal_dist","dt_fit_chi2_reduced",
                             #,"CosmicMuons.eta","CosmicMuons.phi","CosmicMuonsOneLeg.eta","CosmicMuonsOneLeg.phi"]#,"CosmicMuonsPropagated.eta","CosmicMuonsPropagated.phi","CosmicMuonsPropagatedOneLeg.eta","CosmicMuonsPropagatedOneLeg.phi"
                             "Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight",
                             "Jets.timeRecHitsEB","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB",
                             "TriggerWeight","PUWeight",
                             "PUReWeight",
                             "Jets.isGenMatchedCaloCorrLLPAccept","isSR","isWtoMN","isZtoMM","isMC",
                             "Z_pt",
                             "pt",
                             "min_dPhi_jets_*",
                             "n*Passing",
                         ]
        print "\n"
        print "\n"
        print "opening: ", MAIN
        list_files = samples[d]["files"]
        if "SUSY" in d:
            m = samples[d]["mass"]
            eff[d]['m'] = m
            c = samples[d]["ctau"]
            eff[d]['c'] = c
            print m, c
        new_list = []
        gen_events = 0
        for l in list_files:
            new_list.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        if "SMS" in samples[d]["files"][0]:
            name = samples[d]["files"][0]
            gen_events = sample[ name  ]['nevents']
        eff[d]['gen'] = gen_events
        print new_list
        gen = uproot.iterate(new_list,"tree",list_of_variables)
        isMC = np.array([])
        Weight = np.array([])
        RunNumber = np.array([])
        LumiNumber = np.array([])
        EventNumber = np.array([])

        eta = np.array([])
        nRecHitsEB = np.array([])
        timeRecHitsEB = np.array([])
        timeRMSRecHitsEB = np.array([])
        sel1 = np.array([])
        sel2 = np.array([])
        sel3 = np.array([])

        for arrays in gen:
            cut_mask = arrays["isMC"]>0
            cut_mask = np.logical_and(cut_mask, np.logical_or(arrays["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v"],np.logical_or(arrays["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v"],arrays["HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v"] )) )
            sel1 = np.concatenate((sel1, np.multiply(cut_mask[cut_mask],1.) ))
            cut_mask = np.logical_and(cut_mask, arrays["pt"]>200)
            sel2 = np.concatenate((sel2, np.multiply(cut_mask[cut_mask],1.) ))
            cut_mask = np.logical_and(cut_mask, np.logical_and( arrays["nMuonsPassing"]==0, np.logical_and(arrays["nElectronsPassing"]==0, np.logical_and(arrays["nPhotonsPassing"]==0, arrays["nTausPassing"]==0) ) ) )
            sel3 = np.concatenate((sel3,np.multiply(cut_mask[cut_mask],1.)))

        print "gen_events: ", gen_events
        print "************************************"
        print "events passing sel1: ", sel1.sum()
        print ("%d (%.2f%s) ") % (sel1.sum(), 100.*float(sel1.sum()/gen_events),'\%')        
        print "events passing sel2: ", sel2.sum()
        print ("%d (%.2f%s) ") % (sel2.sum(), 100.*float(sel2.sum()/gen_events),'\%')
        print "events passing sel3: ", sel3.sum()
        print ("%d (%.2f%s) ") % (sel3.sum(), 100.*float(sel3.sum()/gen_events),'\%')
        
        eff[d]['sel1'] = sel1.sum()
        eff[d]['sel2'] = sel2.sum()
        eff[d]['sel3'] = sel3.sum()

    print eff
    with open(MAIN+"Dict"+label+".yaml","w") as f:
        yaml.dump(eff, f)
        f.close()
        print "Info: dictionary written in file "+MAIN+"Dict"+label+".yaml"

def draw_eff(label,do_eta=False):
    with open(MAIN+"Dict"+label+".yaml","r") as f:
        print "\n"
        eff = yaml.load(f, Loader=yaml.Loader)
        f.close()

    mass = []#np.array([])
    ctau = []#np.array([])
    g_eff = {}

    for d in data:
        mass.append(eff[d]['m'])
        ctau.append(eff[d]['c'])

    mass = np.unique(np.array(mass))
    ctau = np.unique(np.array(ctau))

    colors = [418,2,856,800,602,920,881,1]        
    count_c = 0

    c1 = TCanvas("c1", "c1", 800, 600)
    #c1.Divide(1, 2)
    #setTopPad(c1.GetPad(1))
    #setBotPad(c1.GetPad(2))
    c1.cd(1)
    #c1.GetPad(bool(1)).SetTopMargin(0.06)
    #c1.GetPad(bool(1)).SetRightMargin(0.05)
    #c1.GetPad(bool(1)).SetTicks(1, 1)
    c1.SetTopMargin(0.1)
    c1.SetBottomMargin(0.125)
    c1.SetRightMargin(0.05)
    c1.SetLeftMargin(0.125)
    c1.SetTicks(1, 1)

    mg = TMultiGraph()
    leg1 = TLegend(0.15+0.4, 0.65-0.45, 0.45+0.4, 0.85-0.4)
    leg1.SetTextSize(0.035)
    leg1.SetBorderSize(0)
    for c in ctau:
        g_eff[c] = TGraph()
        n = 0
        for m in mass:
            for d in data:
                if eff[d]['m']==m and eff[d]['c']==c:                    
                    g_eff[c].SetPoint(n,m,100*float(eff[d]['post']/eff[d]['gen']))
                    n+=1
        
        g_eff[c].SetLineColor(colors[count_c])
        g_eff[c].SetMarkerColor(colors[count_c])
        g_eff[c].SetLineWidth(3)
        g_eff[c].SetLineStyle(1)
        g_eff[c].SetMarkerStyle(24)
        g_eff[c].SetMarkerSize(1.2)

        leg1.AddEntry(g_eff[c],"c_{#tau} = "+str(c/1000.)+" m","PL")
        g_eff[c].GetXaxis().SetTitle("m_{#chi} (GeV)")
        g_eff[c].GetXaxis().SetTitleSize(0.05)
        g_eff[c].GetYaxis().SetTitle("Efficiency (%)")
        g_eff[c].GetYaxis().SetTitleSize(0.05)
        g_eff[c].SetMinimum(0)
        mg.Add(g_eff[c])
        count_c+=1

    c1.cd(1)
    leg1.SetHeader("Offline MET > 200 GeV")
    mg.GetXaxis().SetTitle("m_{\chi} (GeV)")
    mg.GetYaxis().SetTitle("Efficiency (%)")
    mg.GetYaxis().SetTitleSize(0.05)
    mg.GetXaxis().SetTitleSize(0.05)
    mg.SetMinimum(1)
    mg.SetMaximum(101)
    mg.Draw("ALP")
    mg.GetXaxis().SetNoExponent(True)
    mg.GetXaxis().SetMoreLogLabels(True)
    LUMI = 0.
    drawCMS_simple(LUMI, "Simulation", ERA="2017", onTop=True)
    leg1.Draw()

    c1.SetLogx()
    c1.SetLogy()
    c1.Print(OUT+"GenEff"+label+".pdf")
    c1.Print(OUT+"GenEff"+label+".png")
    c1.Clear()


label = "_met200_cut"#"_MinDPhi0p5"
#count(label,do_eta=True)
draw_eff(label)
exit()
