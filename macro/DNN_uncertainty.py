#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import uproot
import root_numpy
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TMultiGraph, TH3F, TF1, TRatioPlot
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare
import yaml
from collections import defaultdict, OrderedDict
import tensorflow as tf
from tensorflow import keras

#import awkward1 as ak
#import matplotlib.pyplot as plt

ERA = "2018"

CHAN = "SUSY"
LUMI = -1

if ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI = lumi[ lumi.keys()[0]]["tot"]    
if ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI = lumi[ lumi.keys()[0]]["tot"]    
if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI = lumi[ lumi.keys()[0]]["tot"]    

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *

#import optparse
#usage = "usage: %prog [options]"
#parser = optparse.OptionParser(usage)
#parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
#(options, args) = parser.parse_args()
gROOT.SetBatch(True)
gStyle.SetOptStat(0000)

#SEL = "TtoEM"
#SEL = "ZtoMM"
SEL = "SR"

if SEL=="ZtoMM":
    data = ["SingleMuon"]
    back = ["DYJetsToLL"]
if SEL=="ZtoEE":
    data = ["SingleElectron"]
    back = ["DYJetsToLL"]
if SEL=="TtoEM":
    data = ["MuonEG"]
    back = ["TTbarGenMET"]
if SEL=="SR":
    data = ["HighMET"]#["QCD"]
    back = ["ZJetsToNuNu"]
sign = [
    'SUSY_mh127_ctau500',
    'SUSY_mh150_ctau500',
    'SUSY_mh175_ctau500',
    'SUSY_mh200_ctau500',
    'SUSY_mh250_ctau500',
    'SUSY_mh300_ctau500',
    'SUSY_mh400_ctau500',
    'SUSY_mh600_ctau500',
    'SUSY_mh800_ctau500',
    'SUSY_mh1000_ctau500',
    'SUSY_mh1250_ctau500',
    'SUSY_mh1500_ctau500',
    'SUSY_mh1800_ctau500'
]


sign += [
    'SUSY_mh127_ctau3000',
    'SUSY_mh150_ctau3000',
    'SUSY_mh175_ctau3000',
    'SUSY_mh200_ctau3000',
    'SUSY_mh250_ctau3000',
    'SUSY_mh300_ctau3000',
    'SUSY_mh400_ctau3000',
    'SUSY_mh600_ctau3000',
    'SUSY_mh800_ctau3000',
    'SUSY_mh1000_ctau3000',
    'SUSY_mh1250_ctau3000',
    'SUSY_mh1500_ctau3000',
    'SUSY_mh1800_ctau3000'
]

#sign = ['SUSY_mh300_ctau500','SUSY_mh600_ctau500','SUSY_mh1000_ctau500']

MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_"+SEL+"_DNN_uncertainty/"
SF = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/data_MC_SF_DNN_"+ERA+"/"
SYST_UNC_DIR = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_SR_signal_uncertainties/"

def deltaPhi_np( a1, v2):
    a2 = np.multiply(np.ones(a1.size),v2)
    pi = np.multiply(np.ones(a1.size),math.pi)
    diff = np.abs(a1 - a2)
    mask = diff>pi
    M = np.ma.masked_array(data=diff, mask = mask)
    new = diff - 2*math.pi*np.ones(diff.size)
    M.filled(new)
    return np.array(M)

    '''
    timeData = np.array([])
    WeightData = np.array([])
    timeBack = np.array([])
    WeightBack = np.array([])
    timeSign = np.array([])
    WeightSign = np.array([])

    for d in data:
        print "\n"
        print "\n"
        print "opening: ", MAIN + d + ".root"
        files_list = samples[d]["files"]
        new_lista = []
        gen_events = 0
        for l in files_list:
            new_lista.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        print new_lista
        gen = uproot.iterate(new_lista,"tree",list_of_variables)

        for arrays in gen:
            cut_mask = arrays["isTtoEM"]>0
            cut_mask = np.logical_and(cut_mask,arrays["pt"]>200)

            cut_mask_csv = arrays["JetsNegative.CSV"]>=0.8
            cut_mask_csv = np.logical_and(cut_mask,cut_mask_csv)
            cut_mask = (cut_mask_csv.any()==True)
            cut_jets = cut_mask_csv

            timeData = np.concatenate(( timeData,arrays["JetsNegative.timeRecHitsEB"][cut_jets][cut_mask] ))

        #print "gen_events: ", gen_events
        #print "events passing: ", RunNumber.size
        #print ("%d (%.2f%s) ") % (RunNumber.size, 100.*float(RunNumber.size/gen_events),'\%')

    '''

def sign_unc(label="",scale=True, do_smear=True,added=""):
    list_of_variables = ["EventNumber","LumiNumber","RunNumber","Jets*.pt","Jets*.eta","Jets*.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight","TriggerWeight","PUWeight","PUReWeight","Jets*.isGenMatchedCaloCorrLLPAccept","Jets*.CSV","Jets*.timeRecHitsEB","is*","JetsNegative*.CSV","JetsNegative*.sigprob","JetsNegative*.timeRecHitsEB"]

    prj_weight = "EventWeight*PUReWeight*TriggerWeight"
    results = defaultdict(dict)


    print "Reading SF in: "+SF+"data_MC_SF"+label+".root"
    infile = TFile(SF+"data_MC_SF"+label+".root","READ")
    ratio_1ns = infile.Get("ratio_1ns")
    ratio_2ns = infile.Get("ratio_2ns")
    ratio_1ns.SetDirectory(0)
    ratio_2ns.SetDirectory(0)

    for s in sign:
        #event-wise variables
        bin_2 = []
        bin_1 = []
        bin_2_shift = []
        bin_1_shift = []

        for j, ss in enumerate(samples[s]['files']):
            print "Performing computation for ", ss
            tmp_file = TFile(MAIN+ss+".root","READ")
            tree = tmp_file.Get("tree")
            tree_weight = tree.GetWeight()
            for e in range(0,tree.GetEntries()):
                tree.GetEntry(e)
                jets = getattr(tree,"Jets")
                met = getattr(tree,"MEt")
                nTagJets = 0

                #Event level selections
                if tree.MinJetMetDPhi<=0.5:
                    continue

                #print "---------"
                #print "event ", e
                #print "nj: ", neg_jets.size()
                shift_2_w = 1.
                shift_1_w = 1.
                shift_w = 1.
                for n in range(jets.size()):
                    if jets[n].eta<=-1:
                        continue
                    if jets[n].eta>=1:
                        continue
                    if ERA=="2017":
                        if jets[n].phi>=2.7: continue
                    if ERA=="2018":
                        if jets[n].phi>=0.4 and jets[n].phi<0.9: continue

                    if jets[n].sigprob>0.996:
                        nTagJets+=1
                        eff_1ns = ratio_1ns.GetBinContent( ratio_1ns.FindBin(jets[n].eta) )
                        eff_2ns = ratio_2ns.GetBinContent( ratio_2ns.FindBin(jets[n].eta) )
                        #print "current n tags ", nTagJets
                        #print jets[n].eta, eff_1ns, eff_2ns
                        #print "max SF: ", max(abs(1-eff_1ns),abs(1-eff_2ns))+1
                        print "PROBABLY WRONG!"
                        shift_w*= (max(abs(1-eff_1ns),abs(1-eff_2ns))+1)
                        #Here: weight due to DNN SFs
                        
                #event-wise variables
                #print "nTagJets: ", nTagJets
                #print "shift_w: ", shift_w
                ev_weight = tree.EventWeight * tree.PUReWeight * tree_weight * tree.TriggerWeight
                if nTagJets==1:
                    bin_1.append(ev_weight)
                    bin_1_shift.append(ev_weight * shift_w)
                if nTagJets>1:
                    bin_2.append(ev_weight)
                    bin_2_shift.append(ev_weight * shift_w)
    
        #Here calculate differences with and without SFs; check also in all eta range
        #print "bin_2"
        #print bin_2
        #print "bin_2_shift"
        #print bin_2_shift
        y1 = np.sum(np.array(bin_1))
        y1_shift = np.sum(np.array(bin_1_shift))
        y2 = np.sum(np.array(bin_2))
        y2_shift = np.sum(np.array(bin_2_shift))
        print "y2: ", y2, " ; y2 shift: ", y2_shift, " ; diff ", 100*abs(y2_shift-y2)/y2
        print "y1: ", y1, " ; y1 shift: ", y1_shift, " ; diff ", 100*abs(y1_shift-y1)/y1
        results[s]['y2'] = y2
        results[s]['y2_shift'] = y2_shift
        results[s]['diff_y2'] = 100*abs(y2_shift-y2)/y2
        results[s]['y1'] = y1
        results[s]['y1_shift'] = y1_shift
        results[s]['diff_y1'] = 100*abs(y1_shift-y1)/y1
        #....
        #fill results

    print results
    with open(OUT+"signal_DNN_unc"+label+".yaml","w") as f:
        yaml.dump(results, f)
        f.close()
    print "Written in ", OUT+"signal_DNN_unc"+label+".yaml"

    with open(SYST_UNC_DIR+"signal_DNN_unc"+label+".yaml","w") as f:
        yaml.dump(results, f)
        f.close()
    print "Written in ", SYST_UNC_DIR+"signal_DNN_unc"+label+".yaml"



def draw_syst_unc(added):
    print "Drawing uncertainty vs mass"
    with open(OUT+"signal_DNN_unc"+added+".yaml","r") as f:
        results = yaml.load(f, Loader=yaml.Loader)
        f.close()
    samp = results.keys()
    masses = []
    ctaus = []
    for s in samp:
        masses.append(samples[s]['mass'])
        ctaus.append(samples[s]['ctau'])
    masses = np.unique(np.sort(np.array(masses)))
    ctaus = np.unique(np.sort(np.array(ctaus)))
    print masses
    print ctaus

    mg = TMultiGraph()
    leg1 = TLegend(0.15+0.5, 0.6+0.2, 0.3+0.5, 0.9)
    colors = [2,418,801,856,602,920,881]
    diff = {}
    count_c = 0
    print "$c \\tau$ (m) & $m_{\chi}$ (GeV) & uncertainty (\%)"+"\\"+"\\"
    for m in masses:
        string = ""
        nt = 0
        for c in ctaus:
            for s in samp:
                if "mh"+str(m)+"_ctau"+str(c) in s:
                    string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % results[s]['diff_y2'])
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"

    for c in ctaus:
        diff[c] = TGraph()
        n = 0
        for m in masses:
            for s in samp:
                if "mh"+str(m)+"_ctau"+str(c) in s:
                    diff[c].SetPoint(n,m,results[s]['diff_y2'])
                    n+=1
        diff[c].SetMarkerStyle(24)
        diff[c].SetMarkerColor(colors[count_c])
        diff[c].SetLineColor(colors[count_c])
        diff[c].SetLineWidth(2)
        diff[c].GetXaxis().SetTitle("m_{\chi} (GeV)")
        diff[c].GetYaxis().SetTitle("Uncertainty (%)")
        diff[c].GetYaxis().SetTitleSize(0.05)
        diff[c].SetMinimum(0)
        leg1.AddEntry(diff[c],"c_{#tau} = "+str(c/1000.)+" m","F")
        mg.Add(diff[c])
        count_c+=1

    c1 = TCanvas("c1", "c1", 800, 600)
    c1.SetTopMargin(0.06)
    c1.SetBottomMargin(0.12)
    c1.SetRightMargin(0.05)
    c1.SetLeftMargin(0.12)
    c1.SetTicks(1, 1)
    mg.GetXaxis().SetTitle("m_{\chi} (GeV)")
    mg.GetYaxis().SetTitle("Uncertainty (%)")
    mg.GetXaxis().SetTitleSize(0.05)
    mg.GetYaxis().SetTitleSize(0.05)
    mg.Draw("APL")
    leg1.Draw()
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    #drawAnalysis("LL"+CHAN)
    drawRegion(SEL)
    #c1.SetLogx()
    c1.Print(OUT+"signal_DNN_uncertainty"+added+".pdf")
    c1.Print(OUT+"signal_DNN_uncertainty"+added+".png")
    c1.Close()
 
lab = "_not_scaled"#"_constant_1"
lab = ""
if SEL=="SR":
    added = ""
    #added="_G-H"
    #added="_B-F"
    sign_unc(label=added)
    draw_syst_unc(added)
