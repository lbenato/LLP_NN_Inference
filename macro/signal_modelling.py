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

#Failed 2016:
#3000:

#Failed fits 2018:
#500:
#150, 250, 1000, 1250, 1500 not failed but horrible, ---> now okay!
#400, 800, 1800 not failed but horrible
#3000:
#175 --> works!
#250, --> works, trick was a negative alpha
#300, --> works, trick was a negative alpha
#left
#sign = ['SUSY_mh400_ctau500','SUSY_mh800_ctau500','SUSY_mh1800_ctau500']
#sign = ['SUSY_mh150_ctau500','SUSY_mh250_ctau500','SUSY_mh1000_ctau500','SUSY_mh1250_ctau500','SUSY_mh1500_ctau500']



#sign = ['SUSY_mh300_ctau500','SUSY_mh600_ctau500','SUSY_mh1000_ctau500']

MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_"+SEL+"_v5_ntuples_validate_timeRecHits/"
if SEL=="ZtoMM" or SEL=="ZtoEE":
    MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_"+SEL+"_v5_ntuples/"
if SEL=="SR":
    MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_"+SEL+"_v5_ntuples/"

MAIN_MET = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
MAIN_GEN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_Gen/"
OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_"+SEL+"_v5_ntuples_validate_timeRecHits/"
OUT_TtoEM = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_"+"TtoEM_v5_ntuples_validate_timeRecHits/"

SYST_UNC_DIR = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_SR_signal_uncertainties/"

'''
if SEL=="ZtoMMPho":
    data = ["SingleMuon"]
    back = ["DYJetsToLL"]
    MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2018_"+SEL+"_v5_ntuples/"
    OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_2018_"+SEL+"_v5_ntuples/"

if SEL=="SR":
    MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2018_"+SEL+"_v5_ntuples/"
    OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_2018_"+SEL+"_time_check/"
'''

def deltaPhi_np( a1, v2):
    a2 = np.multiply(np.ones(a1.size),v2)
    pi = np.multiply(np.ones(a1.size),math.pi)
    diff = np.abs(a1 - a2)
    mask = diff>pi
    M = np.ma.masked_array(data=diff, mask = mask)
    new = diff - 2*math.pi*np.ones(diff.size)
    M.filled(new)
    return np.array(M)

def time_fit(var,cut,label="",scale=True, do_smear=False):
    list_of_variables = ["EventNumber","LumiNumber","RunNumber","Jets*.pt","Jets*.eta","Jets*.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight","TriggerWeight","PUWeight","PUReWeight","Jets*.isGenMatchedCaloCorrLLPAccept","Jets*.CSV","Jets*.timeRecHitsEB","is*","JetsNegative*.CSV","JetsNegative*.sigprob","JetsNegative*.timeRecHitsEB"]

    #uproot: extreeeeemely slow
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

    chain = {}
    sigprob_chain = {}
    hist = {}
    weight = "EventWeight*PUReWeight*TriggerWeight"
    data_sigprob = TH1F("data_sigprob","data_sigprob",50,0,1)
    mc_sigprob = TH1F("mc_sigprob","mc_sigprob",50,0,1)
    mc_pass_sigprob = TH1F("mc_pass_sigprob","mc_pass_sigprob",50,0,1)
    mc_smear_sigprob = TH1F("mc_smear_sigprob","mc_smear_sigprob",50,0,1)
    data_sigprob.Sumw2()
    mc_sigprob.Sumw2()
    mc_pass_sigprob.Sumw2()
    mc_smear_sigprob.Sumw2()

    for s in back+data:
        print "Doing ", s, " .... "
        hist[s] = TH1F(s,";"+variable[var]['title'],variable[var]['nbins'], variable[var]['min'],variable[var]['max'])
        hist[s].Sumw2()
        chain[s] = TChain("tree")
        sigprob_chain[s] = TChain("tree")
        for j, ss in enumerate(samples[s]['files']):
            if "HighMET" in ss:
                print MAIN_MET+ss + ".root"
                chain[s].Add(MAIN_MET + ss + ".root")
                sigprob_chain[s].Add(MAIN_MET + ss + ".root")
            else:
                print MAIN+ss + ".root"
                chain[s].Add(MAIN + ss + ".root")
                sigprob_chain[s].Add(MAIN + ss + ".root")
        chain[s].Project(s, var, weight+"*("+cut+")")
        if s in back:
            hist[s].SetMarkerStyle(21)
            hist[s].SetMarkerColor(samples[s]['linecolor'])
            hist[s].SetLineColor(samples[s]['linecolor'])
            hist[s].SetFillColor(samples[s]['linecolor'])
            hist[s].SetFillColorAlpha(samples[s]['linecolor'],0.5)
            hist[s].SetLineWidth(2)
            if "[0]" in var:
                sigprob_chain[s].Project("mc_sigprob", "JetsNegative[0].sigprob", weight+"*("+cut+")")
            else:
                sigprob_chain[s].Project("mc_sigprob", "JetsNegative.sigprob", weight+"*("+cut+")")
        if s in data:
            hist[s].SetMarkerColor(1)
            hist[s].SetLineColor(1)
            hist[s].SetLineWidth(2)
            hist[s].SetMarkerStyle(21)
            if "[0]" in var:
                sigprob_chain[s].Project("data_sigprob", "JetsNegative[0].sigprob", weight+"*("+cut+")")
            else:
                sigprob_chain[s].Project("data_sigprob", "JetsNegative.sigprob", weight+"*("+cut+")")
        if s in sign:
            hist[s].SetLineColor(samples[s]['linecolor'])
            hist[s].SetLineWidth(2)
        if scale:
            hist[s].Scale(1./hist[s].Integral())

    print "\n"
    print " --- Fit ---"
    print "\n"

    #Fit section
    #CB
    fit_data = TF1("fit_data","crystalball",-5,5)
    #fit_data = TF1("fit_data","gaus",-5,5)
    
    if scale:
        if SEL=="TtoEM" or SEL=="SR":
            if ERA=="2016":
                fit_data.SetParameter(0,0.1)
                fit_data.SetParameter(1,0)
                fit_data.SetParameter(2,0.5)
                fit_data.SetParameter(3,1)
                fit_data.SetParameter(4,1)
            if ERA=="2017":
                fit_data.SetParameter(0,0.1)
                fit_data.SetParameter(1,0)
                fit_data.SetParameter(2,0.5)
                fit_data.SetParameter(3,1)
                fit_data.SetParameter(4,1)
            if ERA=="2018":
                #Good for RunD
                fit_data.SetParameter(0,0.1)
                fit_data.SetParameter(1,-0.6)
                fit_data.SetParameter(2,0.5)
                fit_data.SetParameter(3,2)
                fit_data.SetParameter(4,2)
        elif SEL=="ZtoMM" or SEL=="ZtoEE" or SEL=="ZtoMMPho":
            fit_data.SetParameter(0,0.1)
            fit_data.SetParameter(1,0)
            fit_data.SetParameter(2,0.5)
            fit_data.SetParameter(3,1)
            fit_data.SetParameter(4,1)

            #fit_data.SetParameter(0,0.08)
            ##fit_data.FixParameter(0,0.08)
            #fit_data.SetParameter(1,0.04)
            #fit_data.SetParameter(2,0.8)
            ##fit_data.FixParameter(2,0.8)
            #fit_data.SetParameter(3,1.5)
            ##fit_data.FixParameter(3,1.5)
            #fit_data.SetParameter(4,15)
            ##fit_data.FixParameter(4,15)
    else:
        fit_data.SetParameter(0,7)
        fit_data.SetParameter(1,0)
        fit_data.SetParameter(2,0.5)
        fit_data.SetParameter(3,1)
        fit_data.SetParameter(4,9)
    
    #Gaus
    #fit_data = TF1("fit_data","gaus")
    hist[data[0]].Fit(fit_data,"E")#R fixes range
    #my_data = TF1("my_data","gaus")
    #my_data.SetParameter(0,fit_data.GetParameter(0))
    #my_data.SetParameter(1,fit_data.GetParameter(1))
    #my_data.SetParameter(2,fit_data.GetParameter(2))
    my_data = hist[data[0]].GetFunction("fit_data")
    hist[data[0]].GetListOfFunctions().Remove(hist[data[0]].GetFunction("fit_data"))

    #CB
    fit_back = TF1("fit_back","crystalball",5,5)
    #fit_back = TF1("fit_back","gaus",5,5)
    if scale:
        if SEL=="TtoEM" or SEL=="SR":
            fit_back.SetParameter(0,0.01)
            fit_back.SetParameter(1,0.02)
            fit_back.SetParameter(2,0.31)
            fit_back.SetParameter(3,2)
            fit_back.SetParameter(4,5)
        elif SEL=="ZtoMM" or SEL=="ZtoEE" or SEL=="ZtoMMPho":
            fit_back.SetParameter(0,0.08)
            #fit_back.FixParameter(0,0.085)
            fit_back.SetParameter(1,0)
            fit_back.SetParameter(2,0.5)
            fit_back.SetParameter(3,1)
            fit_back.SetParameter(4,15)
            #fit_back.FixParameter(4,15)
    else:
        if SEL=="TtoEM":
            fit_back.SetParameter(0,7)
            fit_back.SetParameter(1,0)
            fit_back.SetParameter(2,0.5)
            fit_back.SetParameter(3,5)
            fit_back.SetParameter(4,5)
        elif SEL=="ZtoMM" or SEL=="ZtoEE" or SEL=="ZtoMMPho":
            fit_back.SetParameter(0,7)
            fit_back.SetParameter(1,0)
            fit_back.SetParameter(2,0.5)
            fit_back.SetParameter(3,5)
            fit_back.SetParameter(4,5)

    #Gaus
    #fit_back = TF1("fit_back","gaus")
    hist[ back[0] ].Fit(fit_back,"E")
    my_back = hist[ back[0] ].GetFunction("fit_back")
    hist[ back[0] ].GetListOfFunctions().Remove(hist[  back[0]  ].GetFunction("fit_back"))


    '''
    #Fit negative
    fit_data_neg = TF1("fit_data_neg","gaus",-5,0)
    #fit_data_neg.FixParameter(1, 0)
    hist[data[0]].Fit(fit_data_neg,"ER")#R fixes range, B fixes value
    my_data_neg = hist[data[0]].GetFunction("fit_data_neg")
    hist[data[0]].GetListOfFunctions().Remove(hist[data[0]].GetFunction("fit_data_neg"))

    fit_back_neg = TF1("fit_back_neg","gaus",-5,0)
    #fit_back_neg.FixParameter(1, 0)
    hist[ back[0] ].Fit(fit_back_neg,"ER")
    my_back_neg = hist[ back[0] ].GetFunction("fit_back_neg")
    hist[ back[0] ].GetListOfFunctions().Remove(hist[ back[0] ].GetFunction("fit_back_neg"))

    #Fit positive
    fit_data_pos = TF1("fit_data_pos","gaus",0,5)
    #fit_data_pos.FixParameter(1, 0)
    hist[data[0]].Fit(fit_data_pos,"ER")
    my_data_pos = hist[data[0]].GetFunction("fit_data_pos")
    hist[data[0]].GetListOfFunctions().Remove(hist[data[0]].GetFunction("fit_data_pos"))

    fit_back_pos = TF1("fit_back_pos","gaus",0,5)
    #fit_back_pos.FixParameter(1, 0)
    hist[ back[0] ].Fit(fit_back_pos,"ER")
    my_back_pos = hist[ back[0] ].GetFunction("fit_back_pos")
    hist[ back[0] ].GetListOfFunctions().Remove(hist[ back[0] ].GetFunction("fit_back_pos"))
    '''


    my_data.SetLineColor(1)
    my_data.SetLineStyle(2)
    my_back.SetLineStyle(2)

    data_file = TFile(OUT+"data_smear_file"+label+".root","RECREATE")
    data_file.cd()
    my_back.Write("back_CB")
    my_data.Write("data_CB")
    print "Writing "+OUT+"data_smear_file"+label+".root"
    data_file.Close()

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    can.SetRightMargin(0.05)

    leg = TLegend(0.65, 0.7, 1., 1.)
    leg2 = TLegend(0.65, 0.8, 1., 1.)
    for s in back:
        leg.AddEntry(hist[s],samples[s]['label'],"F")
        leg2.AddEntry(hist[s],samples[s]['label'],"F")
        hist[s].GetXaxis().SetTitle(variable[var]['title'])
        hist[s].Draw("HISTO,sames")
        hist['BkgSum'] = hist[s].Clone("BkgErr")
        hist['BkgSum'].SetFillStyle(3003)
        hist['BkgSum'].SetFillColor(1)
        hist['BkgSum'].Draw("SAME, E2")
        my_back.SetLineColor(2)#(samples[s]['linecolor']+1)
        print "Entries in ", s, " : " , hist[s].Integral()

    leg.AddEntry(my_back,"back fit: mean "+str(round(my_back.GetParameter(1),2))+", #sigma "+str(round(my_back.GetParameter(2),2)),"L")
        
    for s in data:
        leg.AddEntry(hist[s],samples[s]['label'],"P")
        leg2.AddEntry(hist[s],samples[s]['label'],"P")
        hist[s].GetXaxis().SetTitle(variable[var]['title'])
        hist[s].Draw("PE,sames")
        print "Entries in ", s, " : " , hist[s].Integral()

    leg.AddEntry(my_data,"data fit: mean "+str(round(my_data.GetParameter(1),2))+", #sigma "+str(round(my_data.GetParameter(2),2)),"L")

    my_data.Draw("L,sames")
    my_back.Draw("L,sames")

    '''
    my_data_pos.SetLineColor(801)
    my_data_pos.SetLineStyle(2)
    my_data_neg.SetLineColor(801)
    my_data_neg.SetLineStyle(3)
    my_data_pos.Draw("L,sames")
    my_data_neg.Draw("L,sames")

    my_back_pos.SetLineColor(8)
    my_back_pos.SetLineStyle(2)
    my_back_neg.SetLineColor(8)
    my_back_neg.SetLineStyle(3)
    my_back_pos.Draw("L,sames")
    my_back_neg.Draw("L,sames")
    '''



    '''
    timeSmear = np.array([])
    WeightSmear = np.array([])


    for d in back:
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

            print my_data.GetRandom()
            timeSmear = np.concatenate(( timeSmear,arrays["JetsNegative.timeRecHitsEB"][cut_jets][cut_mask] ))
            WeightSmear = np.concatenate(( WeightSmear, np.multiply(arrays["EventWeight"][cut_mask], arrays["PUReWeight"][cut_mask] )   ))

    smearVect = []
    for r in range(timeSmear.shape[0]):
        smearVect.append(my_data.GetRandom())

    smearVect = np.array(smearVect)

    print smearVect+timeSmear
    print WeightSmear
    print smearVect.flatten()
    print WeightSmear.flatten()
    #new_time_smear = np.multiply(WeightSmear,timeSmear+smearVect)
    root_numpy.fill_hist(smeared,timeSmear+smearVect,WeightSmear)

    print my_data_pos.GetParameter(2)
    print my_back_pos.GetParameter(2)

    print my_data_neg.GetParameter(2)
    print my_back_neg.GetParameter(2)
    '''

    OUTSTRING = OUT
    OUTSTRING += "jet_time_modelling" 

    if do_smear==False:
        leg.Draw()
        drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
        drawAnalysis("LL"+CHAN)
        drawRegion(SEL)
        latex = TLatex()
        latex.SetNDC()
        latex.SetTextAlign(33)
        latex.SetTextSize(0.04)
        latex.SetTextFont(62)
        #latex.DrawLatex(0.20, 0.96, "CMS")
        latex.SetTextFont(52)
        #latex.DrawLatex(0.36, 0.96, "Preliminary")
        can.Update()
        can.Print(OUTSTRING+label+'_no_smearing.png')
        can.Print(OUTSTRING+label+'_no_smearing.pdf')
        can.SetLogy()
        can.Update()
        can.Print(OUTSTRING+label+'_log_no_smearing.png')
        can.Print(OUTSTRING+label+'_log_no_smearing.pdf')
        can.Close()
        exit()


    smeared = TH1F("smeared",";"+variable[var]['title'],variable[var]['nbins'], variable[var]['min'],variable[var]['max'])
    smeared.Sumw2()

    timeSmeared = np.array([])
    EventWeight = np.array([])

    nTrackConstituents = np.array([])
    nSelectedTracks = np.array([])
    timeRecHitsEB = np.array([])
    eFracRecHitsEB = np.array([])
    nRecHitsEB = np.array([])
    sig1EB = np.array([])
    sig2EB = np.array([])
    ptDEB = np.array([])
    cHadEFrac = np.array([])
    nHadEFrac = np.array([])
    eleEFrac = np.array([])
    photonEFrac = np.array([])
    ptAllTracks = np.array([])
    ptAllPVTracks = np.array([])
    alphaMax = np.array([])
    betaMax = np.array([])
    gammaMax = np.array([])
    gammaMaxEM = np.array([])
    gammaMaxHadronic = np.array([])
    gammaMaxET = np.array([])
    minDeltaRAllTracks = np.array([])
    minDeltaRPVTracks = np.array([])

    #print "my_data parameters"
    #print my_data.GetParameter(0)
    #print my_data.GetParameter(1)
    #print my_data.GetParameter(2)
    #print my_data.GetParameter(3)
    #print my_data.GetParameter(4)

    #smear_cb = TF1("smear_cb","crystalball",5,5)
    smear_cb = my_data.Clone("smear_cb")
    smear_cb.SetParameter(0,my_data.GetParameter(0))
    smear_cb.SetParameter(1,my_data.GetParameter(1)-my_back.GetParameter(1))#0
    smear_cb.SetParameter(2, math.sqrt( abs(my_data.GetParameter(2)**2 - my_back.GetParameter(2)**2)) )
    smear_cb.SetParameter(3,my_data.GetParameter(3))
    smear_cb.SetParameter(4,my_data.GetParameter(4))
    
    #print "smear_cb parameters"
    #print smear_cb.GetParameter(0)
    #print smear_cb.GetParameter(1)
    #print smear_cb.GetParameter(2)
    #print smear_cb.GetParameter(3)
    #print smear_cb.GetParameter(4)

    #print "Throw random"
    #print my_back.GetRandom()
    #print my_data.GetRandom()
    #print smear_cb.GetRandom()

    for b in back:
        for j, ss in enumerate(samples[b]['files']):
            print "Performing gaussian smearing for ", ss
            tmp_file = TFile(MAIN+ss+".root","READ")
            tree = tmp_file.Get("tree")

            tree_weight = tree.GetWeight()
            tmp_time = []
            tmp_weight = []

            tmp_nTrackConstituents = []
            tmp_nSelectedTracks = []
            tmp_timeRecHitsEB = []
            tmp_eFracRecHitsEB = []
            tmp_nRecHitsEB = []
            tmp_sig1EB = []
            tmp_sig2EB = []
            tmp_ptDEB = []
            tmp_cHadEFrac = []
            tmp_nHadEFrac = []
            tmp_eleEFrac = []
            tmp_photonEFrac = []
            tmp_ptAllTracks = []
            tmp_ptAllPVTracks = []
            tmp_alphaMax = []
            tmp_betaMax = []
            tmp_gammaMax = []
            tmp_gammaMaxEM = []
            tmp_gammaMaxHadronic = []
            tmp_gammaMaxET = []
            tmp_minDeltaRAllTracks = []
            tmp_minDeltaRPVTracks = []


            for e in range(0,tree.GetEntries()):
                #print "entry n... ", e
                tree.GetEntry(e)
                neg_jets = getattr(tree,"JetsNegative")
                met = getattr(tree,"MEt")

                #Event level selections
                if SEL=="TtoEM":
                    if met.pt<200:
                        continue

                if tree.MinJetMetDPhi<=0.5:
                    continue

                #If only leading jet
                if neg_jets.size()>0 and "[0]" in var:
                    if SEL=="TtoEM":
                        if neg_jets[0].CSV<=0.8:
                            continue
                    if neg_jets[0].eta<=-1:
                        continue
                    if neg_jets[0].eta>=1:
                        continue
                    if ERA=="2017":
                        if neg_jets[0].phi>=2.7: continue
                    if ERA=="2018":
                        if neg_jets[0].phi>=0.4 and neg_jets[0].phi<0.9: continue

                    tmp_weight.append(tree.EventWeight * tree.PUReWeight * tree_weight)
                    #one gaussian
                    #tmp_time.append( neg_jets[0].timeRecHitsEB + gRandom.Gaus(my_back.GetParameter(1),math.sqrt(my_data.GetParameter(2)**2 - my_back.GetParameter(2)**2)) )
                    #my_data.GetRandom())
                    #CB
                    #my_data.SetParameter(0,1)
                    #my_data.SetParameter(1,0)
                    tmp_time.append( neg_jets[0].timeRecHitsEB + smear_cb.GetRandom())
                    
                    tmp_nTrackConstituents.append(neg_jets[0].nTrackConstituents)
                    tmp_nSelectedTracks.append(neg_jets[0].nSelectedTracks)
                    tmp_timeRecHitsEB.append(neg_jets[0].timeRecHitsEB)
                    tmp_eFracRecHitsEB.append(neg_jets[0].eFracRecHitsEB)
                    tmp_nRecHitsEB.append(neg_jets[0].nRecHitsEB)
                    tmp_sig1EB.append(neg_jets[0].sig1EB)
                    tmp_sig2EB.append(neg_jets[0].sig2EB)
                    tmp_ptDEB.append(neg_jets[0].ptDEB)
                    tmp_cHadEFrac.append(neg_jets[0].cHadEFrac)
                    tmp_nHadEFrac.append(neg_jets[0].nHadEFrac)
                    tmp_eleEFrac.append(neg_jets[0].eleEFrac)
                    tmp_photonEFrac.append(neg_jets[0].photonEFrac)
                    tmp_ptAllTracks.append(neg_jets[0].ptAllTracks)
                    tmp_ptAllPVTracks.append(neg_jets[0].ptAllPVTracks)
                    tmp_alphaMax.append(neg_jets[0].alphaMax)
                    tmp_betaMax.append(neg_jets[0].betaMax)
                    tmp_gammaMax.append(neg_jets[0].gammaMax)
                    tmp_gammaMaxEM.append(neg_jets[0].gammaMaxEM)
                    tmp_gammaMaxHadronic.append(neg_jets[0].gammaMaxHadronic)
                    tmp_gammaMaxET.append(neg_jets[0].gammaMaxET)
                    tmp_minDeltaRAllTracks.append(neg_jets[0].minDeltaRAllTracks)
                    tmp_minDeltaRPVTracks.append(neg_jets[0].minDeltaRPVTracks)
                    
                else:
                    for n in range(neg_jets.size()):
                        if SEL=="TtoEM":
                            if neg_jets[n].CSV<=0.8:
                                continue
                        if neg_jets[n].eta<=-1:
                            continue
                        if neg_jets[n].eta>=1:
                            continue
                        if ERA=="2017":
                            if neg_jets[n].phi>=2.7: continue
                        if ERA=="2018":
                            if neg_jets[n].phi>=0.4 and neg_jets[n].phi<0.9: continue

                        tmp_weight.append(tree.EventWeight * tree.PUReWeight * tree_weight)
                        tmp_time.append( neg_jets[n].timeRecHitsEB + smear_cb.GetRandom())
                        tmp_nTrackConstituents.append(neg_jets[n].nTrackConstituents)
                        tmp_nSelectedTracks.append(neg_jets[n].nSelectedTracks)
                        tmp_timeRecHitsEB.append(neg_jets[n].timeRecHitsEB)
                        tmp_eFracRecHitsEB.append(neg_jets[n].eFracRecHitsEB)
                        tmp_nRecHitsEB.append(neg_jets[n].nRecHitsEB)
                        tmp_sig1EB.append(neg_jets[n].sig1EB)
                        tmp_sig2EB.append(neg_jets[n].sig2EB)
                        tmp_ptDEB.append(neg_jets[n].ptDEB)
                        tmp_cHadEFrac.append(neg_jets[n].cHadEFrac)
                        tmp_nHadEFrac.append(neg_jets[n].nHadEFrac)
                        tmp_eleEFrac.append(neg_jets[n].eleEFrac)
                        tmp_photonEFrac.append(neg_jets[n].photonEFrac)
                        tmp_ptAllTracks.append(neg_jets[n].ptAllTracks)
                        tmp_ptAllPVTracks.append(neg_jets[n].ptAllPVTracks)
                        tmp_alphaMax.append(neg_jets[n].alphaMax)
                        tmp_betaMax.append(neg_jets[n].betaMax)
                        tmp_gammaMax.append(neg_jets[n].gammaMax)
                        tmp_gammaMaxEM.append(neg_jets[n].gammaMaxEM)
                        tmp_gammaMaxHadronic.append(neg_jets[n].gammaMaxHadronic)
                        tmp_gammaMaxET.append(neg_jets[n].gammaMaxET)
                        tmp_minDeltaRAllTracks.append(neg_jets[n].minDeltaRAllTracks)
                        tmp_minDeltaRPVTracks.append(neg_jets[n].minDeltaRPVTracks)
                    
            timeSmeared = np.concatenate((timeSmeared, np.array(tmp_time)))
            EventWeight = np.concatenate((EventWeight, np.array(tmp_weight)))
            nTrackConstituents = np.concatenate(( nTrackConstituents, tmp_nTrackConstituents ))
            nSelectedTracks = np.concatenate(( nSelectedTracks, tmp_nSelectedTracks))
            timeRecHitsEB = np.concatenate(( timeRecHitsEB, tmp_timeRecHitsEB))
            eFracRecHitsEB = np.concatenate(( eFracRecHitsEB, tmp_eFracRecHitsEB))
            nRecHitsEB = np.concatenate(( nRecHitsEB, tmp_nRecHitsEB))
            sig1EB = np.concatenate(( sig1EB, tmp_sig1EB))
            sig2EB = np.concatenate(( sig2EB, tmp_sig2EB))
            ptDEB = np.concatenate(( ptDEB, tmp_ptDEB))
            cHadEFrac = np.concatenate(( cHadEFrac, tmp_cHadEFrac))
            nHadEFrac = np.concatenate(( nHadEFrac, tmp_nHadEFrac))
            eleEFrac = np.concatenate(( eleEFrac, tmp_eleEFrac))
            photonEFrac = np.concatenate(( photonEFrac, tmp_photonEFrac))
            ptAllTracks = np.concatenate(( ptAllTracks, tmp_ptAllTracks))
            ptAllPVTracks = np.concatenate(( ptAllPVTracks, tmp_ptAllPVTracks))
            alphaMax = np.concatenate(( alphaMax, tmp_alphaMax))
            betaMax = np.concatenate(( betaMax, tmp_betaMax))
            gammaMax = np.concatenate(( gammaMax, tmp_gammaMax))
            gammaMaxEM = np.concatenate(( gammaMaxEM, tmp_gammaMaxEM))
            gammaMaxHadronic = np.concatenate(( gammaMaxHadronic, tmp_gammaMaxHadronic))
            gammaMaxET = np.concatenate(( gammaMaxET, tmp_gammaMaxET))
            minDeltaRAllTracks = np.concatenate(( minDeltaRAllTracks, tmp_minDeltaRAllTracks))
            minDeltaRPVTracks = np.concatenate(( minDeltaRPVTracks, tmp_minDeltaRPVTracks))
    
    model = keras.models.load_model('nn_inference/tagger_AK4_v3/model.h5')
    model.summary()

    dat_list = [
        nTrackConstituents,
        nSelectedTracks,
        timeRecHitsEB,
        eFracRecHitsEB,
        nRecHitsEB,
        sig1EB,
        sig2EB,
        ptDEB,
        cHadEFrac,
        nHadEFrac,
        eleEFrac,
        photonEFrac,
        ptAllTracks,
        ptAllPVTracks,
        alphaMax,
        betaMax,
        gammaMax,
        gammaMaxEM,
        gammaMaxHadronic,
        gammaMaxET,
        minDeltaRAllTracks,
        minDeltaRPVTracks,
    ]

    dat_list_smeared = [
        nTrackConstituents,
        nSelectedTracks,
        timeSmeared,#
        eFracRecHitsEB,
        nRecHitsEB,
        sig1EB,
        sig2EB,
        ptDEB,
        cHadEFrac,
        nHadEFrac,
        eleEFrac,
        photonEFrac,
        ptAllTracks,
        ptAllPVTracks,
        alphaMax,
        betaMax,
        gammaMax,
        gammaMaxEM,
        gammaMaxHadronic,
        gammaMaxET,
        minDeltaRAllTracks,
        minDeltaRPVTracks,
    ]
    
    X = np.transpose(np.stack((dat_list)))
    X_smear = np.transpose(np.stack((dat_list_smeared)))
    probs_orig = model.predict(X)
    probs_smear = model.predict(X_smear)

    root_numpy.fill_hist(mc_pass_sigprob,probs_orig[:,1],EventWeight)
    root_numpy.fill_hist(mc_smear_sigprob,probs_smear[:,1],EventWeight)
    root_numpy.fill_hist(smeared,timeSmeared,EventWeight)

    smeared.SetLineColor(418)
    smeared.SetLineWidth(2)
    if scale:
        smeared.Scale(1./smeared.Integral())
    smeared.Draw("HISTO,sames")

    leg.AddEntry(smeared,"MC smeared","L")
    leg.Draw()
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawAnalysis("LL"+CHAN)
    drawRegion(SEL)
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextAlign(33)
    latex.SetTextSize(0.04)
    latex.SetTextFont(62)
    #latex.DrawLatex(0.20, 0.96, "CMS")
    latex.SetTextFont(52)
    #latex.DrawLatex(0.36, 0.96, "Preliminary")
    can.Update()
    can.Print(OUTSTRING+label+'.png')
    can.Print(OUTSTRING+label+'.pdf')
    can.SetLogy()
    can.Update()
    can.Print(OUTSTRING+label+'_log.png')
    can.Print(OUTSTRING+label+'_log.pdf')
    can.Close()

    can2 = TCanvas("can2","can2",900,800)
    can2.cd()
    can2.SetRightMargin(0.05)


    data_sigprob.SetMarkerStyle(21)
    data_sigprob.SetMarkerColor(1)
    mc_sigprob.SetLineColor(samples[back[0]]['linecolor'])
    mc_sigprob.SetFillColorAlpha(samples[back[0]]['linecolor'],0.5)
    mc_sigprob.SetLineWidth(2)
    mc_pass_sigprob.SetLineColor(8)
    mc_pass_sigprob.SetLineWidth(2)
    mc_smear_sigprob.SetLineColor(418)
    mc_smear_sigprob.SetLineWidth(2)

    if scale:
        mc_sigprob.Scale(1./mc_sigprob.Integral())
        data_sigprob.Scale(1./data_sigprob.Integral())
        mc_pass_sigprob.Scale(1./mc_pass_sigprob.Integral())
        mc_smear_sigprob.Scale(1./mc_smear_sigprob.Integral())

    mc_sigprob.SetTitle("")
    mc_sigprob.GetXaxis().SetTitle("AK4 jet DNN score")
    mc_sigprob.Draw("HIST")
    data_sigprob.Draw("PE,sames")
    #mc_pass_sigprob.Draw("PE,sames")
    mc_smear_sigprob.Draw("HIST,sames")

    leg2.AddEntry(mc_smear_sigprob,"MC smeared","L")
    leg2.SetTextSize(0.03)
    leg2.Draw()

    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawAnalysis("LL"+CHAN)
    drawRegion(SEL)

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextAlign(33)
    latex.SetTextSize(0.04)
    latex.SetTextFont(62)
    #latex.DrawLatex(0.20, 0.96, "CMS")
    latex.SetTextFont(52)
    #latex.DrawLatex(0.36, 0.96, "Preliminary")

    can2.SetLogy()
    can2.Print(OUTSTRING+label+'_sigprob.png')
    can2.Print(OUTSTRING+label+'_sigprob.pdf')
    can2.Close()

###
def signal_smearing(var,cut,label="",scale=True, do_smear=True,added=""):
    list_of_variables = ["EventNumber","LumiNumber","RunNumber","Jets*.pt","Jets*.eta","Jets*.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight","TriggerWeight","PUWeight","PUReWeight","Jets*.isGenMatchedCaloCorrLLPAccept","Jets*.CSV","Jets*.timeRecHitsEB","is*","JetsNegative*.CSV","JetsNegative*.sigprob","JetsNegative*.timeRecHitsEB"]

    chain = {}
    hist = {}
    my_sign = {}
    mc_pass_sigprob = {}
    mc_smear_sigprob = {}
    h_time = {}
    h_time_smeared = {}
    h_nTags = {}
    h_nTagsSmeared = {}
    can2 = {}
    can3 = {}
    prj_weight = "EventWeight*PUReWeight*TriggerWeight"
    
    results = defaultdict(dict)


    #CB
    #Here: read TF1 from root file
    data_file = TFile(OUT_TtoEM+"data_smear_file_CSV_0p8_all_jets"+added+".root","READ")
    data_file.cd()
    my_data = data_file.Get("data_CB")
    my_back = data_file.Get("back_CB")
    data_file.Close()


    for s in sign:
        mc_pass_sigprob[s] = TH1F("mc_pass_sigprob"+s,"",50,0,1)
        mc_smear_sigprob[s] = TH1F("mc_smear_sigprob"+s,"",50,0,1)
        h_time[s] = TH1F("h_time"+s,"",variable[var]['nbins'], variable[var]['min'],variable[var]['max'])
        h_time_smeared[s] = TH1F("h_time_smeared"+s,"",variable[var]['nbins'], variable[var]['min'],variable[var]['max'])
        h_nTags[s] = TH1F("h_nTags"+s,"",3+3,-0.5,2.5+3)
        h_nTagsSmeared[s] = TH1F("h_nTagsSmeared"+s,"",3+3,-0.5,2.5+3)

        #mc_sigprob.Sumw2()
        mc_pass_sigprob[s].Sumw2()
        mc_smear_sigprob[s].Sumw2()
        h_time[s].Sumw2()
        h_time_smeared[s].Sumw2()
        h_nTags[s].Sumw2()
        h_nTagsSmeared[s].Sumw2()


    for s in sign:
        print "\n"
        print "***********************************************************************************"
        print "Doing ", s, " .... "
        #This is for the fit
        hist[s] = TH1F(s,";"+variable[var]['title'],variable[var]['nbins'], variable[var]['min'],variable[var]['max'])
        hist[s].Sumw2()

        chain[s] = TChain("tree")
        for j, ss in enumerate(samples[s]['files']):
            print MAIN_MET+ss + ".root"
            chain[s].Add(MAIN_MET + ss + ".root")

        chain[s].Project(s, var, prj_weight+"*("+cut+")")

        hist[s].SetLineColor(samples[s]['linecolor'])
        hist[s].SetFillColorAlpha(samples[s]['linecolor'],0.5)
        hist[s].SetLineWidth(2)
        if scale:
            hist[s].Scale(1./hist[s].Integral())

        print "\n"
        print " --- Fit ---"
        print "\n"

        #CB
        fit_sign = TF1("fit_sign","crystalball",5,5)
        if scale:
            if ERA=="2016":
                if "ctau500" in s:
                    fit_sign.SetParameter(0,0.01)
                    fit_sign.SetParameter(1,hist[s].GetMean())#0.2)
                    fit_sign.SetParameter(2,hist[s].GetRMS())#0.5)
                    fit_sign.SetParameter(3,-0.6)
                    fit_sign.SetParameter(4,7)#6)
                    #500:127 failed but looks perfect,400 failed but looks great,800 looks good too,1800 this is the only truly wrong
                    if samples[s]['mass']==127:
                        fit_sign.SetParameter(0,0.107)
                        fit_sign.SetParameter(1,0.119790)
                        fit_sign.SetParameter(2,0.3452)
                        fit_sign.FixParameter(3,-1.11)
                        fit_sign.SetParameter(4,25.66)
                    if samples[s]['mass']==400:
                        #still failed
                        fit_sign.SetParameter(1,0.33)
                        fit_sign.SetParameter(2,0.46)
                        fit_sign.FixParameter(3,-7.05947e-01)
                        fit_sign.FixParameter(4,3.47988e+06)
                    if samples[s]['mass']==800:
                        #still failed
                        fit_sign.SetParameter(1,0.35)
                        fit_sign.SetParameter(2,0.437)
                        fit_sign.FixParameter(3,-5.40345e-01)
                        fit_sign.SetParameter(4,8.88)
                    if samples[s]['mass']==1800:
                        fit_sign.SetParameter(1,0.38)
                        fit_sign.SetParameter(2,0.45)
                        fit_sign.SetParameter(3,-0.68)
                        fit_sign.SetParameter(4,2.79)
                if "ctau3000" in s:
                    fit_sign.SetParameter(0,0.01)
                    fit_sign.SetParameter(1,hist[s].GetMean())#0.2)
                    fit_sign.SetParameter(2,hist[s].GetRMS())#0.5)
                    fit_sign.SetParameter(3,-0.5)
                    fit_sign.SetParameter(4,5)#6)
                    if samples[s]['mass']==175:
                        fit_sign.SetParameter(1,0.1039)
                        fit_sign.SetParameter(2,0.38)
                        fit_sign.SetParameter(3,-0.84)
                        fit_sign.SetParameter(4,363.5)
                    if samples[s]['mass']==400:
                        fit_sign.SetParameter(1,(1.39505e-01+1.98093e-01)/2)
                        fit_sign.SetParameter(2,(3.99442e-01+4.21746e-01)/2)#0.42)
                        fit_sign.FixParameter(3,(-8.06280e-01-5.85057e-01)/2)#-0.58)
                        fit_sign.FixParameter(4,(6.14105+2.84710)/2)#53.36)
                    if samples[s]['mass']==1500:
                        fit_sign.SetParameter(1,(1.52805e-01+1.55724e-01)/2)
                        fit_sign.SetParameter(2,(3.97711e-01+3.87310e-01)/2)
                        fit_sign.SetParameter(3,(-4.48216e-01-5.14095e-01)/2)
                        fit_sign.SetParameter(4,(1.47212e+00+1.09609)/2)
                    if samples[s]['mass']==1800:
                        fit_sign.SetParameter(1,(1.52805e-01+1.55724e-01)/2)
                        fit_sign.SetParameter(2,(3.97711e-01+3.87310e-01)/2)
                        fit_sign.SetParameter(3,(-4.48216e-01-5.14095e-01)/2)
                        fit_sign.SetParameter(4,(1.47212e+00+1.09609)/2)
                    '''
                    if samples[s]['mass']==1250:
                        fit_sign.SetParameter(1,0.25)
                        fit_sign.SetParameter(2,0.43)#6)
                        #fit_sign.FixParameter(3,-0.3)
                        #fit_sign.FixParameter(4,7.6)#6)
                    if samples[s]['mass']==1000:
                        fit_sign.SetParameter(3,3)
                        fit_sign.SetParameter(4,4)#6)
                    if samples[s]['mass']==800:
                        fit_sign.SetParameter(3,-0.29)
                        fit_sign.SetParameter(4,11)#6)
                    if samples[s]['mass']==600:
                        fit_sign.SetParameter(3,-0.7)
                        fit_sign.SetParameter(4,4)#6)
                    if samples[s]['mass']==200:
                        fit_sign.FixParameter(3,(-9.50524e-01-8.59043e-01)/2.)
                        fit_sign.FixParameter(4,(5.56790+7.75565)/2.)#6)
                    if samples[s]['mass']==1800:
                        fit_sign.FixParameter(3,-4.21829e-01)
                        fit_sign.FixParameter(4,1.45618e+00 )#6)
                    '''
            if ERA=="2017":
                if "ctau500" in s:
                    fit_sign.SetParameter(0,0.01)
                    fit_sign.SetParameter(1,hist[s].GetMean())#0.2)
                    fit_sign.SetParameter(2,hist[s].GetRMS())#0.5)
                    fit_sign.SetParameter(3,-0.6)
                    fit_sign.SetParameter(4,7)#6)
                    if samples[s]['mass']==1800:
                        fit_sign.SetParameter(4,5)
                    if samples[s]['mass']==127:
                        fit_sign.SetParameter(0,9.32483e-02)
                        fit_sign.SetParameter(1,0)
                        fit_sign.SetParameter(2,3.86103e-01)
                        fit_sign.SetParameter(3,3.4)
                        fit_sign.SetParameter(4,1.1)
                        #fit_sign.FixParameter(1,1.85612e-01)
                        #fit_sign.FixParameter(2,3.86117e-01)
                        fit_sign.FixParameter(3,3.43582e+00)
                    if samples[s]['mass']==150:
                        fit_sign.SetParameter(3,1)
                        fit_sign.SetParameter(4,5)
                    if samples[s]['mass']==200:
                        fit_sign.SetParameter(0,0.09)
                        fit_sign.SetParameter(3,2.4)
                        fit_sign.SetParameter(4,30)
                        #fit_sign.FixParameter(1,2.19656e-01)
                        #fit_sign.FixParameter(2,3.85155e-01)
                        fit_sign.FixParameter(3,(-0.84-9.33397e-01)/2)#4.07541e+00)
                        fit_sign.FixParameter(4,(6.55+8.06723)/2)#8.35464e+01)
                    if samples[s]['mass']==300:
                        #fit_sign.FixParameter(1,3.32493e-01)
                        #fit_sign.FixParameter(2,4.50258e-01)
                        #fit_sign.FixParameter(3,-8.74358e-01)
                        fit_sign.FixParameter(4,9.79535e+06)
                    if samples[s]['mass']==800:
                        fit_sign.SetParameter(3,-0.4)
                        fit_sign.SetParameter(4,10)
                    if samples[s]['mass']==1000:
                        fit_sign.SetParameter(3,-0.4)
                        fit_sign.SetParameter(4,5)
                    if samples[s]['mass']==1250:
                        fit_sign.SetParameter(3,1)
                        fit_sign.SetParameter(4,5)
                    if samples[s]['mass']==1500:
                        print "kras"
                        #fit_sign.SetParameter(1,0.4)
                        fit_sign.FixParameter(3,-0.56)
                        fit_sign.FixParameter(4,4.2)

                if "ctau3000" in s:
                    fit_sign.SetParameter(0,0.01)
                    fit_sign.SetParameter(1,hist[s].GetMean())#0.2)
                    fit_sign.SetParameter(2,hist[s].GetRMS())#0.5)
                    fit_sign.SetParameter(3,-0.5)
                    fit_sign.SetParameter(4,5)#6)
                    if samples[s]['mass']==1250:
                        fit_sign.SetParameter(1,0.25)
                        fit_sign.SetParameter(2,0.43)#6)
                        #fit_sign.FixParameter(3,-0.3)
                        #fit_sign.FixParameter(4,7.6)#6)
                    if samples[s]['mass']==1000:
                        fit_sign.SetParameter(3,3)
                        fit_sign.SetParameter(4,4)#6)
                    if samples[s]['mass']==800:
                        fit_sign.SetParameter(3,-0.29)
                        fit_sign.SetParameter(4,11)#6)
                    if samples[s]['mass']==600:
                        fit_sign.SetParameter(3,-0.7)
                        fit_sign.SetParameter(4,4)#6)
                    if samples[s]['mass']==200:
                        fit_sign.FixParameter(3,(-9.50524e-01-8.59043e-01)/2.)
                        fit_sign.FixParameter(4,(5.56790+7.75565)/2.)#6)
                    if samples[s]['mass']==1800:
                        fit_sign.FixParameter(3,-4.21829e-01)
                        fit_sign.FixParameter(4,1.45618e+00 )#6)
                    #200 failed
                    #1800 failed
            if ERA=="2018":
                #good for fit that converged
                #fit_sign.SetParameter(0,0.01)
                #fit_sign.SetParameter(1,0.02)
                #fit_sign.SetParameter(2,0.31)
                #fit_sign.SetParameter(3,2)
                #fit_sign.SetParameter(4,5)
                #good for fit ctau 500 that have a negative alpha
                fit_sign.SetParameter(0,0.01)
                fit_sign.SetParameter(1,hist[s].GetMean())#0.2)
                fit_sign.SetParameter(2,hist[s].GetRMS())#0.5)
                fit_sign.SetParameter(3,-0.6)
                fit_sign.SetParameter(4,3)#6)
                if s=="SUSY_mh150_ctau3000":
                    fit_sign.SetParameter(1,hist[s].GetMean())#0.2)
                    fit_sign.SetParameter(2,hist[s].GetRMS())#0.5)
                    fit_sign.FixParameter(3,(-0.909096-0.879805)/2)
                    fit_sign.FixParameter(4,(7.01905+6.15354)/2)
                if s=="SUSY_mh175_ctau3000":
                    fit_sign.SetParameter(1,hist[s].GetMean())#0.2)
                    fit_sign.SetParameter(2,hist[s].GetRMS())#0.5)
                    fit_sign.SetParameter(3,4)
                    fit_sign.SetParameter(4,4)
                if s=="SUSY_mh400_ctau500":
                    fit_sign.SetParameter(1,hist[s].GetMean())#0.2)
                    fit_sign.SetParameter(2,hist[s].GetRMS())#0.5)
                    fit_sign.SetParameter(3,-0.8)
                    fit_sign.SetParameter(4,4)#6)
                    fit_sign.FixParameter(1,3.54795e-01)
                    fit_sign.FixParameter(2,4.61491e-01)
                    fit_sign.FixParameter(3,-7.16204e-01)
                    fit_sign.FixParameter(4,4.37591e+06)
                if s=="SUSY_mh1500_ctau500":
                    fit_sign.SetParameter(1,hist[s].GetMean())#0.2)
                    fit_sign.SetParameter(2,hist[s].GetRMS())#0.5)
                    fit_sign.SetParameter(3,-0.54)
                    fit_sign.SetParameter(4,(5.12+3.55)/2)#6)
                if s=="SUSY_mh1800_ctau3000":
                    fit_sign.SetParameter(1,hist[s].GetMean())#0.2)
                    fit_sign.SetParameter(2,hist[s].GetRMS())#0.5)
                    fit_sign.SetParameter(3,-5.31002e-01)
                    fit_sign.SetParameter(4,1)#6)
                    
        else:
            fit_sign.SetParameter(0,7)
            fit_sign.SetParameter(1,0)
            fit_sign.SetParameter(2,0.5)
            fit_sign.SetParameter(3,5)
            fit_sign.SetParameter(4,5)

        hist[s].Fit(fit_sign,"EB")
        my_sign[s] = hist[s].GetFunction("fit_sign")
        hist[s].GetListOfFunctions().Remove(hist[s].GetFunction("fit_sign"))

        my_data.SetLineColor(1)
        my_data.SetLineStyle(2)
        my_sign[s].SetLineStyle(2)


        hist[s].GetXaxis().SetTitle(variable[var]['title'])
        #hist[s].Draw("HISTO")

        my_sign[s].SetLineColor(2)#(samples[s]['linecolor']+1)
        print "Entries in ", s, " : " , hist[s].Integral()



        if do_smear==False:
            can = TCanvas("can","can",900,800)
            can.SetRightMargin(0.05)
            can.cd()
            leg = TLegend(0.65, 0.7, 1., 1.)
            leg.AddEntry(hist[s],samples[s]['label'],"F")
            leg.AddEntry(my_sign[s],"sign fit: mean "+str(round(my_sign[s].GetParameter(1),2))+", #sigma "+str(round(my_sign[s].GetParameter(2),2)),"L")
            leg.AddEntry(my_data,"data fit: mean "+str(round(my_data.GetParameter(1),2))+", #sigma "+str(round(my_data.GetParameter(2),2)),"L")

            hist[s].Draw("HISTO")
            my_sign[s].Draw("L,sames")
            my_data.Draw("L,sames")
            leg.Draw()
            drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
            drawAnalysis("LL"+CHAN)
            drawRegion(SEL)

            latex = TLatex()
            latex.SetNDC()
            latex.SetTextAlign(33)
            latex.SetTextSize(0.04)
            latex.SetTextFont(62)
            #latex.DrawLatex(0.20, 0.96, "CMS")
            latex.SetTextFont(52)
            #latex.DrawLatex(0.36, 0.96, "Preliminary")
            OUTSTRING = OUT
            OUTSTRING += s+"_jet_time_modelling" 
            can.Update()
            can.Print(OUTSTRING+label+'_no_smearing.png')
            can.Print(OUTSTRING+label+'_no_smearing.pdf')
            can.SetLogy()
            can.Update()
            can.Print(OUTSTRING+label+'_log_no_smearing.png')
            can.Print(OUTSTRING+label+'_log_no_smearing.pdf')
            can.Close()
            continue



        timeSmeared = np.array([])
        Weight = np.array([])
        WeightSmeared = np.array([])
        timeRecHitsEB = np.array([])
        sigprob = np.array([])
        sigprobSmeared = np.array([])

        #event-wise variables
        nTags = np.array([])
        nTagsSmeared = np.array([])
        EventWeight = np.array([])

        smear_cb = my_data.Clone("smear_cb")
        smear_cb.SetParameter(0,my_data.GetParameter(0))
        #Here: for signal I don't want to change the mean!!!
        #I want to correct only by the residual mean be data and MC background
        smear_cb.SetParameter(1,my_data.GetParameter(1)-my_back.GetParameter(1))#my_sign[s].GetParameter(1))#0
        smear_cb.SetParameter(2, math.sqrt( abs(my_data.GetParameter(2)**2 - my_sign[s].GetParameter(2)**2)) )
        smear_cb.SetParameter(3,my_data.GetParameter(3))
        smear_cb.SetParameter(4,my_data.GetParameter(4))

        model = keras.models.load_model('nn_inference/tagger_AK4_v3/model.h5')
        #model.summary()

        for j, ss in enumerate(samples[s]['files']):
            print "Performing gaussian smearing for ", ss
            tmp_file = TFile(MAIN_MET+ss+".root","READ")
            tree = tmp_file.Get("tree")
            tree_weight = tree.GetWeight()
            tmp_time = []
            tmp_timeRecHitsEB = []
            tmp_weight = []
            tmp_weightSmeared = []
            tmp_sigprob = []
            tmp_sigprobSmeared = []
            
            #event-wise variables
            tmp_eventweight = []
            tmp_nTags = []
            tmp_nTagsSmeared = []


            for e in range(0,tree.GetEntries()):
                tree.GetEntry(e)
                neg_jets = getattr(tree,"JetsNegative")
                met = getattr(tree,"MEt")
                nTagJets = 0
                nTagJetsSmeared = 0

                #Event level selections
                if SEL=="TtoEM":
                    if met.pt<200:
                        continue

                if tree.MinJetMetDPhi<=0.5:
                    continue

                #print "event ", e
                #print "nj: ", neg_jets.size()
                for n in range(neg_jets.size()):
                    if SEL=="TtoEM":
                        if neg_jets[n].CSV<=0.8:
                            continue
                    if neg_jets[n].eta<=-1:
                        continue
                    if neg_jets[n].eta>=1:
                        continue
                    if ERA=="2017":
                        if neg_jets[n].phi>=2.7: continue
                    if ERA=="2018":
                        if neg_jets[n].phi>=0.4 and neg_jets[n].phi<0.9: continue


                    time_rnd = neg_jets[n].timeRecHitsEB + smear_cb.GetRandom()


                    tmp_dat_list = [
                        neg_jets[n].nTrackConstituents,
                        neg_jets[n].nSelectedTracks,
                        neg_jets[n].timeRecHitsEB,
                        neg_jets[n].eFracRecHitsEB,
                        neg_jets[n].nRecHitsEB,
                        neg_jets[n].sig1EB,
                        neg_jets[n].sig2EB,
                        neg_jets[n].ptDEB,
                        neg_jets[n].cHadEFrac,
                        neg_jets[n].nHadEFrac,
                        neg_jets[n].eleEFrac,
                        neg_jets[n].photonEFrac,
                        neg_jets[n].ptAllTracks,
                        neg_jets[n].ptAllPVTracks,
                        neg_jets[n].alphaMax,
                        neg_jets[n].betaMax,
                        neg_jets[n].gammaMax,
                        neg_jets[n].gammaMaxEM,
                        neg_jets[n].gammaMaxHadronic,
                        neg_jets[n].gammaMaxET,
                        neg_jets[n].minDeltaRAllTracks,
                        neg_jets[n].minDeltaRPVTracks,
                    ]

                    tmp_dat_list_smeared = [
                        neg_jets[n].nTrackConstituents,
                        neg_jets[n].nSelectedTracks,
                        time_rnd,#
                        neg_jets[n].eFracRecHitsEB,
                        neg_jets[n].nRecHitsEB,
                        neg_jets[n].sig1EB,
                        neg_jets[n].sig2EB,
                        neg_jets[n].ptDEB,
                        neg_jets[n].cHadEFrac,
                        neg_jets[n].nHadEFrac,
                        neg_jets[n].eleEFrac,
                        neg_jets[n].photonEFrac,
                        neg_jets[n].ptAllTracks,
                        neg_jets[n].ptAllPVTracks,
                        neg_jets[n].alphaMax,
                        neg_jets[n].betaMax,
                        neg_jets[n].gammaMax,
                        neg_jets[n].gammaMaxEM,
                        neg_jets[n].gammaMaxHadronic,
                        neg_jets[n].gammaMaxET,
                        neg_jets[n].minDeltaRAllTracks,
                        neg_jets[n].minDeltaRPVTracks,
                    ]

                    #print "\n"
                    #print "Now doing all with negative jets; must move to jets in eta acceptance!"
                    #print "\n"

                    #X = np.transpose(np.stack((dat_list)))
                    tmp_X_smear = np.array(tmp_dat_list_smeared).reshape(1,len(tmp_dat_list_smeared))
                    tmp_probs_smear = model.predict(tmp_X_smear)

                    #This is a check only
                    #tmp_X = np.array(tmp_dat_list).reshape(1,len(tmp_dat_list))
                    #tmp_probs = model.predict(tmp_X)
                    
                    #HERE TODO
                    #Store only time and sigprob of jets passing acceptance and with time>-1
                    #After smearing: cut away jets with time<-1

                    if neg_jets[n].timeRecHitsEB>-1:
                        tmp_sigprob.append(neg_jets[n].sigprob)
                        tmp_timeRecHitsEB.append(neg_jets[n].timeRecHitsEB)
                        tmp_weight.append(tree.EventWeight * tree.PUReWeight * tree_weight)
                        #count tags
                        if neg_jets[n].sigprob>0.996:
                            nTagJets+=1
                    if time_rnd>-1:
                        tmp_sigprobSmeared.append(tmp_probs_smear[:,1][0])
                        tmp_time.append(time_rnd)
                        tmp_weightSmeared.append(tree.EventWeight * tree.PUReWeight * tree_weight)
                        #count tags
                        if tmp_probs_smear[:,1][0]>0.996:
                            nTagJetsSmeared+=1

                #loop in the event
                #event-wise variables
                tmp_eventweight.append(tree.EventWeight * tree.PUReWeight * tree_weight)
                tmp_nTags.append(nTagJets)
                tmp_nTagsSmeared.append(nTagJetsSmeared)
                    
            sigprob = np.concatenate((sigprob,np.array(tmp_sigprob)))
            sigprobSmeared = np.concatenate((sigprobSmeared,np.array(tmp_sigprobSmeared)))
            timeSmeared = np.concatenate((timeSmeared, np.array(tmp_time)))
            Weight = np.concatenate((Weight, np.array(tmp_weight)))
            WeightSmeared = np.concatenate((WeightSmeared, np.array(tmp_weightSmeared)))
            timeRecHitsEB = np.concatenate(( timeRecHitsEB, tmp_timeRecHitsEB))

            #event-wise variables
            EventWeight = np.concatenate(( EventWeight, tmp_eventweight))
            nTags = np.concatenate(( nTags, tmp_nTags))
            nTagsSmeared = np.concatenate(( nTagsSmeared, tmp_nTagsSmeared ))
    
        root_numpy.fill_hist(mc_pass_sigprob[s],sigprob,Weight)
        root_numpy.fill_hist(mc_smear_sigprob[s],sigprobSmeared,WeightSmeared)
        root_numpy.fill_hist(h_time[s],timeRecHitsEB,Weight)
        root_numpy.fill_hist(h_time_smeared[s],timeSmeared,WeightSmeared)

        root_numpy.fill_hist(h_nTags[s],nTags,EventWeight)
        root_numpy.fill_hist(h_nTagsSmeared[s],nTagsSmeared,EventWeight)

        h_time_smeared[s].SetLineColor(418)
        h_time_smeared[s].SetLineWidth(2)

        h_time[s].SetLineColor(samples[s]['linecolor'])
        h_time[s].SetFillColorAlpha(samples[s]['linecolor'],0.5)
        h_time[s].SetLineWidth(2)


        if scale:
            h_time[s].Scale(1./h_time[s].Integral())
            h_time_smeared[s].Scale(1./h_time_smeared[s].Integral())

        results[s]['SR'] = h_nTags[s].Integral(2,-1)
        results[s]['SR_smear'] = h_nTagsSmeared[s].Integral(2,-1)
        results[s]['diff'] = 100*abs(h_nTagsSmeared[s].Integral(2,-1) - h_nTags[s].Integral(2,-1))/(h_nTags[s].Integral(2,-1))


    print results
    with open(OUT+"signal_time_smearing_integrals"+added+".yaml","w") as f:
        yaml.dump(results, f)
        f.close()
    print "Written in ", OUT+"signal_time_smearing_integrals"+added+".yaml"

    with open(SYST_UNC_DIR+"signal_time_smearing_unc"+added+".yaml","w") as f:
        yaml.dump(results, f)
        f.close()
    print "Written in ", SYST_UNC_DIR+"signal_time_smearing_unc"+added+".yaml"

    for s in sign:
        print s, h_time[s].Print()
        OUTSTRING = OUT
        OUTSTRING += s+"_jet_time_modelling" 

        can = TCanvas("can","can",900,800)
        can.SetRightMargin(0.05)
        can.cd()
        h_time[s].Draw("HISTO")
        my_data.Draw("L,sames")
        my_sign[s].Draw("L,sames")
        h_time_smeared[s].Draw("HISTO,sames")

        leg = TLegend(0.65, 0.7, 1., 1.)
        leg.AddEntry(h_time[s],samples[s]['label'],"F")
        leg.AddEntry(my_sign[s],"sign fit: mean "+str(round(my_sign[s].GetParameter(1),2))+", #sigma "+str(round(my_sign[s].GetParameter(2),2)),"L")
        leg.AddEntry(my_data,"data fit: mean "+str(round(my_data.GetParameter(1),2))+", #sigma "+str(round(my_data.GetParameter(2),2)),"L")
        leg.AddEntry(h_time_smeared[s],"MC smeared","L")
        leg.Draw()

        drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
        drawAnalysis("LL"+CHAN)
        drawRegion(SEL)

        latex = TLatex()
        latex.SetNDC()
        latex.SetTextAlign(33)
        latex.SetTextSize(0.04)
        latex.SetTextFont(62)
        #latex.DrawLatex(0.20, 0.96, "CMS")
        latex.SetTextFont(52)
        #latex.DrawLatex(0.36, 0.96, "Preliminary")
        can.Update()
        can.Print(OUTSTRING+label+'.png')
        can.Print(OUTSTRING+label+'.pdf')
        can.SetLogy()
        can.Update()
        can.Print(OUTSTRING+label+'_log.png')
        can.Print(OUTSTRING+label+'_log.pdf')
        can.Close()

        can2 = TCanvas("can2","can2",900,800)
        can2.cd()
        can2.SetRightMargin(0.05)
        
        
        mc_smear_sigprob[s].SetLineColor(418)
        mc_smear_sigprob[s].SetLineWidth(2)

        mc_pass_sigprob[s].SetTitle("")
        mc_pass_sigprob[s].GetXaxis().SetTitle("AK4 jet DNN score")
        mc_pass_sigprob[s].SetFillColorAlpha(samples[s]['linecolor'],0.5)
        mc_pass_sigprob[s].SetLineColor(samples[s]['linecolor'])

        mc_pass_sigprob[s].Draw("HIST")
        mc_smear_sigprob[s].Draw("HIST,sames")

        leg2 = TLegend(0.65-0.2, 0.8, 1.-0.2, 1.)
        leg3 = TLegend(0.65-0.2, 0.8, 1.-0.2, 1.)
        leg2.AddEntry(h_time[s],samples[s]['label'],"F")
        leg3.AddEntry(h_time[s],samples[s]['label'],"F")
        leg2.AddEntry(mc_smear_sigprob[s],"MC smeared","L")
        leg3.AddEntry(mc_smear_sigprob[s],"MC smeared","L")
        leg2.Draw()

        drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
        drawAnalysis("LL"+CHAN)
        drawRegion(SEL)

        latex = TLatex()
        latex.SetNDC()
        latex.SetTextAlign(33)
        latex.SetTextSize(0.04)
        latex.SetTextFont(62)
        #latex.DrawLatex(0.20, 0.96, "CMS")
        latex.SetTextFont(52)
        #latex.DrawLatex(0.36, 0.96, "Preliminary")

        can2.SetLogy()
        can2.Print(OUTSTRING+label+'_sigprob.png')
        can2.Print(OUTSTRING+label+'_sigprob.pdf')
        can2.Close()

        #can3
        
        root_numpy.fill_hist(h_nTags[s],nTags,EventWeight)
        root_numpy.fill_hist(h_nTagsSmeared[s],nTagsSmeared,EventWeight)

        can3 = TCanvas("can3","can3",900,800)
        can3.cd()
        can3.SetRightMargin(0.05)

        h_nTags[s].SetMarkerStyle(20)
        h_nTags[s].SetMarkerColor(samples[s]['linecolor'])
        h_nTags[s].SetFillColorAlpha(samples[s]['linecolor'],0.5)
        h_nTags[s].SetLineColor(samples[s]['linecolor'])
        h_nTags[s].SetLineWidth(2)
        h_nTagsSmeared[s].SetMarkerStyle(20)
        h_nTagsSmeared[s].SetMarkerColor(418)
        h_nTagsSmeared[s].SetLineColor(418)
        h_nTagsSmeared[s].SetLineWidth(2)
        h_nTags[s].SetTitle("")
        h_nTags[s].GetXaxis().SetTitle("n. tagged jets")
        h_nTags[s].Draw("HIST")
        h_nTagsSmeared[s].Draw("HIST,sames")

        leg3.Draw()

        drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
        drawAnalysis("LL"+CHAN)
        drawRegion(SEL)

        latex = TLatex()
        latex.SetNDC()
        latex.SetTextAlign(33)
        latex.SetTextSize(0.04)
        latex.SetTextFont(62)
        #latex.DrawLatex(0.20, 0.96, "CMS")
        latex.SetTextFont(52)
        #latex.DrawLatex(0.36, 0.96, "Preliminary")

        can3.SetLogy()
        can3.Print(OUTSTRING+label+'_nTags.png')
        can3.Print(OUTSTRING+label+'_nTags.pdf')        
        can3.Close()


def draw_syst_unc(added):
    print "Drawing uncertainty vs mass"
    with open(OUT+"signal_time_smearing_integrals"+added+".yaml","r") as f:
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
                    string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % results[s]['diff'])
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"

    for c in ctaus:
        diff[c] = TGraph()
        n = 0
        for m in masses:
            for s in samp:
                if "mh"+str(m) in s and "ctau"+str(c) in s:
                    diff[c].SetPoint(n,m,results[s]['diff'])
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
    c1.SetTicks(1, 1)
    mg.GetXaxis().SetTitle("m_{\chi} (GeV)")
    mg.GetYaxis().SetTitle("Uncertainty (%)")
    mg.GetXaxis().SetTitleSize(0.05)
    mg.GetYaxis().SetTitleSize(0.05)
    mg.Draw("APL")
    leg1.Draw()
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawAnalysis("LL"+CHAN)
    drawRegion(SEL)
    #c1.SetLogx()
    c1.Print(OUT+"signal_uncertainty_time_smearing"+added+".pdf")
    c1.Print(OUT+"signal_uncertainty_time_smearing"+added+".png")
    c1.Close()
 
lab = "_not_scaled"#"_constant_1"
lab = ""
if SEL=="TtoEM":
    #time_fit(var="JetsNegative[0].timeRecHitsEB",cut = "MEt.pt>200 && isTtoEM && JetsNegative[0].CSV>0.8",label=lab,scale=True,do_smear=True)
    time_fit(var="JetsNegative.timeRecHitsEB",cut = "MEt.pt>200 && isTtoEM && JetsNegative.CSV>0.8 && fabs(Jets.eta)<1 && MinJetMetDPhi>0.5",label=lab+"_CSV_0p8_all_jets_G-H",scale=True,do_smear=True)#True#"_B-F"
    #time_fit(var="JetsNegative[0].timeRecHitsEB",cut = "MEt.pt>100 && isTtoEM && JetsNegative[0].CSV>0.5",label=lab+"_CSV_0p5_MET_100",scale=True,do_smear=True)
    #time_fit(var="JetsNegative[0].timeRecHitsEB",cut = "isTtoEM && JetsNegative[0].CSV>0.5",label=lab+"_CSV_0p5_MET_0",scale=True,do_smear=True)
if SEL=="SR":
    #time_fit(var="JetsNegative[0].timeRecHitsEB",cut = "isSR",label=lab,scale=True,do_smear=False)
    #time_fit(var="JetsNegative.timeRecHitsEB",cut = "isSR && fabs(Jets.eta)<1 && MinJetMetDPhi>0.5",label=lab+"_all_jets",scale=True,do_smear=False)
    added = ""
    #added="_B-F"
    signal_smearing(var="JetsNegative.timeRecHitsEB",cut = "isSR && fabs(Jets.eta)<1 && MinJetMetDPhi>0.5",label=lab+"_all_jets"+added,added=added,scale=True,do_smear=True)
    draw_syst_unc(added)
if SEL=="ZtoMM":
    time_fit(var="JetsNegative[0].timeRecHitsEB",cut = "isZtoMM && fabs(Jets.eta)<1 && MinJetMetDPhi>0.5",label=lab,scale=True,do_smear=False)
    ##time_fit(var="JetsNegative[0].timeRecHitsEB",cut = "MEt.pt>100",label=lab+"_MET_100",scale=True,do_smear=False)
if SEL=="ZtoEE":
    time_fit(var="JetsNegative[0].timeRecHitsEB",cut = "isZtoEE && fabs(Jets.eta)<1 && MinJetMetDPhi>0.5",label=lab,scale=True,do_smear=False)
if SEL=="ZtoMMPho":
    time_fit(var="JetsNegative[0].timeRecHitsEB",cut = "isZtoMMPho",label=lab,scale=True,do_smear=False)
