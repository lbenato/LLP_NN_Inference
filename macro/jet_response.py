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

#SEL = "TtoEM"
#SEL = "ZtoMM"
SEL = "ZtoLL"

ERA = "2018"
LUMI = -1

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

CHAN = "SUSY"

if SEL=="ZtoMMPho":
    data = ["SingleMuon"]
    back = ["DYJetsToLL"]
if SEL=="ZtoEEPho":
    if ERA=="2018":
        data = ["EGamma"]
    else:
        data = ["SingleElectron"]
    back = ["DYJetsToLL"]
if SEL=="ZtoLL":
    if ERA=="2018":
        dataEE = ["EGamma"]
    else:
        dataEE = ["SingleElectron"]
    dataMM = ["SingleMuon"]
    data = ["SingleLepton"]#dataMM+dataEE
    back = ["DYJetsToLL"]
if SEL=="TtoEM":
    data = ["MuonEG"]
    back = ["TTbarGenMET"]
if SEL=="SR":
    data = ["HighMET"]#["QCD"]
    back = ["ZJetsToNuNu"]
sign = ['SUSY_mh127_ctau500','SUSY_mh150_ctau500','SUSY_mh175_ctau500','SUSY_mh200_ctau500','SUSY_mh250_ctau500','SUSY_mh300_ctau500','SUSY_mh400_ctau500','SUSY_mh600_ctau500','SUSY_mh800_ctau500','SUSY_mh1000_ctau500','SUSY_mh1250_ctau500','SUSY_mh1500_ctau500','SUSY_mh1800_ctau500']
sign += ['SUSY_mh127_ctau3000','SUSY_mh150_ctau3000','SUSY_mh175_ctau3000','SUSY_mh200_ctau3000','SUSY_mh250_ctau3000','SUSY_mh300_ctau3000','SUSY_mh400_ctau3000','SUSY_mh600_ctau3000','SUSY_mh800_ctau3000','SUSY_mh1000_ctau3000','SUSY_mh1250_ctau3000','SUSY_mh1500_ctau3000','SUSY_mh1800_ctau3000']

sign = ['SUSY_mh127_ctau500_HH','SUSY_mh300_ctau500_HH','SUSY_mh600_ctau500_HH','SUSY_mh1000_ctau500_HH','SUSY_mh1800_ctau500_HH']

MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_"+SEL+"_v5_ntuples/"
MAIN_MM = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_ZtoMM_v5_ntuples/"
MAIN_EE = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_ZtoEE_v5_ntuples/"

MAIN_MET = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
MAIN_GEN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_Gen/"
OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_"+SEL+"_v5_ntuples/"
OUT_TtoEM = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_TtoEM_v5_ntuples_validate_timeRecHits/"
OUT_ALL = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+SEL+"/"

def deltaPhi_np( a1, v2):
    a2 = np.multiply(np.ones(a1.size),v2)
    pi = np.multiply(np.ones(a1.size),math.pi)
    diff = np.abs(a1 - a2)
    mask = diff>pi
    M = np.ma.masked_array(data=diff, mask = mask)
    new = diff - 2*math.pi*np.ones(diff.size)
    M.filled(new)
    return np.array(M)

def plot_all_years(cut,label=""):
    #For some reasons saving TH2F in a root file leads to crashes
    #Need to loop over all years

    binmin_y = 0.#0.5##0.84#0
    binmax_y = 2##1.06#3#1.5
    nbins_y = 50#50

    if SEL=="SR":
        prj_weight = "EventWeight*PUReWeight*TriggerWeight"
    else:
        prj_weight = "EventWeight*PUReWeight"

    MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_"+SEL+"_v5_ntuples/"
    MAIN_MM = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_ZtoMM_v5_ntuples/"
    MAIN_EE = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_ZtoEE_v5_ntuples/"

    b_hist_prob_zoom = TH2F("back_prob_zoom","", 25, 0.99, 1., nbins_y, binmin_y, binmax_y)
    b_hist_prob_zoom.Sumw2()
    d_hist_prob_zoom = TH2F("data_prob_zoom","", 25, 0.99, 1., nbins_y, binmin_y, binmax_y)
    d_hist_prob_zoom.Sumw2()

    b_chain = TChain("tree")
    d_chain = TChain("tree")

    eras = ["2016","2017","2018"]
    for s in back:
        for era in eras:
            if era=="2016":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples
            if era=="2017":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
            if era=="2018":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
            for j, ss in enumerate(samples[s]['files']):
                print "TChain "+(MAIN_EE%era)+ss + ".root"
                b_chain.Add((MAIN_EE%era) + ss + ".root")
                print "TChain "+(MAIN_MM%era)+ss + ".root"
                b_chain.Add((MAIN_MM%era) + ss + ".root")

    for s in data:
        for era in eras:
            if era=="2016":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples
            if era=="2017":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
            if era=="2018":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
            for j, ss in enumerate(samples[s]['files']):
                if 'Muon' in ss:
                    print "TChain: " + (MAIN_MM%era)+ss + ".root"
                    d_chain.Add((MAIN_MM%era) + ss + ".root")
                if 'Electron' in ss or 'EGamma' in ss:
                    print (MAIN_EE%era)+ss + ".root"
                    if "EGammaRun2018D-PromptReco" in ss:
                        print "too large"
                        d_chain.Add((MAIN_EE%era) + ss + "_part1.root")
                    else:
                        print "ok"
                        d_chain.Add((MAIN_EE%era) + ss + ".root")

    #Most likely this TChain is too large and it will crash
    #So no TChain, no histograms... probably will have to store only tagged stuff?

    b_chain.Project("back_prob_zoom", "(Jets[0].pt/Z_pt):Jets[0].sigprob", prj_weight+"*("+cut+")")
    d_chain.Project("data_prob_zoom", "(Jets[0].pt/Z_pt):Jets[0].sigprob", prj_weight+"*("+cut+")")

    #Plot zoom prob
    can = TCanvas("can","can",900,800)
    can.SetRightMargin(0.04)
    can.SetLeftMargin(0.12)
    can.cd()
    can.SetGrid()
    leg = TLegend(0.7, 0.7, 0.95, 0.9)
    #print b_hist_prob_zoom.Print()
    b_profX = TProfile(b_hist_prob_zoom.ProfileX("b_prof"))
    b_profX.SetMarkerStyle(21)
    b_profX.SetMarkerColor(samples[back[0]]['linecolor'])
    b_profX.SetLineColor(samples[back[0]]['linecolor'])
    b_profX.SetLineWidth(2)
    b_profX.Draw("PE,sames")
    b_profX.GetYaxis().SetTitle("Jet response (p_{T}^{jet_{0}}/p_{T}^{Z})")
    b_profX.GetXaxis().SetTitle("AK4 jet[0] DNN score")#("p_{T}^{Z}")

    d_profX = TProfile(d_hist_prob_zoom.ProfileX("d_prof"))
    d_profX.SetMarkerStyle(21)
    d_profX.SetMarkerColor(1)
    d_profX.SetLineColor(1)
    d_profX.SetLineWidth(2)
        
    d_profX.Draw("PE,sames")
    b_profX.SetMinimum(binmin_y)
    b_profX.SetMaximum(binmax_y)

    OUTSTRING = OUT_ALL
    OUTSTRING += "jet_response"
    OUTSTRING += "_vs_DNN_score_zoom"
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawRegion(SEL)
    leg.AddEntry(b_profX,"MC","PL")
    leg.AddEntry(d_profX,"data","PL")
    leg.Draw()
    can.Update()
    can.Print(OUTSTRING+label+'.png')
    can.Print(OUTSTRING+label+'.pdf')
    #b_profX.Delete()
    #d_profX.Delete()
    can.Close()

def jet_response(var_list,cut,dnn,label=""):
    if ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples
        from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
        if SEL=="ZtoLL":
            LUMI = lumi["SingleMuon"]["tot"]+lumi["SingleElectron"]["tot"]
    if ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
        from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
        if SEL=="ZtoLL":
            LUMI = lumi["SingleMuon"]["tot"]+lumi["SingleElectron"]["tot"]
    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
        from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
        if SEL=="ZtoLL":
            LUMI = lumi["SingleMuon"]["tot"]+lumi["EGamma"]["tot"]

    nbins_x = 25#variable[var]["nbins"]#50#50

    binmin_y = 0.#0.5##0.84#0
    binmax_y = 2##1.06#3#1.5
    nbins_y = 50#50
    b_chain = TChain("tree")
    d_chain = TChain("tree")
    b_chain_tag = TChain("tree")
    d_chain_tag = TChain("tree")

    b_hist = {}
    d_hist = {}
    b_hist_tag = {}
    d_hist_tag = {}
    b_hist_untag_shift1 = {}
    b_hist_tag_shift1 = {}
    b_hist_tag_shift2 = {}
    b_hist_tag_shift3 = {}
    b_hist_tag_shift4 = {}
    d_hist_untag_shift1 = {}
    d_hist_tag_shift1 = {}
    d_hist_tag_shift2 = {}
    d_hist_tag_shift3 = {}
    d_hist_tag_shift4 = {}

    for var in var_list:
        binmin_x = variable[var]["min"]#0##0
        binmax_x = variable[var]["max"]#200##0.4#2#1

        b_hist[var] = TH2F("back"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        b_hist[var].Sumw2()
        d_hist[var] = TH2F("data"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        d_hist[var].Sumw2()

        b_hist_tag[var] = TH2F("back_tag"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        b_hist_tag[var].Sumw2()
        d_hist_tag[var] = TH2F("data_tag"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        d_hist_tag[var].Sumw2()

        b_hist_untag_shift1[var] = TH2F("back_untag_shift1"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        b_hist_untag_shift1[var].Sumw2()
        d_hist_untag_shift1[var] = TH2F("data_untag_shift1"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        d_hist_untag_shift1[var].Sumw2()

        b_hist_tag_shift1[var] = TH2F("back_tag_shift1"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        b_hist_tag_shift1[var].Sumw2()
        d_hist_tag_shift1[var] = TH2F("data_tag_shift1"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        d_hist_tag_shift1[var].Sumw2()

        b_hist_tag_shift2[var] = TH2F("back_tag_shift2"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        b_hist_tag_shift2[var].Sumw2()
        d_hist_tag_shift2[var] = TH2F("data_tag_shift2"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        d_hist_tag_shift2[var].Sumw2()

        b_hist_tag_shift3[var] = TH2F("back_tag_shift3"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        b_hist_tag_shift3[var].Sumw2()
        d_hist_tag_shift3[var] = TH2F("data_tag_shift3"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        d_hist_tag_shift3[var].Sumw2()
        
        b_hist_tag_shift4[var] = TH2F("back_tag_shift4"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        b_hist_tag_shift4[var].Sumw2()
        d_hist_tag_shift4[var] = TH2F("data_tag_shift4"+var,"", nbins_x, binmin_x, binmax_x, nbins_y, binmin_y, binmax_y)
        d_hist_tag_shift4[var].Sumw2()
        

    b_hist_time = TH1F("b_hist_time","",variable["JetsNegative.timeRecHitsEB"]['nbins'],variable["JetsNegative.timeRecHitsEB"]['min'],variable["JetsNegative.timeRecHitsEB"]['max'])
    b_hist_time.Sumw2()
    b_hist_time_smeared = TH1F("b_hist_time_smeared","",variable["JetsNegative.timeRecHitsEB"]['nbins'],variable["JetsNegative.timeRecHitsEB"]['min'],variable["JetsNegative.timeRecHitsEB"]['max'])
    b_hist_time_smeared.Sumw2()

    d_hist_time = TH1F("d_hist_time","",variable["JetsNegative.timeRecHitsEB"]['nbins'],variable["JetsNegative.timeRecHitsEB"]['min'],variable["JetsNegative.timeRecHitsEB"]['max'])
    d_hist_time.Sumw2()



    b_hist_1d = TH1F("back_1d","", nbins_y, binmin_y, binmax_y)
    b_hist_1d.Sumw2()
    d_hist_1d = TH1F("data_1d","", nbins_y, binmin_y, binmax_y)
    d_hist_1d.Sumw2()


    #x-check: does response change if we perform the time cut?
    b_hist_1d_neg = TH1F("back_1d_neg","", nbins_y, binmin_y, binmax_y)
    b_hist_1d_neg.Sumw2()
    d_hist_1d_neg = TH1F("data_1d_neg","", nbins_y, binmin_y, binmax_y)
    d_hist_1d_neg.Sumw2()
    b_hist_1d_neg_cut = TH1F("back_1d_neg_cut","", nbins_y, binmin_y, binmax_y)
    b_hist_1d_neg_cut.Sumw2()
    d_hist_1d_neg_cut = TH1F("data_1d_neg_cut","", nbins_y, binmin_y, binmax_y)
    d_hist_1d_neg_cut.Sumw2()

    b_hist_tag_1d = TH1F("back_tag_1d","", nbins_y, binmin_y, binmax_y)
    b_hist_tag_1d.Sumw2()
    d_hist_tag_1d = TH1F("data_tag_1d","", nbins_y, binmin_y, binmax_y)
    d_hist_tag_1d.Sumw2()

    b_hist_tag_1d_shift1 = TH1F("back_tag_1d_shift1","", nbins_y, binmin_y, binmax_y)
    b_hist_tag_1d_shift1.Sumw2()
    d_hist_tag_1d_shift1 = TH1F("data_tag_1d_shift1","", nbins_y, binmin_y, binmax_y)
    d_hist_tag_1d_shift1.Sumw2()

    b_hist_tag_1d_shift2 = TH1F("back_tag_1d_shift2","", nbins_y, binmin_y, binmax_y)
    b_hist_tag_1d_shift2.Sumw2()
    d_hist_tag_1d_shift2 = TH1F("data_tag_1d_shift2","", nbins_y, binmin_y, binmax_y)
    d_hist_tag_1d_shift2.Sumw2()

    b_hist_tag_1d_shift3 = TH1F("back_tag_1d_shift3","", nbins_y, binmin_y, binmax_y)
    b_hist_tag_1d_shift3.Sumw2()
    d_hist_tag_1d_shift3 = TH1F("data_tag_1d_shift3","", nbins_y, binmin_y, binmax_y)
    d_hist_tag_1d_shift3.Sumw2()

    b_hist_tag_1d_shift4 = TH1F("back_tag_1d_shift4","", nbins_y, binmin_y, binmax_y)
    b_hist_tag_1d_shift4.Sumw2()
    d_hist_tag_1d_shift4 = TH1F("data_tag_1d_shift4","", nbins_y, binmin_y, binmax_y)
    d_hist_tag_1d_shift4.Sumw2()

    '''
    b_hist_prob_zoom = TH2F("back_prob_zoom","", 100, 0.95, 1., nbins_y, binmin_y, binmax_y)
    b_hist_prob_zoom.Sumw2()
    d_hist_prob_zoom = TH2F("data_prob_zoom","", 100, 0.95, 1., nbins_y, binmin_y, binmax_y)
    d_hist_prob_zoom.Sumw2()
    '''

    b_hist_prob_zoom = TH2F("back_prob_zoom","", 25, 0.95, 1., nbins_y, binmin_y, binmax_y)
    b_hist_prob_zoom.Sumw2()
    d_hist_prob_zoom = TH2F("data_prob_zoom","", 25, 0.95, 1., nbins_y, binmin_y, binmax_y)
    d_hist_prob_zoom.Sumw2()


    if SEL=="SR":
        prj_weight = "EventWeight*PUReWeight*TriggerWeight"
    else:
        prj_weight = "EventWeight*PUReWeight"

    #bins_eta = array('d',[-1.5, -1.25, -1., -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5])
    bins_eta = array('d',[-1.5, -1., -0.5, 0., 0.5, 1., 1.5])
    bins_sigprob = array('d',[0.0001,0.001,0.01,0.1,0.5,0.8,0.9,0.95,0.99,0.995,1.])

    model = keras.models.load_model('nn_inference/tagger_AK4_v3/model.h5')
    print "Cut: ", cut

    for s in back:
        for j, ss in enumerate(samples[s]['files']):
            print "TChain "+MAIN_EE+ss + ".root"
            b_chain.Add(MAIN_EE + ss + ".root")
            b_chain_tag.Add(MAIN_EE + ss + ".root")
            print "TChain "+MAIN_MM+ss + ".root"
            b_chain.Add(MAIN_MM + ss + ".root")
            b_chain_tag.Add(MAIN_MM + ss + ".root")


    for var in var_list:
        b_chain.Project("back"+var, "(Jets[0].pt/Z_pt):"+var, prj_weight+"*("+cut+" && Jets[0].sigprob<="+str(dnn)+")")
        b_chain_tag.Project("back_tag"+var, "(Jets[0].pt/Z_pt):"+var, prj_weight+"*("+cut+" && Jets[0].sigprob>"+str(dnn)+")")

    b_chain.Project("back_1d", "(Jets[0].pt/Z_pt)", prj_weight+"*("+cut+" && Jets[0].sigprob<="+str(dnn)+")")
    b_chain_tag.Project("back_tag_1d", "(Jets[0].pt/Z_pt)", prj_weight+"*("+cut+" && Jets[0].sigprob>"+str(dnn)+")")



    cut_neg_1j = "fabs(Z_mass-91.2)<10. && Z_pt>30 && abs(JetsNegative[0].eta)<1"
    b_chain.Project("back_1d_neg", "(JetsNegative[0].pt/Z_pt)", prj_weight+"*("+cut_neg_1j+" && JetsNegative[0].sigprob<="+str(dnn)+")")
    b_chain.Project("back_1d_neg_cut", "(JetsNegative[0].pt/Z_pt)", prj_weight+"*("+cut_neg_1j+" && JetsNegative[0].timeRecHitsEB>-1 && JetsNegative[0].sigprob<="+str(dnn)+")")

    #time
    cut_neg = "fabs(Z_mass-91.2)<10. && Z_pt>30 && abs(JetsNegative.eta)<1"
    b_chain.Project("b_hist_time", "JetsNegative.timeRecHitsEB", prj_weight+"*("+cut_neg+")")

    #zoom prob
    b_chain.Project("back_prob_zoom", "(Jets[0].pt/Z_pt):Jets[0].sigprob", prj_weight+"*("+cut+")")

    for s in data:
        for j, ss in enumerate(samples[s]['files']):
            if 'Muon' in ss:
                print "TChain: " + MAIN_MM+ss + ".root"
                d_chain.Add(MAIN_MM + ss + ".root")
                d_chain_tag.Add(MAIN_MM + ss + ".root")
            if 'Electron' in ss or 'EGamma' in ss:
                print MAIN_EE+ss + ".root"
                if "EGammaRun2018D-PromptReco" in ss:
                    print "too large"
                    d_chain.Add(MAIN_EE + ss + "_part1.root")
                    #d_chain.Add(MAIN_EE + ss + "_part2.root")
                else:
                    d_chain.Add(MAIN_EE + ss + ".root")
                    d_chain_tag.Add(MAIN_EE + ss + ".root")

    for var in var_list:
        d_chain.Project("data"+var, "(Jets[0].pt/Z_pt):"+var, prj_weight+"*("+cut+"  && Jets[0].sigprob<="+str(dnn)+")")
        d_chain_tag.Project("data_tag"+var, "(Jets[0].pt/Z_pt):"+var, prj_weight+"*("+cut+" && Jets[0].sigprob>"+str(dnn)+")")

    d_chain.Project("data_1d", "(Jets[0].pt/Z_pt)", prj_weight+"*("+cut+"  && Jets[0].sigprob<="+str(dnn)+")")
    d_chain_tag.Project("data_tag_1d", "(Jets[0].pt/Z_pt)", prj_weight+"*("+cut+" && Jets[0].sigprob>"+str(dnn)+")")
    d_chain.Project("d_hist_time", "JetsNegative.timeRecHitsEB", prj_weight+"*("+cut_neg+")")

    cut_neg_1j = "fabs(Z_mass-91.2)<10. && Z_pt>30 && abs(JetsNegative[0].eta)<1"
    d_chain.Project("data_1d_neg", "(JetsNegative[0].pt/Z_pt)", prj_weight+"*("+cut_neg_1j+" && JetsNegative[0].sigprob<="+str(dnn)+")")
    d_chain.Project("data_1d_neg_cut", "(JetsNegative[0].pt/Z_pt)", prj_weight+"*("+cut_neg_1j+" && JetsNegative[0].timeRecHitsEB>-1 && JetsNegative[0].sigprob<="+str(dnn)+")")

    #zoom prob
    d_chain.Project("data_prob_zoom", "(Jets[0].pt/Z_pt):Jets[0].sigprob", prj_weight+"*("+cut+")")

    #Store TH2F in a root file, will add histograms together later
    prob_file = TFile(OUT+"jet_response_prob_file"+label+".root","RECREATE")
    prob_file.cd()
    b_hist_prob_zoom.Write("b_hist_prob_zoom")
    d_hist_prob_zoom.Write("d_hist_prob_zoom")
    print "Writing "+OUT+"jet_response_prob_file"+label+".root"
    prob_file.Write()
    prob_file.Close()


    #Plot zoom prob
    can = TCanvas("can","can",900,800)
    can.SetRightMargin(0.04)
    can.SetLeftMargin(0.12)
    can.cd()
    can.SetGrid()
    leg = TLegend(0.7, 0.7, 0.95, 0.9)
    b_profX = TProfile(b_hist_prob_zoom.ProfileX("b_prof"))
    b_profX.SetMarkerStyle(21)
    b_profX.SetMarkerColor(samples[back[0]]['linecolor'])
    b_profX.SetLineColor(samples[back[0]]['linecolor'])
    b_profX.SetLineWidth(2)
        
    b_profX.Draw("PE,sames")
    b_profX.GetYaxis().SetTitle("Jet response (p_{T}^{jet_{0}}/p_{T}^{Z})")
    b_profX.GetXaxis().SetTitle("AK4 jet[0] DNN score")#("p_{T}^{Z}")

    d_profX = TProfile(d_hist_prob_zoom.ProfileX("d_prof"))
    d_profX.SetMarkerStyle(21)
    d_profX.SetMarkerColor(1)
    d_profX.SetLineColor(1)
    d_profX.SetLineWidth(2)
        
    d_profX.Draw("PE,sames")
    b_profX.SetMinimum(binmin_y)
    b_profX.SetMaximum(binmax_y)

    OUTSTRING = OUT
    OUTSTRING += "jet_response"
    OUTSTRING += "_vs_DNN_score_zoom"
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawRegion(SEL)
    leg.AddEntry(b_profX,"MC","PL")
    leg.AddEntry(d_profX,"data","PL")
    leg.Draw()
    can.Update()
    can.Print(OUTSTRING+label+'.png')
    can.Print(OUTSTRING+label+'.pdf')
    b_profX.Delete()
    d_profX.Delete()
    can.Close()
    exit()




    ##This has a very different resolution, not good without MET
    #data_file = TFile(OUT_TtoEM+"data_smear_file_CSV_0p8_all_jets.root","READ")
    #data_file.cd()
    #my_data = data_file.Get("data_CB")
    #data_file.Close()
    #my_data.SetLineColor(1)
    #my_data.SetLineStyle(2)

    #Time fit for shift/smear
    b_hist_time.Scale(1./b_hist_time.Integral())
    d_hist_time.Scale(1./d_hist_time.Integral())

    fit_back = TF1("fit_back","crystalball",5,5)
    #fit_back = TF1("fit_back","gaus",5,5)
    fit_back.SetParameter(0,0.01)
    fit_back.SetParameter(1,b_hist_time.GetMean())
    fit_back.SetParameter(2,b_hist_time.GetRMS())
    fit_back.SetParameter(3,5)
    fit_back.SetParameter(4,5)                                                                         
    b_hist_time.Fit(fit_back,"E")
    my_back = b_hist_time.GetFunction("fit_back")
    b_hist_time.GetListOfFunctions().Remove(b_hist_time.GetFunction("fit_back"))
    b_hist_time.SetLineColor(samples[back[0]]['linecolor'])
    b_hist_time.SetFillColorAlpha(samples[back[0]]['linecolor'],0.5)
    my_back.SetLineStyle(2)
    my_back.SetLineColor(2)

    fit_data = TF1("fit_data","crystalball",5,5)
    fit_data.SetParameter(0,0.01)
    fit_data.SetParameter(1,d_hist_time.GetMean())
    fit_data.SetParameter(2,d_hist_time.GetRMS())
    fit_data.SetParameter(3,2)
    fit_data.SetParameter(4,4)                                                                         
    d_hist_time.Fit(fit_data,"E")
    my_data = d_hist_time.GetFunction("fit_data")
    d_hist_time.GetListOfFunctions().Remove(d_hist_time.GetFunction("fit_data"))
    my_data.SetLineStyle(2)
    my_data.SetLineColor(1)
    d_hist_time.SetMarkerStyle(21)
    d_hist_time.SetMarkerColor(1)
    d_hist_time.SetLineColor(1)

    smear_cb = my_data.Clone("smear_cb")
    smear_cb.SetParameter(0,my_data.GetParameter(0))
    smear_cb.SetParameter(1,0)#do not modify the mean to match data!!!#my_data.GetParameter(1)-my_back.GetParameter(1)) 
    smear_cb.SetParameter(2, math.sqrt( abs(my_data.GetParameter(2)**2 - my_back.GetParameter(2)**2)) )
    smear_cb.SetParameter(3,my_data.GetParameter(3))
    smear_cb.SetParameter(4,my_data.GetParameter(4))
    smear_cb.SetLineColor(4)


    b_nTrackConstituents = np.array([])
    b_nSelectedTracks = np.array([])
    b_timeRecHitsEB = np.array([])
    b_timeSmeared = np.array([])
    b_timeShift1 = np.array([])
    b_timeShift2 = np.array([])
    b_timeShift3 = np.array([])
    b_timeShift4 = np.array([])
    b_timeShift5 = np.array([])
    b_timeShift6 = np.array([])
    b_pt = np.array([])
    b_Z_pt = np.array([])
    b_response = np.array([])
    b_eta = np.array([])
    b_eFracRecHitsEB = np.array([])
    b_nRecHitsEB = np.array([])
    b_sig1EB = np.array([])
    b_sig2EB = np.array([])
    b_ptDEB = np.array([])
    b_cHadEFrac = np.array([])
    b_nHadEFrac = np.array([])
    b_eleEFrac = np.array([])
    b_photonEFrac = np.array([])
    b_ptAllTracks = np.array([])
    b_ptAllPVTracks = np.array([])
    b_alphaMax = np.array([])
    b_betaMax = np.array([])
    b_gammaMax = np.array([])
    b_gammaMaxEM = np.array([])
    b_gammaMaxHadronic = np.array([])
    b_gammaMaxET = np.array([])
    b_minDeltaRAllTracks = np.array([])
    b_minDeltaRPVTracks = np.array([])
    b_EventWeight = np.array([])

    d_nTrackConstituents = np.array([])
    d_nSelectedTracks = np.array([])
    d_timeRecHitsEB = np.array([])
    #d_timeSmeared = np.array([])
    d_timeShift1 = np.array([])
    d_timeShift2 = np.array([])
    d_timeShift3 = np.array([])
    d_timeShift4 = np.array([])
    d_timeShift5 = np.array([])
    d_timeShift6 = np.array([])
    d_pt = np.array([])
    d_Z_pt = np.array([])
    d_response = np.array([])
    d_eta = np.array([])
    d_eFracRecHitsEB = np.array([])
    d_nRecHitsEB = np.array([])
    d_sig1EB = np.array([])
    d_sig2EB = np.array([])
    d_ptDEB = np.array([])
    d_cHadEFrac = np.array([])
    d_nHadEFrac = np.array([])
    d_eleEFrac = np.array([])
    d_photonEFrac = np.array([])
    d_ptAllTracks = np.array([])
    d_ptAllPVTracks = np.array([])
    d_alphaMax = np.array([])
    d_betaMax = np.array([])
    d_gammaMax = np.array([])
    d_gammaMaxEM = np.array([])
    d_gammaMaxHadronic = np.array([])
    d_gammaMaxET = np.array([])
    d_minDeltaRAllTracks = np.array([])
    d_minDeltaRPVTracks = np.array([])
    d_EventWeight = np.array([])


    #Do time shift
    #Back
    for b in back:
        for j, ss in enumerate(samples[b]['files']):
            for M in [MAIN_EE,MAIN_MM]:
                print "Performing gaussian smearing and time shift for ", M, ss
                b_tmp_file = TFile(M+ss+".root","READ")
                tree = b_tmp_file.Get("tree")
                tree_weight = tree.GetWeight()
                b_tmp_time = []
                b_tmp_weight = []
                b_tmp_nTrackConstituents = []
                b_tmp_nSelectedTracks = []
                b_tmp_timeRecHitsEB = []
                b_tmp_eFracRecHitsEB = []
                b_tmp_nRecHitsEB = []
                b_tmp_sig1EB = []
                b_tmp_sig2EB = []
                b_tmp_ptDEB = []
                b_tmp_cHadEFrac = []
                b_tmp_nHadEFrac = []
                b_tmp_eleEFrac = []
                b_tmp_photonEFrac = []
                b_tmp_ptAllTracks = []
                b_tmp_ptAllPVTracks = []
                b_tmp_alphaMax = []
                b_tmp_betaMax = []
                b_tmp_gammaMax = []
                b_tmp_gammaMaxEM = []
                b_tmp_gammaMaxHadronic = []
                b_tmp_gammaMaxET = []
                b_tmp_minDeltaRAllTracks = []
                b_tmp_minDeltaRPVTracks = []
                b_tmp_eta = []
                b_tmp_pt = []
                b_tmp_Z_pt = []
                b_tmp_response = []

                #for e in range(0,5000):
                for e in range(0,tree.GetEntries()):
                    tree.GetEntry(e)
                    neg_jets = getattr(tree,"JetsNegative")
                    met = getattr(tree,"MEt")
                    
                    if tree.Z_pt<30:
                        continue
                    
                    if tree.MinJetMetDPhi<=0.5:
                        continue

                    for n in range(min(1,neg_jets.size())):#(neg_jets.size()):
                        if neg_jets[n].eta<=-1:
                            continue
                        if neg_jets[n].eta>=1:
                            continue
                        if ERA=="2017":
                            if neg_jets[n].phi>=2.7: continue
                        if ERA=="2018":
                            if neg_jets[n].phi>=0.4 and neg_jets[n].phi<0.9: continue

                        time_rnd = neg_jets[n].timeRecHitsEB + smear_cb.GetRandom()
                        b_tmp_weight.append(tree.EventWeight * tree.PUReWeight * tree_weight)
                        b_tmp_time.append(time_rnd)
                        b_tmp_nTrackConstituents.append(neg_jets[n].nTrackConstituents)
                        b_tmp_nSelectedTracks.append(neg_jets[n].nSelectedTracks)
                        b_tmp_timeRecHitsEB.append(neg_jets[n].timeRecHitsEB)
                        b_tmp_eFracRecHitsEB.append(neg_jets[n].eFracRecHitsEB)
                        b_tmp_nRecHitsEB.append(neg_jets[n].nRecHitsEB)
                        b_tmp_sig1EB.append(neg_jets[n].sig1EB)
                        b_tmp_sig2EB.append(neg_jets[n].sig2EB)
                        b_tmp_ptDEB.append(neg_jets[n].ptDEB)
                        b_tmp_cHadEFrac.append(neg_jets[n].cHadEFrac)
                        b_tmp_nHadEFrac.append(neg_jets[n].nHadEFrac)
                        b_tmp_eleEFrac.append(neg_jets[n].eleEFrac)
                        b_tmp_photonEFrac.append(neg_jets[n].photonEFrac)
                        b_tmp_ptAllTracks.append(neg_jets[n].ptAllTracks)
                        b_tmp_ptAllPVTracks.append(neg_jets[n].ptAllPVTracks)
                        b_tmp_alphaMax.append(neg_jets[n].alphaMax)
                        b_tmp_betaMax.append(neg_jets[n].betaMax)
                        b_tmp_gammaMax.append(neg_jets[n].gammaMax)
                        b_tmp_gammaMaxEM.append(neg_jets[n].gammaMaxEM)
                        b_tmp_gammaMaxHadronic.append(neg_jets[n].gammaMaxHadronic)
                        b_tmp_gammaMaxET.append(neg_jets[n].gammaMaxET)
                        b_tmp_minDeltaRAllTracks.append(neg_jets[n].minDeltaRAllTracks)
                        b_tmp_minDeltaRPVTracks.append(neg_jets[n].minDeltaRPVTracks)
                        b_tmp_eta.append(neg_jets[n].eta)
                        b_tmp_pt.append(neg_jets[n].pt)
                        b_tmp_response.append(neg_jets[n].pt/tree.Z_pt)
                        b_tmp_Z_pt.append(tree.Z_pt)

                b_timeSmeared = np.concatenate((b_timeSmeared, np.array(b_tmp_time)))
                b_EventWeight = np.concatenate((b_EventWeight, np.array(b_tmp_weight)))
                b_nTrackConstituents = np.concatenate(( b_nTrackConstituents, b_tmp_nTrackConstituents ))
                b_nSelectedTracks = np.concatenate(( b_nSelectedTracks, b_tmp_nSelectedTracks))
                b_timeRecHitsEB = np.concatenate(( b_timeRecHitsEB, b_tmp_timeRecHitsEB))
                b_eFracRecHitsEB = np.concatenate(( b_eFracRecHitsEB, b_tmp_eFracRecHitsEB))
                b_nRecHitsEB = np.concatenate(( b_nRecHitsEB, b_tmp_nRecHitsEB))
                b_sig1EB = np.concatenate(( b_sig1EB, b_tmp_sig1EB))
                b_sig2EB = np.concatenate(( b_sig2EB, b_tmp_sig2EB))
                b_ptDEB = np.concatenate(( b_ptDEB, b_tmp_ptDEB))
                b_cHadEFrac = np.concatenate(( b_cHadEFrac, b_tmp_cHadEFrac))
                b_nHadEFrac = np.concatenate(( b_nHadEFrac, b_tmp_nHadEFrac))
                b_eleEFrac = np.concatenate(( b_eleEFrac, b_tmp_eleEFrac))
                b_photonEFrac = np.concatenate(( b_photonEFrac, b_tmp_photonEFrac))
                b_ptAllTracks = np.concatenate(( b_ptAllTracks, b_tmp_ptAllTracks))
                b_ptAllPVTracks = np.concatenate(( b_ptAllPVTracks, b_tmp_ptAllPVTracks))
                b_alphaMax = np.concatenate(( b_alphaMax, b_tmp_alphaMax))
                b_betaMax = np.concatenate(( b_betaMax, b_tmp_betaMax))
                b_gammaMax = np.concatenate(( b_gammaMax, b_tmp_gammaMax))
                b_gammaMaxEM = np.concatenate(( b_gammaMaxEM, b_tmp_gammaMaxEM))
                b_gammaMaxHadronic = np.concatenate(( b_gammaMaxHadronic, b_tmp_gammaMaxHadronic))
                b_gammaMaxET = np.concatenate(( b_gammaMaxET, b_tmp_gammaMaxET))
                b_minDeltaRAllTracks = np.concatenate(( b_minDeltaRAllTracks, b_tmp_minDeltaRAllTracks))
                b_minDeltaRPVTracks = np.concatenate(( b_minDeltaRPVTracks, b_tmp_minDeltaRPVTracks))
                b_eta = np.concatenate(( b_eta,b_tmp_eta ))
                b_pt = np.concatenate(( b_pt,b_tmp_pt ))
                b_Z_pt = np.concatenate(( b_Z_pt,b_tmp_Z_pt ))
                b_response = np.concatenate(( b_response,b_tmp_response ))

    b_timeShift1 = b_timeSmeared+1
    b_timeShift2 = b_timeSmeared+2
    b_timeShift3 = b_timeSmeared+3
    b_timeShift4 = b_timeSmeared+4
    b_timeShift5 = b_timeSmeared+5
    b_timeShift6 = b_timeSmeared+6

    b_dat_list = [
        b_nTrackConstituents,
        b_nSelectedTracks,
        b_timeRecHitsEB,
        b_eFracRecHitsEB,
        b_nRecHitsEB,
        b_sig1EB,
        b_sig2EB,
        b_ptDEB,
        b_cHadEFrac,
        b_nHadEFrac,
        b_eleEFrac,
        b_photonEFrac,
        b_ptAllTracks,
        b_ptAllPVTracks,
        b_alphaMax,
        b_betaMax,
        b_gammaMax,
        b_gammaMaxEM,
        b_gammaMaxHadronic,
        b_gammaMaxET,
        b_minDeltaRAllTracks,
        b_minDeltaRPVTracks,
    ]

    b_dat_list_smeared = [
        b_nTrackConstituents,
        b_nSelectedTracks,
        b_timeSmeared,
        b_eFracRecHitsEB,
        b_nRecHitsEB,
        b_sig1EB,
        b_sig2EB,
        b_ptDEB,
        b_cHadEFrac,
        b_nHadEFrac,
        b_eleEFrac,
        b_photonEFrac,
        b_ptAllTracks,
        b_ptAllPVTracks,
        b_alphaMax,
        b_betaMax,
        b_gammaMax,
        b_gammaMaxEM,
        b_gammaMaxHadronic,
        b_gammaMaxET,
        b_minDeltaRAllTracks,
        b_minDeltaRPVTracks,
    ]

    b_dat_list_shift1 = [
        b_nTrackConstituents,
        b_nSelectedTracks,
        b_timeShift1,
        b_eFracRecHitsEB,
        b_nRecHitsEB,
        b_sig1EB,
        b_sig2EB,
        b_ptDEB,
        b_cHadEFrac,
        b_nHadEFrac,
        b_eleEFrac,
        b_photonEFrac,
        b_ptAllTracks,
        b_ptAllPVTracks,
        b_alphaMax,
        b_betaMax,
        b_gammaMax,
        b_gammaMaxEM,
        b_gammaMaxHadronic,
        b_gammaMaxET,
        b_minDeltaRAllTracks,
        b_minDeltaRPVTracks,
    ]

    b_dat_list_shift2 = [
        b_nTrackConstituents,
        b_nSelectedTracks,
        b_timeShift2,
        b_eFracRecHitsEB,
        b_nRecHitsEB,
        b_sig1EB,
        b_sig2EB,
        b_ptDEB,
        b_cHadEFrac,
        b_nHadEFrac,
        b_eleEFrac,
        b_photonEFrac,
        b_ptAllTracks,
        b_ptAllPVTracks,
        b_alphaMax,
        b_betaMax,
        b_gammaMax,
        b_gammaMaxEM,
        b_gammaMaxHadronic,
        b_gammaMaxET,
        b_minDeltaRAllTracks,
        b_minDeltaRPVTracks,
    ]

    b_dat_list_shift3 = [
        b_nTrackConstituents,
        b_nSelectedTracks,
        b_timeShift3,
        b_eFracRecHitsEB,
        b_nRecHitsEB,
        b_sig1EB,
        b_sig2EB,
        b_ptDEB,
        b_cHadEFrac,
        b_nHadEFrac,
        b_eleEFrac,
        b_photonEFrac,
        b_ptAllTracks,
        b_ptAllPVTracks,
        b_alphaMax,
        b_betaMax,
        b_gammaMax,
        b_gammaMaxEM,
        b_gammaMaxHadronic,
        b_gammaMaxET,
        b_minDeltaRAllTracks,
        b_minDeltaRPVTracks,
    ]

    b_dat_list_shift4 = [
        b_nTrackConstituents,
        b_nSelectedTracks,
        b_timeShift4,
        b_eFracRecHitsEB,
        b_nRecHitsEB,
        b_sig1EB,
        b_sig2EB,
        b_ptDEB,
        b_cHadEFrac,
        b_nHadEFrac,
        b_eleEFrac,
        b_photonEFrac,
        b_ptAllTracks,
        b_ptAllPVTracks,
        b_alphaMax,
        b_betaMax,
        b_gammaMax,
        b_gammaMaxEM,
        b_gammaMaxHadronic,
        b_gammaMaxET,
        b_minDeltaRAllTracks,
        b_minDeltaRPVTracks,
    ]

    b_dat_list_shift5 = [
        b_nTrackConstituents,
        b_nSelectedTracks,
        b_timeShift5,
        b_eFracRecHitsEB,
        b_nRecHitsEB,
        b_sig1EB,
        b_sig2EB,
        b_ptDEB,
        b_cHadEFrac,
        b_nHadEFrac,
        b_eleEFrac,
        b_photonEFrac,
        b_ptAllTracks,
        b_ptAllPVTracks,
        b_alphaMax,
        b_betaMax,
        b_gammaMax,
        b_gammaMaxEM,
        b_gammaMaxHadronic,
        b_gammaMaxET,
        b_minDeltaRAllTracks,
        b_minDeltaRPVTracks,
    ]

    b_dat_list_shift6 = [
        b_nTrackConstituents,
        b_nSelectedTracks,
        b_timeShift6,
        b_eFracRecHitsEB,
        b_nRecHitsEB,
        b_sig1EB,
        b_sig2EB,
        b_ptDEB,
        b_cHadEFrac,
        b_nHadEFrac,
        b_eleEFrac,
        b_photonEFrac,
        b_ptAllTracks,
        b_ptAllPVTracks,
        b_alphaMax,
        b_betaMax,
        b_gammaMax,
        b_gammaMaxEM,
        b_gammaMaxHadronic,
        b_gammaMaxET,
        b_minDeltaRAllTracks,
        b_minDeltaRPVTracks,
    ]

    b_X_smear = np.transpose(np.stack((b_dat_list_smeared)))
    b_X_shift1 = np.transpose(np.stack((b_dat_list_shift1)))
    b_X_shift2 = np.transpose(np.stack((b_dat_list_shift2)))
    b_X_shift3 = np.transpose(np.stack((b_dat_list_shift3)))
    b_X_shift4 = np.transpose(np.stack((b_dat_list_shift4)))
    b_X_shift5 = np.transpose(np.stack((b_dat_list_shift5)))
    b_X_shift6 = np.transpose(np.stack((b_dat_list_shift6)))

    b_probs_smear = model.predict(b_X_smear)[:,1]
    b_probs_shift1 = model.predict(b_X_shift1)[:,1]
    b_probs_shift2 = model.predict(b_X_shift2)[:,1]
    b_probs_shift3 = model.predict(b_X_shift3)[:,1]
    b_probs_shift4 = model.predict(b_X_shift4)[:,1]
    b_probs_shift5 = model.predict(b_X_shift5)[:,1]
    b_probs_shift6 = model.predict(b_X_shift6)[:,1]

    m_b_time_smear  = (b_timeSmeared  > -1 )
    m_b_time_shift1 = (b_timeShift1   > -1)
    m_b_time_shift2 = (b_timeShift2   > -1)
    m_b_time_shift3 = (b_timeShift3   > -1)
    m_b_time_shift4 = (b_timeShift4   > -1)
    m_b_time_shift5 = (b_timeShift5   > -1)
    m_b_time_shift6 = (b_timeShift6   > -1)

    m_b_probs_smear  = (b_probs_smear  > 0.996)
    m_b_probs_shift1 = (b_probs_shift1 > 0.996)
    m_b_probs_shift2 = (b_probs_shift2 > 0.996)
    m_b_probs_shift3 = (b_probs_shift3 > 0.996)
    m_b_probs_shift4 = (b_probs_shift4 > 0.996)
    m_b_probs_shift5 = (b_probs_shift5 > 0.996)
    m_b_probs_shift6 = (b_probs_shift6 > 0.996)

    m_b_and_smear  = np.logical_and(m_b_time_smear,m_b_probs_smear)
    m_b_and_shift1 = np.logical_and(m_b_time_shift1,m_b_probs_shift1)
    m_b_and_shift2 = np.logical_and(m_b_time_shift2,m_b_probs_shift2)
    m_b_and_shift3 = np.logical_and(m_b_time_shift3,m_b_probs_shift3)
    m_b_and_shift4 = np.logical_and(m_b_time_shift4,m_b_probs_shift4)
    m_b_and_shift5 = np.logical_and(m_b_time_shift5,m_b_probs_shift5)
    m_b_and_shift6 = np.logical_and(m_b_time_shift6,m_b_probs_shift6)

    #root_numpy.fill_hist(b_h_smeared,  b_timeSmeared[m_b_time_smear], b_EventWeight[m_b_time_smear])
    root_numpy.fill_hist(b_hist_time_smeared,  b_timeSmeared, b_EventWeight)
    b_hist_time_smeared.Scale(1./b_hist_time_smeared.Integral())
    b_hist_time_smeared.SetLineColor(801)
    b_hist_time_smeared.SetFillColorAlpha(801,0.3)

    #Here draw time fit as cross-check
    can = TCanvas("can","can",900,800)
    can.SetRightMargin(0.05)
    can.cd()
    leg = TLegend(0.65, 0.6, 1., 0.9)
    b_hist_time.Draw("HISTO")
    d_hist_time.Draw("PE,sames")
    b_hist_time_smeared.Draw("HISTO,sames")
    my_back.Draw("L,sames")
    my_data.Draw("L,sames")
    OUTSTRING = OUT
    OUTSTRING += "jet_time"
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawRegion(SEL)
    #leg.Draw()
    can.Update()
    can.Print(OUTSTRING+'.png')
    can.Print(OUTSTRING+'.pdf')
    can.SetLogy()
    can.Update()
    can.Print(OUTSTRING+'_log.png')
    can.Print(OUTSTRING+'_log.pdf')
    can.Close()

    #Data
    for b in data:
        for j, ss in enumerate(samples[b]['files']):
            if 'Muon' in ss:
                print "Performing time shift for ", ss
                d_tmp_file = TFile(MAIN_MM+ss+".root","READ")
            
            if 'Electron' in ss or 'EGamma' in ss:
                if "EGammaRun2018D-PromptReco" in ss:
                    print "Performing time shift for ", ss, "_part1"
                    print "too large"
                    d_tmp_file = TFile(MAIN_EE+ss+"_part1.root","READ")
                else:
                    print "Performing time shift for ", ss
                    d_tmp_file = TFile(MAIN_EE+ss+".root","READ")

            
            #else:                                                                                                        
            #    d_tmp_file = TFile(MAIN+ss+".root","READ")                                                               
            tree = d_tmp_file.Get("tree")

            tree_weight = tree.GetWeight()
            d_tmp_time = []
            d_tmp_weight = []
            d_tmp_nTrackConstituents = []
            d_tmp_nSelectedTracks = []
            d_tmp_timeRecHitsEB = []

            d_tmp_eFracRecHitsEB = []
            d_tmp_nRecHitsEB = []
            d_tmp_sig1EB = []
            d_tmp_sig2EB = []
            d_tmp_ptDEB = []
            d_tmp_cHadEFrac = []
            d_tmp_nHadEFrac = []
            d_tmp_eleEFrac = []
            d_tmp_photonEFrac = []
            d_tmp_ptAllTracks = []
            d_tmp_ptAllPVTracks = []
            d_tmp_alphaMax = []
            d_tmp_betaMax = []
            d_tmp_gammaMax = []
            d_tmp_gammaMaxEM = []
            d_tmp_gammaMaxHadronic = []
            d_tmp_gammaMaxET = []
            d_tmp_minDeltaRAllTracks = []
            d_tmp_minDeltaRPVTracks = []
            d_tmp_eta = []
            d_tmp_pt = []
            d_tmp_Z_pt = []
            d_tmp_response = []
                        
            #for e in range(0,5000):
            for e in range(0,tree.GetEntries()):
                tree.GetEntry(e)
                neg_jets = getattr(tree,"JetsNegative")
                met = getattr(tree,"MEt")

                if tree.Z_pt<30:
                    continue

                if tree.MinJetMetDPhi<=0.5:
                    continue

                for n in range(min(1,neg_jets.size())):#(neg_jets.size()):
                    if neg_jets[n].eta<=-1:
                        continue
                    if neg_jets[n].eta>=1:
                        continue
                    if ERA=="2017":
                        if neg_jets[n].phi>=2.7: continue
                    if ERA=="2018":
                        if neg_jets[n].phi>=0.4 and neg_jets[n].phi<0.9: continue
                    d_tmp_weight.append(tree.EventWeight * tree.PUReWeight * tree_weight)
                    d_tmp_nTrackConstituents.append(neg_jets[n].nTrackConstituents)
                    d_tmp_nSelectedTracks.append(neg_jets[n].nSelectedTracks)
                    d_tmp_timeRecHitsEB.append(neg_jets[n].timeRecHitsEB)
                    d_tmp_eFracRecHitsEB.append(neg_jets[n].eFracRecHitsEB)
                    d_tmp_nRecHitsEB.append(neg_jets[n].nRecHitsEB)
                    d_tmp_sig1EB.append(neg_jets[n].sig1EB)
                    d_tmp_sig2EB.append(neg_jets[n].sig2EB)
                    d_tmp_ptDEB.append(neg_jets[n].ptDEB)
                    d_tmp_cHadEFrac.append(neg_jets[n].cHadEFrac)
                    d_tmp_nHadEFrac.append(neg_jets[n].nHadEFrac)
                    d_tmp_eleEFrac.append(neg_jets[n].eleEFrac)
                    d_tmp_photonEFrac.append(neg_jets[n].photonEFrac)
                    d_tmp_ptAllTracks.append(neg_jets[n].ptAllTracks)
                    d_tmp_ptAllPVTracks.append(neg_jets[n].ptAllPVTracks)
                    d_tmp_alphaMax.append(neg_jets[n].alphaMax)
                    d_tmp_betaMax.append(neg_jets[n].betaMax)
                    d_tmp_gammaMax.append(neg_jets[n].gammaMax)
                    d_tmp_gammaMaxEM.append(neg_jets[n].gammaMaxEM)
                    d_tmp_gammaMaxHadronic.append(neg_jets[n].gammaMaxHadronic)
                    d_tmp_gammaMaxET.append(neg_jets[n].gammaMaxET)
                    d_tmp_minDeltaRAllTracks.append(neg_jets[n].minDeltaRAllTracks)
                    d_tmp_minDeltaRPVTracks.append(neg_jets[n].minDeltaRPVTracks)
                    d_tmp_eta.append(neg_jets[n].eta)
                    d_tmp_pt.append(neg_jets[n].pt)
                    d_tmp_Z_pt.append(tree.Z_pt)
                    d_tmp_response.append(neg_jets[n].pt/tree.Z_pt)


            d_EventWeight = np.concatenate((d_EventWeight, np.array(d_tmp_weight)))
            d_nTrackConstituents = np.concatenate(( d_nTrackConstituents, d_tmp_nTrackConstituents ))
            d_nSelectedTracks = np.concatenate(( d_nSelectedTracks, d_tmp_nSelectedTracks))
            d_timeRecHitsEB = np.concatenate(( d_timeRecHitsEB, d_tmp_timeRecHitsEB))
            d_eFracRecHitsEB = np.concatenate(( d_eFracRecHitsEB, d_tmp_eFracRecHitsEB))
            d_nRecHitsEB = np.concatenate(( d_nRecHitsEB, d_tmp_nRecHitsEB))
            d_sig1EB = np.concatenate(( d_sig1EB, d_tmp_sig1EB))
            d_sig2EB = np.concatenate(( d_sig2EB, d_tmp_sig2EB))
            d_ptDEB = np.concatenate(( d_ptDEB, d_tmp_ptDEB))
            d_cHadEFrac = np.concatenate(( d_cHadEFrac, d_tmp_cHadEFrac))
            d_nHadEFrac = np.concatenate(( d_nHadEFrac, d_tmp_nHadEFrac))
            d_eleEFrac = np.concatenate(( d_eleEFrac, d_tmp_eleEFrac))
            d_photonEFrac = np.concatenate(( d_photonEFrac, d_tmp_photonEFrac))
            d_ptAllTracks = np.concatenate(( d_ptAllTracks, d_tmp_ptAllTracks))
            d_ptAllPVTracks = np.concatenate(( d_ptAllPVTracks, d_tmp_ptAllPVTracks))
            d_alphaMax = np.concatenate(( d_alphaMax, d_tmp_alphaMax))
            d_betaMax = np.concatenate(( d_betaMax, d_tmp_betaMax))
            d_gammaMax = np.concatenate(( d_gammaMax, d_tmp_gammaMax))
            d_gammaMaxEM = np.concatenate(( d_gammaMaxEM, d_tmp_gammaMaxEM))
            d_gammaMaxHadronic = np.concatenate(( d_gammaMaxHadronic, d_tmp_gammaMaxHadronic))
            d_gammaMaxET = np.concatenate(( d_gammaMaxET, d_tmp_gammaMaxET))
            d_minDeltaRAllTracks = np.concatenate(( d_minDeltaRAllTracks, d_tmp_minDeltaRAllTracks))
            d_minDeltaRPVTracks = np.concatenate(( d_minDeltaRPVTracks, d_tmp_minDeltaRPVTracks))
            d_eta = np.concatenate(( d_eta,d_tmp_eta ))
            d_pt = np.concatenate(( d_pt,d_tmp_pt ))
            d_Z_pt = np.concatenate(( d_Z_pt,d_tmp_Z_pt ))
            d_response = np.concatenate(( d_response,d_tmp_response ))

    d_timeShift1 = d_timeRecHitsEB+1
    d_timeShift2 = d_timeRecHitsEB+2
    d_timeShift3 = d_timeRecHitsEB+3
    d_timeShift4 = d_timeRecHitsEB+4
    d_timeShift5 = d_timeRecHitsEB+5
    d_timeShift6 = d_timeRecHitsEB+6

    d_dat_list = [
        d_nTrackConstituents,
        d_nSelectedTracks,
        d_timeRecHitsEB,
        d_eFracRecHitsEB,
        d_nRecHitsEB,
        d_sig1EB,
        d_sig2EB,
        d_ptDEB,
        d_cHadEFrac,
        d_nHadEFrac,
        d_eleEFrac,
        d_photonEFrac,
        d_ptAllTracks,
        d_ptAllPVTracks,
        d_alphaMax,
        d_betaMax,
        d_gammaMax,
        d_gammaMaxEM,
        d_gammaMaxHadronic,
        d_gammaMaxET,
        d_minDeltaRAllTracks,
        d_minDeltaRPVTracks,
    ]

    d_dat_list_shift1 = [
        d_nTrackConstituents,
        d_nSelectedTracks,
        d_timeShift1,
        d_eFracRecHitsEB,
        d_nRecHitsEB,
        d_sig1EB,
        d_sig2EB,
        d_ptDEB,
        d_cHadEFrac,
        d_nHadEFrac,
        d_eleEFrac,
        d_photonEFrac,
        d_ptAllTracks,
        d_ptAllPVTracks,
        d_alphaMax,
        d_betaMax,
        d_gammaMax,
        d_gammaMaxEM,
        d_gammaMaxHadronic,
        d_gammaMaxET,
        d_minDeltaRAllTracks,
        d_minDeltaRPVTracks,
    ]
    d_dat_list_shift2 = [
        d_nTrackConstituents,
        d_nSelectedTracks,
        d_timeShift2,
        d_eFracRecHitsEB,
        d_nRecHitsEB,
        d_sig1EB,
        d_sig2EB,
        d_ptDEB,
        d_cHadEFrac,
        d_nHadEFrac,
        d_eleEFrac,
        d_photonEFrac,
        d_ptAllTracks,
        d_ptAllPVTracks,
        d_alphaMax,
        d_betaMax,
        d_gammaMax,
        d_gammaMaxEM,
        d_gammaMaxHadronic,
        d_gammaMaxET,
        d_minDeltaRAllTracks,
        d_minDeltaRPVTracks,
    ]
    d_dat_list_shift3 = [
        d_nTrackConstituents,
        d_nSelectedTracks,
        d_timeShift3,
        d_eFracRecHitsEB,
        d_nRecHitsEB,
        d_sig1EB,
        d_sig2EB,
        d_ptDEB,
        d_cHadEFrac,
        d_nHadEFrac,
        d_eleEFrac,
        d_photonEFrac,
        d_ptAllTracks,
        d_ptAllPVTracks,
        d_alphaMax,
        d_betaMax,
        d_gammaMax,
        d_gammaMaxEM,
        d_gammaMaxHadronic,
        d_gammaMaxET,
        d_minDeltaRAllTracks,
        d_minDeltaRPVTracks,
    ]
    d_dat_list_shift4 = [
        d_nTrackConstituents,
        d_nSelectedTracks,
        d_timeShift4,
        d_eFracRecHitsEB,
        d_nRecHitsEB,
        d_sig1EB,
        d_sig2EB,
        d_ptDEB,
        d_cHadEFrac,
        d_nHadEFrac,
        d_eleEFrac,
        d_photonEFrac,
        d_ptAllTracks,
        d_ptAllPVTracks,
        d_alphaMax,
        d_betaMax,
        d_gammaMax,
        d_gammaMaxEM,
        d_gammaMaxHadronic,
        d_gammaMaxET,
        d_minDeltaRAllTracks,
        d_minDeltaRPVTracks,
    ]
    d_dat_list_shift5 = [
        d_nTrackConstituents,
        d_nSelectedTracks,
        d_timeShift5,
        d_eFracRecHitsEB,
        d_nRecHitsEB,
        d_sig1EB,
        d_sig2EB,
        d_ptDEB,
        d_cHadEFrac,
        d_nHadEFrac,
        d_eleEFrac,
        d_photonEFrac,
        d_ptAllTracks,
        d_ptAllPVTracks,
        d_alphaMax,
        d_betaMax,
        d_gammaMax,
        d_gammaMaxEM,
        d_gammaMaxHadronic,
        d_gammaMaxET,
        d_minDeltaRAllTracks,
        d_minDeltaRPVTracks,
    ]
    d_dat_list_shift6 = [
        d_nTrackConstituents,
        d_nSelectedTracks,
        d_timeShift6,
        d_eFracRecHitsEB,
        d_nRecHitsEB,
        d_sig1EB,
        d_sig2EB,
        d_ptDEB,
        d_cHadEFrac,
        d_nHadEFrac,
        d_eleEFrac,
        d_photonEFrac,
        d_ptAllTracks,
        d_ptAllPVTracks,
        d_alphaMax,
        d_betaMax,
        d_gammaMax,
        d_gammaMaxEM,
        d_gammaMaxHadronic,
        d_gammaMaxET,
        d_minDeltaRAllTracks,
        d_minDeltaRPVTracks,
    ]

    d_X = np.transpose(np.stack((d_dat_list)))
    d_X_shift1 = np.transpose(np.stack((d_dat_list_shift1)))
    d_X_shift2 = np.transpose(np.stack((d_dat_list_shift2)))
    d_X_shift3 = np.transpose(np.stack((d_dat_list_shift3)))
    d_X_shift4 = np.transpose(np.stack((d_dat_list_shift4)))
    d_X_shift5 = np.transpose(np.stack((d_dat_list_shift5)))
    d_X_shift6 = np.transpose(np.stack((d_dat_list_shift6)))
    d_probs = model.predict(d_X)[:,1]
    d_probs_shift1 = model.predict(d_X_shift1)[:,1]
    d_probs_shift2 = model.predict(d_X_shift2)[:,1]
    d_probs_shift3 = model.predict(d_X_shift3)[:,1]
    d_probs_shift4 = model.predict(d_X_shift4)[:,1]
    d_probs_shift5 = model.predict(d_X_shift5)[:,1]
    d_probs_shift6 = model.predict(d_X_shift6)[:,1]

    m_d_time        = (d_timeRecHitsEB > -1 )
    m_d_time_shift1 = (d_timeShift1    > -1)
    m_d_time_shift2 = (d_timeShift2    > -1)
    m_d_time_shift3 = (d_timeShift3    > -1)
    m_d_time_shift4 = (d_timeShift4    > -1)
    m_d_time_shift5 = (d_timeShift5    > -1)
    m_d_time_shift6 = (d_timeShift6    > -1)

    m_d_probs        = (d_probs  > 0.996 )
    m_d_probs_shift1 = (d_probs_shift1 > 0.996)
    m_d_probs_shift2 = (d_probs_shift2 > 0.996)
    m_d_probs_shift3 = (d_probs_shift3 > 0.996)
    m_d_probs_shift4 = (d_probs_shift4 > 0.996)
    m_d_probs_shift5 = (d_probs_shift5 > 0.996)
    m_d_probs_shift6 = (d_probs_shift6 > 0.996)

    m_d_and        = np.logical_and(m_d_time       ,m_d_probs)
    m_d_and_shift1 = np.logical_and(m_d_time_shift1,m_d_probs_shift1)
    m_d_and_shift2 = np.logical_and(m_d_time_shift2,m_d_probs_shift2)
    m_d_and_shift3 = np.logical_and(m_d_time_shift3,m_d_probs_shift3)
    m_d_and_shift4 = np.logical_and(m_d_time_shift4,m_d_probs_shift4)
    m_d_and_shift5 = np.logical_and(m_d_time_shift5,m_d_probs_shift5)
    m_d_and_shift6 = np.logical_and(m_d_time_shift6,m_d_probs_shift6)

    #Fill 2D plots of shifted stuff
    #b_timeShift1 and b_response
    root_numpy.fill_hist(b_hist_tag_1d_shift1,b_response[m_b_and_shift1],b_EventWeight[m_b_and_shift1])
    root_numpy.fill_hist(d_hist_tag_1d_shift1,d_response[m_d_and_shift1],d_EventWeight[m_d_and_shift1])

    root_numpy.fill_hist(b_hist_tag_1d_shift2,b_response[m_b_and_shift2],b_EventWeight[m_b_and_shift2])
    root_numpy.fill_hist(d_hist_tag_1d_shift2,d_response[m_d_and_shift2],d_EventWeight[m_d_and_shift2])

    #Here loop over vars:
    for var in var_list:
        if var=="Z_pt":
            b_st = np.transpose(np.stack((b_Z_pt[m_b_time_shift1],b_response[m_b_time_shift1])))
            root_numpy.fill_hist(b_hist_untag_shift1[var], b_st, b_EventWeight[m_b_time_shift1] )
            d_st = np.transpose(np.stack((d_Z_pt[m_d_time_shift1],d_response[m_d_time_shift1])))
            root_numpy.fill_hist(d_hist_untag_shift1[var], d_st, d_EventWeight[m_d_time_shift1] )

            b_st = np.transpose(np.stack((b_Z_pt[m_b_and_shift1],b_response[m_b_and_shift1])))
            root_numpy.fill_hist(b_hist_tag_shift1[var], b_st, b_EventWeight[m_b_and_shift1] )
            d_st = np.transpose(np.stack((d_Z_pt[m_d_and_shift1],d_response[m_d_and_shift1])))
            root_numpy.fill_hist(d_hist_tag_shift1[var], d_st, d_EventWeight[m_d_and_shift1] )

            b_st = np.transpose(np.stack((b_Z_pt[m_b_and_shift2],b_response[m_b_and_shift2])))
            root_numpy.fill_hist(b_hist_tag_shift2[var], b_st, b_EventWeight[m_b_and_shift2] )
            d_st = np.transpose(np.stack((d_Z_pt[m_d_and_shift2],d_response[m_d_and_shift2])))
            root_numpy.fill_hist(d_hist_tag_shift2[var], d_st, d_EventWeight[m_d_and_shift2] )

        if var=="Jets[0].timeRecHitsEB":
            b_st = np.transpose(np.stack((b_timeShift1[m_b_time_shift1],b_response[m_b_time_shift1])))
            root_numpy.fill_hist(b_hist_untag_shift1[var], b_st, b_EventWeight[m_b_time_shift1] )
            d_st = np.transpose(np.stack((d_timeShift1[m_d_time_shift1],d_response[m_d_time_shift1])))
            root_numpy.fill_hist(d_hist_untag_shift1[var], d_st, d_EventWeight[m_d_time_shift1] )

            b_st = np.transpose(np.stack((b_timeShift1[m_b_and_shift1],b_response[m_b_and_shift1])))
            root_numpy.fill_hist(b_hist_tag_shift1[var], b_st, b_EventWeight[m_b_and_shift1] )
            d_st = np.transpose(np.stack((d_timeShift1[m_d_and_shift1],d_response[m_d_and_shift1])))
            root_numpy.fill_hist(d_hist_tag_shift1[var], d_st, d_EventWeight[m_d_and_shift1] )

            b_st = np.transpose(np.stack((b_timeShift2[m_b_and_shift2],b_response[m_b_and_shift2])))
            root_numpy.fill_hist(b_hist_tag_shift2[var], b_st, b_EventWeight[m_b_and_shift2] )
            d_st = np.transpose(np.stack((d_timeShift2[m_d_and_shift2],d_response[m_d_and_shift2])))
            root_numpy.fill_hist(d_hist_tag_shift2[var], d_st, d_EventWeight[m_d_and_shift2] )

        if var=="Jets[0].sigprob":
            b_st = np.transpose(np.stack((b_probs_shift1[m_b_time_shift1],b_response[m_b_time_shift1])))
            root_numpy.fill_hist(b_hist_untag_shift1[var], b_st, b_EventWeight[m_b_time_shift1] )
            d_st = np.transpose(np.stack((d_probs_shift1[m_d_time_shift1],d_response[m_d_time_shift1])))
            root_numpy.fill_hist(d_hist_untag_shift1[var], d_st, d_EventWeight[m_d_time_shift1] )

            b_st = np.transpose(np.stack((b_probs_shift1[m_b_and_shift1],b_response[m_b_and_shift1])))
            root_numpy.fill_hist(b_hist_tag_shift1[var], b_st, b_EventWeight[m_b_and_shift1] )
            d_st = np.transpose(np.stack((d_probs_shift1[m_d_and_shift1],d_response[m_d_and_shift1])))
            root_numpy.fill_hist(d_hist_tag_shift1[var], d_st, d_EventWeight[m_d_and_shift1] )
            
            b_st = np.transpose(np.stack((b_probs_shift2[m_b_and_shift2],b_response[m_b_and_shift2])))
            root_numpy.fill_hist(b_hist_tag_shift2[var], b_st, b_EventWeight[m_b_and_shift2] )
            d_st = np.transpose(np.stack((d_probs_shift2[m_d_and_shift2],d_response[m_d_and_shift2])))
            root_numpy.fill_hist(d_hist_tag_shift2[var], d_st, d_EventWeight[m_d_and_shift2] )


    #There is one problem. So far we looked at the response of the leading jet after time cut. If we do a time shift, that changes.
    #If we store only leading jets and we do the time cut afterwards, we lose events. We can perhaps store the second leading if the time of the leading is negative. In this way we recover all the leading jets.
    #A bit tricky to implement. Can instead store only the leading negative jets and simply discard events where the leading is negative. Compare resolution w.r.t. normal jets.
    #Try to plot:
    #1. Resolution of Jets[0]
    #2. Resolution of JetsNegative[0]
    #3. Resolution of JetsNegative[0] after time cut
    ##----> differences seem negligible!

    #Plot response
    #One canvas per var
    for var in var_list:
        can = TCanvas("can","can",900,800)
        can.SetRightMargin(0.04)
        can.SetLeftMargin(0.12)
        can.cd()
        can.SetGrid()
        leg = TLegend(0.7, 0.7, 0.95, 0.9)
        b_profX               = TProfile(b_hist[var].ProfileX("b_prof"))
        b_profX_tag           = TProfile(b_hist_tag[var].ProfileX("b_prof_tag"))
        b_profX_untag_shift1 = TProfile(b_hist_untag_shift1[var].ProfileX("b_profX_untag_shift1"))
        b_profX_tag_shift1   = TProfile(b_hist_tag_shift1[var].ProfileX("b_profX_tag_shift1"))
        b_profX_tag_shift2   = TProfile(b_hist_tag_shift2[var].ProfileX("b_profX_tag_shift2"))
        b_profX.SetMarkerStyle(21)
        b_profX.SetMarkerColor(samples[back[0]]['linecolor'])
        b_profX.SetLineColor(samples[back[0]]['linecolor'])
        b_profX.SetLineWidth(2)
        
        b_profX_tag_shift1.SetMarkerStyle(25)
        b_profX_tag_shift1.SetMarkerColor(samples[back[0]]['linecolor']+2)
        b_profX_tag_shift1.SetLineColor(samples[back[0]]['linecolor']+2)
        b_profX_tag_shift1.SetLineWidth(2)
        
        b_profX_tag_shift2.SetMarkerStyle(26)
        b_profX_tag_shift2.SetMarkerColor(samples[back[0]]['linecolor']+2)
        b_profX_tag_shift2.SetLineColor(samples[back[0]]['linecolor']+2)
        b_profX_tag_shift2.SetLineWidth(2)
        
        b_profX_tag.SetMarkerStyle(21)
        b_profX_tag.SetMarkerColor(samples['WJetsToLNu']['linecolor'])
        b_profX_tag.SetLineColor(samples['WJetsToLNu']['linecolor'])
        b_profX_tag.SetLineWidth(2)

        #d_hist.Draw("COLZ")
        b_profX.Draw("PE,sames")
        b_profX_tag.Draw("PE,sames")
        ##b_hist_tag.GetYaxis().SetTitle("Jet response (p_{T}^{jet_{0}}/p_{T}^{Z})")
        ##b_hist_tag.GetXaxis().SetTitle(variable[var]["title"])
        ##b_hist.GetYaxis().SetTitle("Jet response (p_{T}^{jet_{0}}/p_{T}^{Z})")
        ##b_hist.GetXaxis().SetTitle(variable[var]["title"])#("p_{T}^{Z}")
        b_profX.GetYaxis().SetTitle("Jet response (p_{T}^{jet_{0}}/p_{T}^{Z})")
        b_profX.GetXaxis().SetTitle(variable[var]["title"])#("p_{T}^{Z}")

        d_profX               = TProfile(d_hist[var].ProfileX("d_prof"))
        d_profX_tag           = TProfile(d_hist_tag[var].ProfileX("d_prof_tag"))
        d_profX_untag_shift1 = TProfile(d_hist_untag_shift1[var].ProfileX("d_profX_untag_shift1"))
        d_profX_tag_shift1   = TProfile(d_hist_tag_shift1[var].ProfileX("d_profX_tag_shift1"))
        d_profX_tag_shift2   = TProfile(d_hist_tag_shift2[var].ProfileX("d_profX_tag_shift2"))
        d_profX.SetMarkerStyle(21)
        d_profX.SetMarkerColor(1)
        d_profX.SetLineColor(1)
        d_profX.SetLineWidth(2)
        
        d_profX_tag.SetMarkerStyle(21)
        d_profX_tag.SetMarkerColor(2)
        d_profX_tag.SetLineColor(2)
        d_profX_tag.SetLineWidth(2)

        d_profX_tag_shift1.SetMarkerStyle(25)
        d_profX_tag_shift1.SetMarkerColor(4)
        d_profX_tag_shift1.SetLineColor(4)
        d_profX_tag_shift1.SetLineWidth(2)
        
        d_profX_tag_shift2.SetMarkerStyle(32)
        d_profX_tag_shift2.SetMarkerColor(4)
        d_profX_tag_shift2.SetLineColor(4)
        d_profX_tag_shift2.SetLineWidth(2)

        d_profX.Draw("PE,sames")
        d_profX_tag.Draw("PE,sames")
        #b_profX_tag_shift1.Draw("PE,sames")
        b_profX_tag_shift2.Draw("PE,sames")
        #d_profX_tag_shift1.Draw("PE,sames")
        d_profX_tag_shift2.Draw("PE,sames")

        
        b_profX.SetMinimum(binmin_y)
        b_profX.SetMaximum(binmax_y)

        OUTSTRING = OUT
        OUTSTRING += "jet_response"
        OUTSTRING += "_vs_"+var.replace(".","_")
        drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
        #drawAnalysis("LL"+CHAN)
        drawRegion(SEL)
        leg.AddEntry(b_profX,"MC not tagged","PL")
        leg.AddEntry(d_profX,"data not tagged","PL")
        leg.AddEntry(b_profX_tag,"MC tagged","PL")
        leg.AddEntry(d_profX_tag,"data tagged","PL")
        #leg.AddEntry(b_profX_tag_shift1,"+1 ns shift tagged","PL")
        leg.AddEntry(b_profX_tag_shift2,"+2 ns shift tag MC","PL")
        leg.AddEntry(d_profX_tag_shift2,"+2 ns shift tag data","PL")
        leg.Draw()
        can.Update()
        can.Print(OUTSTRING+label+'.png')
        can.Print(OUTSTRING+label+'.pdf')
        b_profX.Delete()
        d_profX.Delete()
        b_profX_tag.Delete()
        d_profX_tag.Delete()
        b_profX_tag_shift1.Delete()
        d_profX_tag_shift1.Delete()
        b_profX_tag_shift2.Delete()
        d_profX_tag_shift2.Delete()
        can.Close()

    #Plot resolution 1d
    can = TCanvas("can","can",900,800)
    can.SetRightMargin(0.04)
    can.SetLeftMargin(0.12)
    can.cd()
    can.SetGrid()
    leg = TLegend(0.7, 0.7, 0.95, 0.9)
    b_hist_1d.SetMarkerStyle(21)
    b_hist_1d.SetMarkerColor(samples[back[0]]['linecolor'])
    b_hist_1d.SetLineColor(samples[back[0]]['linecolor'])
    b_hist_1d.SetLineWidth(2)

    b_hist_tag_1d_shift1.SetMarkerStyle(25)
    b_hist_tag_1d_shift1.SetMarkerColor(samples[back[0]]['linecolor']+2)
    b_hist_tag_1d_shift1.SetLineColor(samples[back[0]]['linecolor']+2)
    b_hist_tag_1d_shift1.SetLineWidth(2)

    b_hist_tag_1d_shift2.SetMarkerStyle(26)
    b_hist_tag_1d_shift2.SetMarkerColor(samples[back[0]]['linecolor']+2)
    b_hist_tag_1d_shift2.SetLineColor(samples[back[0]]['linecolor']+2)
    b_hist_tag_1d_shift2.SetLineWidth(2)

    b_hist_tag_1d.SetMarkerStyle(21)
    b_hist_tag_1d.SetMarkerColor(samples['WJetsToLNu']['linecolor'])
    b_hist_tag_1d.SetLineColor(samples['WJetsToLNu']['linecolor'])
    b_hist_tag_1d.SetLineWidth(2)

    b_hist_1d.GetXaxis().SetTitle("Jet response (p_{T}^{jet_{0}}/p_{T}^{Z})")

    d_hist_1d.SetMarkerStyle(21)
    d_hist_1d.SetMarkerColor(1)
    d_hist_1d.SetLineColor(1)
    d_hist_1d.SetLineWidth(2)

    d_hist_tag_1d.SetMarkerStyle(21)
    d_hist_tag_1d.SetMarkerColor(2)
    d_hist_tag_1d.SetLineColor(2)
    d_hist_tag_1d.SetLineWidth(2)


    d_hist_tag_1d_shift1.SetMarkerStyle(25)
    d_hist_tag_1d_shift1.SetMarkerColor(4)
    d_hist_tag_1d_shift1.SetLineColor(4)
    d_hist_tag_1d_shift1.SetLineWidth(2)

    d_hist_tag_1d_shift2.SetMarkerStyle(32)
    d_hist_tag_1d_shift2.SetMarkerColor(4)
    d_hist_tag_1d_shift2.SetLineColor(4)
    d_hist_tag_1d_shift2.SetLineWidth(2)


    b_hist_1d.Scale(1./b_hist_1d.Integral())
    b_hist_tag_1d.Scale(1./b_hist_tag_1d.Integral())
    d_hist_1d.Scale(1./d_hist_1d.Integral())
    d_hist_tag_1d.Scale(1./d_hist_tag_1d.Integral())

    b_hist_tag_1d_shift1.Scale(1./b_hist_tag_1d_shift1.Integral())
    b_hist_tag_1d_shift2.Scale(1./b_hist_tag_1d_shift2.Integral())
    d_hist_tag_1d_shift1.Scale(1./d_hist_tag_1d_shift1.Integral())
    d_hist_tag_1d_shift2.Scale(1./d_hist_tag_1d_shift2.Integral())

    b_hist_1d.Draw("PE,sames")
    b_hist_tag_1d.Draw("PE,sames")
    d_hist_tag_1d.Draw("PE,sames")
    d_hist_1d.Draw("PE,sames")

    #b_hist_tag_1d_shift1.Draw("PE,sames")
    b_hist_tag_1d_shift2.Draw("PE,sames")
    #d_hist_tag_1d_shift1.Draw("PE,sames")
    d_hist_tag_1d_shift2.Draw("PE,sames")

    #b_hist_1d.SetMinimum(binmin_y)
    #b_hist_1d.SetMaximum(binmax_y)

    OUTSTRING = OUT
    OUTSTRING += "jet_response"
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    #drawAnalysis("LL"+CHAN)
    drawRegion(SEL)
    leg.AddEntry(b_hist_1d,"MC not tagged","PL")
    leg.AddEntry(d_hist_1d,"data not tagged","PL")
    leg.AddEntry(b_hist_tag_1d,"MC tagged","PL")
    leg.AddEntry(d_hist_tag_1d,"data tagged","PL")
    leg.AddEntry(b_hist_tag_1d_shift1,"+2 ns shift tag MC","PL")
    leg.AddEntry(d_hist_tag_1d_shift2,"+2 ns shift tag data","PL")
    leg.Draw()
    can.Update()
    can.Print(OUTSTRING+'.png')
    can.Print(OUTSTRING+'.pdf')
    b_hist_1d.SetMaximum(0.07)
    b_hist_1d.SetMinimum(0.0001)
    can.SetLogy()
    can.Update()
    can.Print(OUTSTRING+'_log.png')
    can.Print(OUTSTRING+'_log.pdf')
    can.Close()
 
    '''
    #Plot resolution 1d
    can = TCanvas("can","can",900,800)
    can.SetRightMargin(0.04)
    can.SetLeftMargin(0.12)
    can.cd()
    can.SetGrid()
    leg = TLegend(0.7, 0.7, 0.95, 0.9)
    b_hist_1d.SetMarkerStyle(21)
    b_hist_1d.SetMarkerColor(samples[back[0]]['linecolor'])
    b_hist_1d.SetLineColor(samples[back[0]]['linecolor'])
    b_hist_1d.SetLineWidth(2)

    b_hist_tag_1d.SetMarkerStyle(21)
    b_hist_tag_1d.SetMarkerColor(samples['WJetsToLNu']['linecolor'])
    b_hist_tag_1d.SetLineColor(samples['WJetsToLNu']['linecolor'])
    b_hist_tag_1d.SetLineWidth(2)

    b_hist_1d.GetXaxis().SetTitle("Jet response (p_{T}^{jet_{0}}/p_{T}^{Z})")

    d_hist_1d.SetMarkerStyle(21)
    d_hist_1d.SetMarkerColor(1)
    d_hist_1d.SetLineColor(1)
    d_hist_1d.SetLineWidth(2)

    d_hist_tag_1d.SetMarkerStyle(21)
    d_hist_tag_1d.SetMarkerColor(2)
    d_hist_tag_1d.SetLineColor(2)
    d_hist_tag_1d.SetLineWidth(2)

    b_hist_1d.Scale(1./b_hist_1d.Integral())
    b_hist_tag_1d.Scale(1./b_hist_tag_1d.Integral())
    d_hist_1d.Scale(1./d_hist_1d.Integral())
    d_hist_tag_1d.Scale(1./d_hist_tag_1d.Integral())


    b_hist_1d.Draw("PE,sames")
    b_hist_tag_1d.Draw("PE,sames")
    d_hist_1d.Draw("PE,sames")
    d_hist_tag_1d.Draw("PE,sames")

    b_hist_1d_neg.SetMarkerColor(samples[back[0]]['linecolor'])
    b_hist_1d_neg_cut.SetMarkerColor(samples[back[0]]['linecolor']+2)
    b_hist_1d_neg.SetLineColor(samples[back[0]]['linecolor'])
    b_hist_1d_neg_cut.SetLineColor(samples[back[0]]['linecolor']+2)
    b_hist_1d_neg.SetMarkerStyle(25)
    b_hist_1d_neg_cut.SetMarkerStyle(26)

    d_hist_1d_neg.SetMarkerColor(1)
    d_hist_1d_neg_cut.SetMarkerColor(4)
    d_hist_1d_neg.SetLineColor(1)
    d_hist_1d_neg_cut.SetLineColor(4)
    d_hist_1d_neg.SetMarkerStyle(25)
    d_hist_1d_neg_cut.SetMarkerStyle(26)

    b_hist_1d_neg.Scale(1./b_hist_1d_neg.Integral())
    b_hist_1d_neg_cut.Scale(1./b_hist_1d_neg_cut.Integral())
    d_hist_1d_neg.Scale(1./d_hist_1d_neg.Integral())
    d_hist_1d_neg_cut.Scale(1./d_hist_1d_neg_cut.Integral())

    #b_hist_1d_neg.Draw("PE,sames")
    #b_hist_1d_neg_cut.Draw("PE,sames")
    #d_hist_1d_neg.Draw("PE,sames")
    #d_hist_1d_neg_cut.Draw("PE,sames")

    #b_hist_1d.SetMinimum(binmin_y)
    #b_hist_1d.SetMaximum(binmax_y)

    OUTSTRING = OUT
    OUTSTRING += "jet_response"
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    #drawAnalysis("LL"+CHAN)
    drawRegion(SEL)
    leg.AddEntry(b_hist_1d,"MC not tagged","PL")
    leg.AddEntry(d_hist_1d,"data not tagged","PL")
    leg.AddEntry(b_hist_tag_1d,"MC tagged","PL")
    leg.AddEntry(d_hist_tag_1d,"data tagged","PL")
    leg.Draw()
    can.Update()
    can.Print(OUTSTRING+'.png')
    can.Print(OUTSTRING+'.pdf')
    '''

    #We must build a variable to calculate the response
    #Compare data and MC
    #y = pT jet / pT Z
    #x = pT 2nd jet / pT Z
    #pT Z>30 GeV

    '''

    #Data/MC Ratio
    ratio0 = TH1F()
    h_den_r0 = TH1F("h_den_r0", "", len(bins_eta)-1, bins_eta)
    h_num_r0 = TH1F("h_num_r0", "", len(bins_eta)-1, bins_eta)
    ratio0.Sumw2()
    h_den_r0.Sumw2()
    h_num_r0.Sumw2()

    ratio1 = TH1F()
    h_den_r1 = TH1F("h_den_r1", "", len(bins_eta)-1, bins_eta)
    h_num_r1 = TH1F("h_num_r1", "", len(bins_eta)-1, bins_eta)
    ratio1.Sumw2()
    h_den_r1.Sumw2()
    h_num_r1.Sumw2()

    ratio2 = TH1F()
    h_den_r2 = TH1F("h_den_r2", "", len(bins_eta)-1, bins_eta)
    h_num_r2 = TH1F("h_num_r2", "", len(bins_eta)-1, bins_eta)
    ratio2.Sumw2()
    h_den_r2.Sumw2()
    h_num_r2.Sumw2()

    ratio3 = TH1F()
    h_den_r3 = TH1F("h_den_r3", "", len(bins_eta)-1, bins_eta)
    h_num_r3 = TH1F("h_num_r3", "", len(bins_eta)-1, bins_eta)
    ratio3.Sumw2()
    h_den_r3.Sumw2()
    h_num_r3.Sumw2()

    ratio4 = TH1F()
    h_den_r4 = TH1F("h_den_r4", "", len(bins_eta)-1, bins_eta)
    h_num_r4 = TH1F("h_num_r4", "", len(bins_eta)-1, bins_eta)
    ratio4.Sumw2()
    h_den_r4.Sumw2()
    h_num_r4.Sumw2()

    ratio5 = TH1F()
    h_den_r5 = TH1F("h_den_r5", "", len(bins_eta)-1, bins_eta)
    h_num_r5 = TH1F("h_num_r5", "", len(bins_eta)-1, bins_eta)
    ratio5.Sumw2()
    h_den_r5.Sumw2()
    h_num_r5.Sumw2()

    ratio6 = TH1F()
    h_den_r6 = TH1F("h_den_r6", "", len(bins_eta)-1, bins_eta)
    h_num_r6 = TH1F("h_num_r6", "", len(bins_eta)-1, bins_eta)
    ratio6.Sumw2()
    h_den_r6.Sumw2()
    h_num_r6.Sumw2()


    npoints = b_graph0.GetN()
    diff0 = TGraph(npoints)#,"",len(bins_eta)-1, bins_eta)
    diff1 = TGraph(npoints)#,"",len(bins_eta)-1, bins_eta)
    diff2 = TGraph(npoints)#,"",len(bins_eta)-1, bins_eta)
    diff3 = TGraph(npoints)#,"",len(bins_eta)-1, bins_eta)
    diff4 = TGraph(npoints)#,"",len(bins_eta)-1, bins_eta)
    diff5 = TGraph(npoints)#,"",len(bins_eta)-1, bins_eta)
    diff6 = TGraph(npoints)#,"",len(bins_eta)-1, bins_eta)
    for i in range(0,npoints):
        h_den_r0.Fill(b_graph0.GetPointX(i),b_graph0.GetPointY(i))
        h_den_r0.SetBinError(i,b_graph0.GetErrorY(i))
        h_num_r0.Fill(d_graph0.GetPointX(i),d_graph0.GetPointY(i))
        h_num_r0.SetBinError(i,d_graph0.GetErrorY(i))
        den = d_graph0.GetPointY(i) if d_graph0.GetPointY(i)>0 else 1
        diff0.SetPoint(i,b_graph0.GetPointX(i),100*abs(d_graph0.GetPointY(i)-b_graph0.GetPointY(i))/den)#/(d_graph0.GetPointY(i) if d_graph0.GetPointY(i)>0 else 1))
        print "r0 : b ", b_graph0.GetPointX(i), " , d  ", d_graph0.GetPointY(i)

        h_den_r1.Fill(b_graph1.GetPointX(i),b_graph1.GetPointY(i))
        h_den_r1.SetBinError(i,b_graph1.GetErrorY(i))
        h_num_r1.Fill(d_graph1.GetPointX(i),d_graph1.GetPointY(i))
        h_num_r1.SetBinError(i,d_graph1.GetErrorY(i))
        den = d_graph1.GetPointY(i) if d_graph1.GetPointY(i)>0 else 1
        diff1.SetPoint(i,b_graph1.GetPointX(i),100*abs(d_graph1.GetPointY(i)-b_graph1.GetPointY(i))/den)#/(d_graph1.GetPointY(i) if d_graph1.GetPointY(i)>0 else 1))

        h_den_r2.Fill(b_graph2.GetPointX(i),b_graph2.GetPointY(i))
        h_den_r2.SetBinError(i,b_graph2.GetErrorY(i))
        h_num_r2.Fill(d_graph2.GetPointX(i),d_graph2.GetPointY(i))
        h_num_r2.SetBinError(i,d_graph2.GetErrorY(i))
        den = d_graph2.GetPointY(i) if d_graph2.GetPointY(i)>0 else 1
        diff2.SetPoint(i,b_graph2.GetPointX(i),100*abs(d_graph2.GetPointY(i)-b_graph2.GetPointY(i))/den)#/(d_graph2.GetPointY(i) if d_graph2.GetPointY(i)>0 else 1))

        h_den_r3.Fill(b_graph3.GetPointX(i),b_graph3.GetPointY(i))
        h_den_r3.SetBinError(i,b_graph3.GetErrorY(i))
        h_num_r3.Fill(d_graph3.GetPointX(i),d_graph3.GetPointY(i))
        h_num_r3.SetBinError(i,d_graph3.GetErrorY(i))
        den = d_graph3.GetPointY(i) if d_graph3.GetPointY(i)>0 else 1
        diff3.SetPoint(i,b_graph3.GetPointX(i),100*abs(d_graph3.GetPointY(i)-b_graph3.GetPointY(i))/den)#/(d_graph3.GetPointY(i) if d_graph3.GetPointY(i)>0 else 1))

        h_den_r4.Fill(b_graph4.GetPointX(i),b_graph4.GetPointY(i))
        h_den_r4.SetBinError(i,b_graph4.GetErrorY(i))
        h_num_r4.Fill(d_graph4.GetPointX(i),d_graph4.GetPointY(i))
        h_num_r4.SetBinError(i,d_graph4.GetErrorY(i))
        den = d_graph4.GetPointY(i) if d_graph4.GetPointY(i)>0 else 1
        diff4.SetPoint(i,b_graph4.GetPointX(i),100*abs(d_graph4.GetPointY(i)-b_graph4.GetPointY(i))/den)#/(d_graph4.GetPointY(i) if d_graph4.GetPointY(i)>0 else 1))

        h_den_r5.Fill(b_graph5.GetPointX(i),b_graph5.GetPointY(i))
        h_den_r5.SetBinError(i,b_graph5.GetErrorY(i))
        h_num_r5.Fill(d_graph5.GetPointX(i),d_graph5.GetPointY(i))
        h_num_r5.SetBinError(i,d_graph5.GetErrorY(i))
        den = d_graph5.GetPointY(i) if d_graph5.GetPointY(i)>0 else 1
        diff5.SetPoint(i,b_graph5.GetPointX(i),100*abs(d_graph5.GetPointY(i)-b_graph5.GetPointY(i))/den)#/(d_graph5.GetPointY(i) if d_graph5.GetPointY(i)>0 else 1))

        h_den_r6.Fill(b_graph6.GetPointX(i),b_graph6.GetPointY(i))
        h_den_r6.SetBinError(i,b_graph6.GetErrorY(i))
        h_num_r6.Fill(d_graph6.GetPointX(i),d_graph6.GetPointY(i))
        h_num_r6.SetBinError(i,d_graph6.GetErrorY(i))
        den = d_graph6.GetPointY(i) if d_graph6.GetPointY(i)>0 else 1
        diff6.SetPoint(i,b_graph6.GetPointX(i),100*abs(d_graph6.GetPointY(i)-b_graph6.GetPointY(i))/den)#/(d_graph6.GetPointY(i) if d_graph6.GetPointY(i)>0 else 1))
    
    print "diff graphs"
    print "diff0"
    print diff0.Print()
    print b_graph0.Print()
    print d_graph0.Print()
    print "diff1"
    print diff1.Print()
    print b_graph1.Print()
    print d_graph1.Print()
    print "diff2"
    print diff2.Print()
    print b_graph2.Print()
    print d_graph2.Print()
    print "diff3"
    print diff3.Print()
    print b_graph3.Print()
    print d_graph3.Print()
    print "diff4"
    print diff4.Print()
    print b_graph4.Print()
    print d_graph4.Print()
    print "diff5"
    print diff5.Print()
    print b_graph5.Print()
    print d_graph5.Print()
    print "diff6"
    print diff6.Print()
    print b_graph6.Print()
    print d_graph6.Print()

    ratio0 = h_num_r0
    ratio0.Divide(h_num_r0,h_den_r0,1,1,"b")

    ratio1 = h_num_r1
    ratio1.Divide(h_num_r1,h_den_r1,1,1,"b")

    ratio2 = h_num_r2
    ratio2.Divide(h_num_r2,h_den_r2,1,1,"b")

    ratio3 = h_num_r3
    ratio3.Divide(h_num_r3,h_den_r3,1,1,"b")

    ratio4 = h_num_r4
    ratio4.Divide(h_num_r4,h_den_r4,1,1,"b")

    ratio5 = h_num_r5
    ratio5.Divide(h_num_r5,h_den_r5,1,1,"b")

    ratio6 = h_num_r6
    ratio6.Divide(h_num_r6,h_den_r6,1,1,"b")

    print "ratio graphs"
    print "ratio0"
    print ratio0.Print()
    print "ratio1"
    print ratio1.Print()
    print "ratio2"
    print ratio2.Print()
    print "ratio3"
    print ratio3.Print()
    print "ratio4"
    print ratio4.Print()
    print "ratio5"
    print ratio5.Print()
    print "ratio6"
    print ratio6.Print()

    can5_bkg = TCanvas("can5_bkg","can5_bkg",900,800)
    can5_bkg.SetRightMargin(0.04)
    can5_bkg.SetLeftMargin(0.14)
    can5_bkg.cd()
    leg2 = TLegend(0.1+0.45, 0.6, 0.4+0.45, 0.9)
    leg2.AddEntry(b_graph0,"time smeared MC","PL")
    leg2.AddEntry(b_graph1,"+1 ns shift","PL")
    leg2.AddEntry(b_graph2,"+2 ns shift","PL")
    leg2.AddEntry(b_graph3,"+3 ns shift","PL")
    leg2.AddEntry(b_graph4,"+4 ns shift","PL")
    leg2.AddEntry(b_graph5,"+5 ns shift","PL")
    leg2.AddEntry(b_graph6,"+6 ns shift","PL")
    leg2.AddEntry(d_graph0,"time data","PL")
    #leg2.AddEntry(d_graph0,"time data, mean"+str(mean_d_h_shift0),"PL")
    #leg2.AddEntry(d_graph1,"+1 ns shift, mean "+str(mean_d_h_shift1),"PL")
    #leg2.AddEntry(d_graph2,"+2 ns shift, mean "+str(mean_d_h_shift2),"PL")
    #leg2.AddEntry(d_graph3,"+3 ns shift, mean "+str(mean_d_h_shift3),"PL")
    #leg2.AddEntry(d_graph4,"+4 ns shift, mean "+str(mean_d_h_shift4),"PL")
    #leg2.AddEntry(d_graph5,"+5 ns shift, mean "+str(mean_d_h_shift5),"PL")
    #leg2.AddEntry(d_graph6,"+6 ns shift, mean "+str(mean_d_h_shift6),"PL")
    b_graph0.Draw("AP")
    b_graph1.Draw("P,sames")
    b_graph2.Draw("P,sames")
    b_graph3.Draw("P,sames")
    b_graph4.Draw("P,sames")
    b_graph5.Draw("P,sames")
    b_graph6.Draw("P,sames")
    d_graph0.Draw("P,sames")
    d_graph1.Draw("P,sames")
    d_graph2.Draw("P,sames")
    d_graph3.Draw("P,sames")
    d_graph4.Draw("P,sames")
    d_graph5.Draw("P,sames")
    d_graph6.Draw("P,sames")
    #for s in sign:
    #    leg2.AddEntry(s_graph[s],s+" mean: "+str(mean_s[s]),"PL")
    #    s_graph[s].Draw("P,sames")
    b_graph0.SetMaximum(1.01)
    b_graph0.GetXaxis().SetTitle("jet #eta")
    b_graph0.GetYaxis().SetTitle("tag efficiency")
    leg2.Draw()
    OUTSTRING = OUT
    OUTSTRING += "data_vs_back_tag_eff_shifted_jet_time"
    #drawAnalysis("LL"+CHAN)
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawRegion(SEL)
    can5_bkg.Update()
    can5_bkg.Print(OUTSTRING+label+'.png')
    can5_bkg.Print(OUTSTRING+label+'.pdf')
    can5_bkg.Close()


    can5_sgn = TCanvas("can5_sgn","can5_sgn",900,800)
    can5_sgn.SetRightMargin(0.05)
    can5_sgn.cd()
    leg2 = TLegend(0.2+0.15, 0.6, 0.4+0.35+0.15, 0.9)
    #leg2.AddEntry(b_graph0,"time smeared MC","PL")
    #leg2.AddEntry(b_graph1,"+1 ns shift","PL")
    #leg2.AddEntry(b_graph2,"+2 ns shift","PL")
    #leg2.AddEntry(b_graph3,"+3 ns shift","PL")
    #leg2.AddEntry(b_graph4,"+4 ns shift","PL")
    #leg2.AddEntry(b_graph5,"+5 ns shift","PL")
    #leg2.AddEntry(b_graph6,"+6 ns shift","PL")
    #leg2.AddEntry(d_graph0,"time data","PL")
    leg2.AddEntry(d_graph0,"time data, mean: "+str("%.2f"%mean_d_h_shift0),"PL")
    leg2.AddEntry(d_graph1,"+1 ns, mean: "+str("%.2f"%mean_d_h_shift1),"PL")
    leg2.AddEntry(d_graph2,"+2 ns, mean: "+str("%.2f"%mean_d_h_shift2),"PL")
    leg2.AddEntry(d_graph3,"+3 ns, mean: "+str("%.2f"%mean_d_h_shift3),"PL")
    leg2.AddEntry(d_graph4,"+4 ns, mean: "+str("%.2f"%mean_d_h_shift4),"PL")
    leg2.AddEntry(d_graph5,"+5 ns, mean: "+str("%.2f"%mean_d_h_shift5),"PL")
    leg2.AddEntry(d_graph6,"+6 ns, mean: "+str("%.2f"%mean_d_h_shift6),"PL")
    #b_graph0.Draw("AP")
    #b_graph1.Draw("P,sames")
    #b_graph2.Draw("P,sames")
    #b_graph3.Draw("P,sames")
    #b_graph4.Draw("P,sames")
    #b_graph5.Draw("P,sames")
    #b_graph6.Draw("P,sames")

    d_graph0.SetLineStyle(1)
    d_graph1.SetLineStyle(1)
    d_graph2.SetLineStyle(1)
    d_graph3.SetLineStyle(1)
    d_graph4.SetLineStyle(1)
    d_graph5.SetLineStyle(1)
    d_graph6.SetLineStyle(1)


    d_graph0.Draw("AP")
    d_graph1.Draw("P,sames")
    d_graph2.Draw("P,sames")
    d_graph3.Draw("P,sames")
    d_graph4.Draw("P,sames")
    d_graph5.Draw("P,sames")
    d_graph6.Draw("P,sames")
    for s in sign:
        leg2.AddEntry(s_graph[s],samples[s]['label']+", mean: "+str("%.2f"%mean_s[s]),"PL")
        s_graph[s].Draw("P,sames")
    d_graph0.SetMaximum(1.01)
    d_graph0.GetXaxis().SetTitle("jet #eta")
    d_graph0.GetYaxis().SetTitle("tag efficiency")
    leg2.Draw()
    OUTSTRING = OUT
    OUTSTRING += "data_vs_sign_tag_eff_shifted_jet_time"
    #drawAnalysis("LL"+CHAN)
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawRegion(SEL)
    can5_sgn.Update()
    can5_sgn.Print(OUTSTRING+label+'.png')
    can5_sgn.Print(OUTSTRING+label+'.pdf')
    can5_sgn.Close()



    ratio0.SetMarkerStyle(21)
    ratio1.SetMarkerStyle(21)
    ratio2.SetMarkerStyle(21)
    ratio3.SetMarkerStyle(21)
    ratio4.SetMarkerStyle(21)
    ratio5.SetMarkerStyle(21)
    ratio6.SetMarkerStyle(21)
    ratio0.SetMarkerColor(colors[0])
    ratio1.SetMarkerColor(colors[1])
    ratio2.SetMarkerColor(colors[2])
    ratio3.SetMarkerColor(colors[3])
    ratio4.SetMarkerColor(colors[4])
    ratio5.SetMarkerColor(colors[5])
    ratio6.SetMarkerColor(colors[6])
    ratio0.SetLineColor(colors[0])
    ratio1.SetLineColor(colors[1])
    ratio2.SetLineColor(colors[2])
    ratio3.SetLineColor(colors[3])
    ratio4.SetLineColor(colors[4])
    ratio5.SetLineColor(colors[5])
    ratio6.SetLineColor(colors[6])
    ratio0.SetLineWidth(2)
    ratio1.SetLineWidth(2)
    ratio2.SetLineWidth(2)
    ratio3.SetLineWidth(2)
    ratio4.SetLineWidth(2)
    ratio5.SetLineWidth(2)
    ratio6.SetLineWidth(2)

    can6 = TCanvas("can6","can6",900,800)
    can6.SetRightMargin(0.05)
    can6.cd()
    leg2 = TLegend(0.1+0.45, 0.6, 0.4+0.45, 0.9)
    leg2.AddEntry(ratio0,"time smeared","PL")
    leg2.AddEntry(ratio1,"+1 ns shift","PL")
    leg2.AddEntry(ratio2,"+2 ns shift","PL")
    leg2.AddEntry(ratio3,"+3 ns shift","PL")
    leg2.AddEntry(ratio4,"+4 ns shift","PL")
    leg2.AddEntry(ratio5,"+5 ns shift","PL")
    leg2.AddEntry(ratio6,"+6 ns shift","PL")
    ratio0.Draw("P")
    ratio1.Draw("P,sames")
    ratio2.Draw("P,sames")
    ratio3.Draw("P,sames")
    ratio4.Draw("P,sames")
    ratio5.Draw("P,sames")
    ratio6.Draw("P,sames")
    ratio0.GetXaxis().SetTitle("jet #eta")
    ratio0.GetYaxis().SetTitle("data/MC scale factor")
    ratio0.SetMinimum(0)
    ratio0.SetMaximum(4)
    leg2.Draw()
    OUTSTRING = OUT
    OUTSTRING += "data_back_SF_shifted_jet_time"
    #drawAnalysis("LL"+CHAN)
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawRegion(SEL)
    can6.Update()
    can6.Print(OUTSTRING+label+'.png')
    can6.Print(OUTSTRING+label+'.pdf')
    can6.Close()

    #diff
    diff0.SetMarkerStyle(21)
    diff1.SetMarkerStyle(21)
    diff2.SetMarkerStyle(21)
    diff3.SetMarkerStyle(21)
    diff4.SetMarkerStyle(21)
    diff5.SetMarkerStyle(21)
    diff6.SetMarkerStyle(21)
    diff0.SetMarkerColor(colors[0])
    diff1.SetMarkerColor(colors[1])
    diff2.SetMarkerColor(colors[2])
    diff3.SetMarkerColor(colors[3])
    diff4.SetMarkerColor(colors[4])
    diff5.SetMarkerColor(colors[5])
    diff6.SetMarkerColor(colors[6])
    diff0.SetLineColor(colors[0])
    diff1.SetLineColor(colors[1])
    diff2.SetLineColor(colors[2])
    diff3.SetLineColor(colors[3])
    diff4.SetLineColor(colors[4])
    diff5.SetLineColor(colors[5])
    diff6.SetLineColor(colors[6])
    diff0.SetLineWidth(2)
    diff1.SetLineWidth(2)
    diff2.SetLineWidth(2)
    diff3.SetLineWidth(2)
    diff4.SetLineWidth(2)
    diff5.SetLineWidth(2)
    diff6.SetLineWidth(2)

    can7 = TCanvas("can7","can7",900,800)
    can7.SetRightMargin(0.05)
    can7.cd()
    leg2 = TLegend(0.1+0.35, 0.6, 0.4+0.35, 0.9)
    leg2.AddEntry(diff0,"time smeared","PL")
    leg2.AddEntry(diff1,"+1 ns shift","PL")
    leg2.AddEntry(diff2,"+2 ns shift","PL")
    leg2.AddEntry(diff3,"+3 ns shift","PL")
    leg2.AddEntry(diff4,"+4 ns shift","PL")
    leg2.AddEntry(diff5,"+5 ns shift","PL")
    leg2.AddEntry(diff6,"+6 ns shift","PL")
    diff0.SetTitle("")
    diff0.Draw("APL")
    diff1.Draw("PL,sames")
    diff2.Draw("PL,sames")
    diff3.Draw("PL,sames")
    diff4.Draw("PL,sames")
    diff5.Draw("PL,sames")
    diff6.Draw("PL,sames")
    diff0.GetXaxis().SetTitle("jet #eta")
    diff0.GetYaxis().SetTitle("data/MC tag efficiency difference (%)")
    diff0.SetMinimum(0)
    diff0.SetMaximum(200)
    leg2.Draw()
    OUTSTRING = OUT
    OUTSTRING += "data_back_diff_perc_shifted_jet_time"
    #drawAnalysis("LL"+CHAN)
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawRegion(SEL)
    can7.Update()
    can7.Print(OUTSTRING+label+'.png')
    can7.Print(OUTSTRING+label+'.pdf')
    can7.Close()

    '''


def signal_jet_response(cut,dnn,label=""):

    if SEL=="SR":
        prj_weight = "EventWeight*PUReWeight*TriggerWeight"
    else:
        prj_weight = "EventWeight*PUReWeight"

    if ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples
        from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    if ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
        from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
        from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi

    for s in sign:
        chain = TChain("tree")
        for j, ss in enumerate(samples[s]['files']):
            chain.Add(MAIN_MET + ss + ".root")

        var_x = "Jets.pt/Jets.ptGenJ"
        nbins_x = 50
        binmin_x = 0
        binmax_x = 5#10
        nbins_y = 50
        binmin_y = 0
        binmax_y = 1


        h = TH1F("h"+s,"", nbins_x, binmin_x, binmax_x)
        h.Sumw2()
        ht = TH1F("ht"+s,"", nbins_x, binmin_x, binmax_x)
        ht.Sumw2()
        h2 = TH2F("h2"+s,"", nbins_y, binmin_y, binmax_y, nbins_x, binmin_x, binmax_x)
        h2.Sumw2()

        chain.Project("h"+s, var_x, prj_weight+"*("+cut+")")
        chain.Project("ht"+s, var_x, prj_weight+"*("+cut+" && Jets.sigprob>0.996)")
        chain.Project("h2"+s, var_x+":Jets.sigprob", prj_weight+"*("+cut+")")

        #Plot resolution
        can = TCanvas("can","can",900,800)
        can.SetRightMargin(0.04)
        can.SetLeftMargin(0.12)
        can.cd()
        can.SetGrid()
        can.SetLogy()
        leg = TLegend(0.6, 0.65, 0.97, 0.9)
        h.SetMarkerStyle(21)
        h.SetMarkerColor(samples[s]['linecolor'])
        h.SetLineColor(samples[s]['linecolor'])
        h.SetLineWidth(2)

        ht.SetMarkerStyle(20)
        ht.SetMarkerColor(samples[s]['linecolor']+5)
        ht.SetLineColor(samples[s]['linecolor']+5)
        ht.SetLineWidth(2)
        
        h.Scale(1./h.Integral())
        ht.Scale(1./ht.Integral())

        h.Draw("PE,sames")
        ht.Draw("PE,sames")
        h.GetXaxis().SetTitle("Jet response (p_{T}^{reco}/p_{T}^{gen})")

        #h.SetMinimum(binmin_y)
        #h.SetMaximum(binmax_y)
        
        OUTSTRING = OUT
        OUTSTRING += s+"_jet_response"
        #OUTSTRING += "_vs_DNN_score_zoom"
        drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
        drawRegion("isSR")
        leg.AddEntry(h,samples[s]['label'],"PL")
        leg.AddEntry(ht,"tagged jets","PL")
        leg.Draw()
        can.Update()
        can.Print(OUTSTRING+label+'.png')
        can.Print(OUTSTRING+label+'.pdf')
        can.Close()


        #Plot response vs prob
        can = TCanvas("can","can",900,800)
        can.SetRightMargin(0.04)
        can.SetLeftMargin(0.12)
        can.cd()
        can.SetGrid()
        leg = TLegend(0.7, 0.7, 0.95, 0.9)
        profX = TProfile(h2.ProfileX("b_prof"))
        profX.SetMarkerStyle(21)
        profX.SetMarkerColor(samples[s]['linecolor'])
        profX.SetLineColor(samples[s]['linecolor'])
        profX.SetLineWidth(2)
        
        profX.Draw("PE,sames")
        profX.GetYaxis().SetTitle("Jet response (p_{T}^{reco}/p_{T}^{gen})")
        profX.GetXaxis().SetTitle("AK4 jet[0] DNN score")#("p_{T}^{Z}")

        profX.SetMinimum(0)
        profX.SetMaximum(2)
        
        OUTSTRING = OUT
        OUTSTRING += s+"_jet_response"
        OUTSTRING += "_vs_DNN_score_zoom"
        drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
        drawRegion("isSR")
        leg.AddEntry(profX,samples[s]['label'],"PL")
        leg.Draw()
        can.Update()
        can.Print(OUTSTRING+label+'.png')
        can.Print(OUTSTRING+label+'.pdf')
        profX.Delete()
        can.Close()

        h.Delete()
        ht.Delete()
        h2.Delete()


def draw_syst_unc():
    print "Drawing uncertainty vs mass"
    with open(OUT+"signal_time_smearing_integrals.yaml","r") as f:
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
    for c in ctaus:
        diff[c] = TGraph()
        n = 0
        for m in masses:
            for s in samp:
                if "mh"+str(m) in s and "ctau"+str(c) in s:
                    print str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % results[s]['diff'])+"\\"+"\\"
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
    #c1.SetLogx()
    c1.Print(OUT+"signal_uncertainty_time_smearing.pdf")
    c1.Print(OUT+"signal_uncertainty_time_smearing.png")
    c1.Close()
 

lab = ""

signal_jet_response(cut = "abs(Jets.eta)<1 && isSR && MinJetMetDPhi>0.5",label="_DNN_0p996",dnn=0.996)
exit()

if SEL=="SR":
    #time_fit(var="JetsNegative[0].timeRecHitsEB",cut = "isSR",label=lab,scale=True,do_smear=False)
    #time_fit(var="JetsNegative.timeRecHitsEB",cut = "isSR && fabs(Jets.eta)<1 && MinJetMetDPhi>0.5",label=lab+"_all_jets",scale=True,do_smear=False)
    #signal_smearing(var="JetsNegative.timeRecHitsEB",cut = "isSR && fabs(Jets.eta)<1 && MinJetMetDPhi>0.5",label=lab+"_all_jets",scale=True,do_smear=True)
    draw_syst_unc()
if SEL=="ZtoLL":
    #jet_response(var="JetsNegative.timeRecHitsEB",cut = "fabs(Z_mass-91.2)<10. && Z_pt>30 && nCHSJetsAcceptanceCalo==2 && (Jets[1].pt/Jets[0].pt)<0.3 && abs(Jets[0].eta)<1 && abs(Jets[1].eta)<1 && Jets[1].sigprob<=0.996",label="_alpha_0p3",dnn=0.996)
    #jet_response(var="JetsNegative.timeRecHitsEB",cut = "fabs(Z_mass-91.2)<10. && Z_pt>30 && nCHSJetsAcceptanceCalo==2 && abs(Jets[0].eta)<1 && abs(Jets[1].eta)<1 && Jets[1].sigprob<=0.996",label="_no_alpha",dnn=0.996)
    #jet_response(var="JetsNegative.timeRecHitsEB",cut = "fabs(Z_mass-91.2)<10. && Z_pt>30 && nCHSJetsAcceptanceCalo==2 && abs(Jets[0].eta)<1 && abs(Jets[1].eta)<1 && Jets[1].sigprob<=0.9",label="_no_alpha_DNN_0p9",dnn=0.9)
    #HERE:
    #jet_response(var_list=["Z_pt","Jets[0].timeRecHitsEB","Jets[0].sigprob"],cut = "fabs(Z_mass-91.2)<10. && Z_pt>30 && abs(Jets[0].eta)<1",label="_DNN_0p996",dnn=0.996)
    plot_all_years(cut = "fabs(Z_mass-91.2)<10. && Z_pt>30 && abs(Jets[0].eta)<1",label="_DNN_0p996")

    #jet_response(var="Z_pt",cut = "fabs(Z_mass-91.2)<10. && Z_pt>30 && abs(Jets[0].eta)<1",label="_vs_Zpt_DNN_0p996",dnn=0.996)
    #jet_response(var="Jets[0].timeRecHitsEB",cut = "fabs(Z_mass-91.2)<10. && Z_pt>30 && abs(Jets[0].eta)<1",label="_vs_time_DNN_0p996",dnn=0.996)
    #jet_response(var="Jets[0].sigprob",cut = "fabs(Z_mass-91.2)<10. && Z_pt>30 && abs(Jets[0].eta)<1",label="_vs_sigprob_DNN_0p996",dnn=0.996)
