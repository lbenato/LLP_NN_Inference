#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import uproot
import root_numpy
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TMultiGraph, TH3F, TF1, TRatioPlot, TFitResult
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
SEL = "ZtoLLPho"

ERA = "2016"
LUMI = -1

if ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    if SEL=="ZtoLLPho":
        LUMI = lumi["SingleMuon"]["tot"]#+lumi["SingleElectron"]["tot"]
if ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    if SEL=="ZtoLLPho":
        LUMI = lumi["SingleMuon"]["tot"]#+lumi["SingleElectron"]["tot"]
if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    if SEL=="ZtoLLPho":
        LUMI = lumi["SingleMuon"]["tot"]#+lumi["EGamma"]["tot"]
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
if SEL=="ZtoLLPho":
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

sign = ['SUSY_mh127_ctau500','SUSY_mh300_ctau500','SUSY_mh600_ctau500','SUSY_mh800_ctau500','SUSY_mh1000_ctau500','SUSY_mh1800_ctau500']

sel_sign = ['SUSY_mh400_ctau500']

MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_"+SEL+"_v5_ntuples/"
MAIN_MM = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_ZtoMMPho_v5_ntuples/"
MAIN_EE = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_ZtoEEPho_v5_ntuples/"

MAIN_MET = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_no_photonEFrac_cut/"
#MAIN_GEN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_Gen/"
OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_"+SEL+"_v5_ntuples/"
OUT_TtoEM = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_TtoEM_v5_ntuples_validate_timeRecHits/"

def deltaPhi_np( a1, v2):
    a2 = np.multiply(np.ones(a1.size),v2)
    pi = np.multiply(np.ones(a1.size),math.pi)
    diff = np.abs(a1 - a2)
    mask = diff>pi
    M = np.ma.masked_array(data=diff, mask = mask)
    new = diff - 2*math.pi*np.ones(diff.size)
    M.filled(new)
    return np.array(M)

def calc_sf(var,cut,label="",scale=True, do_smear=False):
    list_of_variables = ["EventNumber","LumiNumber","RunNumber","Jets*.pt","Jets*.eta","Jets*.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight","TriggerWeight","PUWeight","PUReWeight","Jets*.isGenMatchedCaloCorrLLPAccept","Jets*.CSV","Jets*.timeRecHitsEB","is*","JetsNegative*.CSV","JetsNegative*.sigprob","JetsNegative*.timeRecHitsEB"]

    chain = {}
    hist = {}

    var_to_model = ["nHadEFrac","eFracRecHitsEB","photonEFrac","nRecHitsEB","ptDEB"]

    rb_cHadEFrac = np.array([])
    rb_nHadEFrac = np.array([])
    rb_eFracRecHitsEB = np.array([])
    rb_photonEFrac = np.array([])

    rd_cHadEFrac = np.array([])
    rd_nHadEFrac = np.array([])
    rd_eFracRecHitsEB = np.array([])
    rd_photonEFrac = np.array([])

    hist_var_b = {}
    hist_var_d = {}
    hist_var_s = {}
    #hist_var_rb = {}
    #hist_var_rd = {}
    hist_var_sb = {}
    hist_var_sd = {}

    my_sign_var = {}
    my_data_var = {}
    my_back_var = {}
    shift_b = {}
    shift_d = {}
    for v in var_to_model:
        jv = "Jets[0]."+v
        hist_var_b[v] = TH1F("b_"+v,";"+variable[jv]['title'],variable[jv]['nbins'], variable[jv]['min'],variable[jv]['max'])
        hist_var_d[v] = TH1F("d_"+v,";"+variable[jv]['title'],variable[jv]['nbins'], variable[jv]['min'],variable[jv]['max'])
        hist_var_s[v] = TH1F("s_"+v,";"+variable[jv]['title'],variable[jv]['nbins'], variable[jv]['min'],variable[jv]['max'])
        #hist_var_rb[v] = TH1F("rb_"+v,";"+variable[jv]['title'],variable[jv]['nbins'], variable[jv]['min'],variable[jv]['max'])
        #hist_var_rd[v] = TH1F("rd_"+v,";"+variable[jv]['title'],variable[jv]['nbins'], variable[jv]['min'],variable[jv]['max'])
        hist_var_sb[v] = TH1F("sb_"+v,";"+variable[jv]['title'],variable[jv]['nbins'], variable[jv]['min'],variable[jv]['max'])
        hist_var_sd[v] = TH1F("sd_"+v,";"+variable[jv]['title'],variable[jv]['nbins'], variable[jv]['min'],variable[jv]['max'])
        hist_var_b[v].Sumw2()
        hist_var_d[v].Sumw2()
        hist_var_s[v].Sumw2()
        #hist_var_rb[v].Sumw2()
        #hist_var_rd[v].Sumw2()
        hist_var_sb[v].Sumw2()
        hist_var_sd[v].Sumw2()
        my_back_var[v] = TF1()
        my_data_var[v] = TF1()
        my_sign_var[v] = TF1()

    if SEL=="SR":
        prj_weight = "EventWeight*PUReWeight*TriggerWeight"
    else:
        prj_weight = "EventWeight*PUReWeight"

    #bins_eta = array('d',[-1.5, -1.25, -1., -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5])
    bins_eta = array('d',[-1.5, -1., -0.5, 0., 0.5, 1., 1.5])
    bins_eta = array('d',[-1.,0.,1.,])
    bins_sigprob = array('d',[0.0001,0.001,0.01,0.1,0.5,0.8,0.9,0.95,0.99,0.995,1.])

    data_file = TFile(OUT_TtoEM+"data_smear_file_CSV_0p8_all_jets"+label+".root","READ")
    data_file.cd()
    my_data = data_file.Get("data_CB")
    data_file.Close()

    model = keras.models.load_model('nn_inference/tagger_AK4_v3/model.h5')

    print "Cut: ", cut

    #sel signal first
    for s in sel_sign:
        chain[s] = TChain("tree")
        for j, ss in enumerate(samples[s]['files']):
            print MAIN_MET+ss + ".root"
            chain[s].Add(MAIN_MET + ss + ".root")
        for v in var_to_model:
            jv = "Jets[0]."+v
            chain[s].Project("s_"+v, jv, prj_weight+"*(MinJetMetDPhi>0.5 && Jets[0].isGenMatchedCaloCorrLLPAccept && fabs(Jets[0].eta)<1)")
            hist_var_s[v].Scale(1./hist_var_s[v].Integral())


    #TODO merge EE and MM...
    for s in back:
        hist[s] = TH1F(s,";"+variable[var]['title'],variable[var]['nbins'], variable[var]['min'],variable[var]['max'])
        #hist[s] = TH1F(s,";"+variable[var]['title'],5,-5,5)
        hist[s].Sumw2()
        chain[s] = TChain("tree")
        for j, ss in enumerate(samples[s]['files']):
            print MAIN_EE+ss + ".root"
            chain[s].Add(MAIN_EE + ss + ".root")
            print MAIN_MM+ss + ".root"
            chain[s].Add(MAIN_MM + ss + ".root")
        if "JetsNegative[0]" in var:
            chain[s].Project(s, var, prj_weight+"*("+cut+" && fabs(JetsNegative[0].eta)<1 )")
        if "Jets[0]" in var:
            chain[s].Project(s, var, prj_weight+"*("+cut+" && fabs(Jets[0].eta)<1 )")
        for v in var_to_model:
            jv = "Jets[0]."+v
            chain[s].Project("b_"+v, jv, prj_weight+"*("+cut+" && fabs(Jets[0].eta)<1)")
            hist_var_b[v].Scale(1./hist_var_b[v].Integral())
            #hist_var_rb[v] = hist_var_s[v].Clone("rb_"+v)
            #hist_var_rb[v].Divide(hist_var_b[v])
        if scale:
            hist[s].Scale(1./hist[s].Integral())
        #fit_back = TF1("fit_back","crystalball",5,5)
        fit_back = TF1("fit_back","gaus",5,5)
        fit_back.SetParameter(0,0.01)
        fit_back.SetParameter(1,hist[s].GetMean())
        fit_back.SetParameter(2,hist[s].GetRMS())
        #fit_back.FixParameter(1,hist[s].GetMean())
        #fit_back.FixParameter(2,hist[s].GetRMS())
        hist[s].Fit(fit_back,"E")
        my_back = hist[s].GetFunction("fit_back")
        hist[s].GetListOfFunctions().Remove(hist[s].GetFunction("fit_back"))
        #print hist[s].GetMean()
        #print hist[s].GetRMS()
        hist[s].SetLineColor(samples[s]['linecolor'])
        hist[s].SetFillColorAlpha(samples[s]['linecolor'],0.5)
        my_back.SetLineStyle(2)
            
    for s in data:
        print s
        hist[s] = TH1F(s,";"+variable[var]['title'],variable[var]['nbins'], variable[var]['min'],variable[var]['max'])
        #hist[s] = TH1F(s,";"+variable[var]['title'],5,-5,5)
        hist[s].Sumw2()
        chain[s] = TChain("tree")
        for j, ss in enumerate(samples[s]['files']):
            if 'Muon' in ss:
                print MAIN_MM+ss + ".root"
                chain[s].Add(MAIN_MM + ss + ".root")
            if 'Electron' in ss or 'EGamma' in ss:
                print MAIN_EE+ss + ".root"
                chain[s].Add(MAIN_EE + ss + ".root")

        if "JetsNegative[0]" in var:
            chain[s].Project(s, var, prj_weight+"*("+cut+" && fabs(JetsNegative[0].eta)<1 )")
        if "Jets[0]" in var:
            chain[s].Project(s, var, prj_weight+"*("+cut+" && fabs(Jets[0].eta)<1 )")
        for v in var_to_model:
            jv = "Jets[0]."+v
            chain[s].Project("d_"+v, jv, prj_weight+"*("+cut+" && fabs(Jets[0].eta)<1)")
            hist_var_d[v].Scale(1./hist_var_d[v].Integral())
            #hist_var_rd[v] = hist_var_s[v].Clone("rd_"+v)
            #hist_var_rd[v].Divide(hist_var_d[v])
        if scale:
            hist[s].Scale(1./hist[s].Integral())
        hist[s].SetMarkerStyle(21)
        hist[s].SetMarkerColor(1)
        hist[s].SetLineColor(1)

    #Plot variables to scale first
    for v in var_to_model:
        jv = "Jets[0]."+v
        can = TCanvas("can","can",900,800)
        can.SetRightMargin(0.05)
        can.cd()
        #Try CB fits
        #fit_back = TF1("fit_back","crystalball",5,5)
        print "\n"
        print "fit back ", v
        #fit_back = TF1("fit_back","gaus",variable[jv]['min'],variable[jv]['max'])
        fit_back = TF1("fit_back","gaus",hist_var_b[v].GetMean()-2*hist_var_b[v].GetRMS(),hist_var_b[v].GetMean()+2*hist_var_b[v].GetRMS())
        fit_back.SetParameter(0,0.01)
        fit_back.SetParameter(1,hist_var_b[v].GetMean())
        fit_back.SetParameter(2,hist_var_b[v].GetRMS())
        #fit_back.SetParameter(3,4)
        #fit_back.SetParameter(4,5)
        res = hist_var_b[v].Fit(fit_back,"ERBS")
        my_back_var[v] = hist_var_b[v].GetFunction("fit_back")
        hist_var_b[v].GetListOfFunctions().Remove(hist_var_b[v].GetFunction("fit_back"))
        if res.Status()!=0:
            print "Fit failed, using mean and RMS instead"
            my_back_var[v].SetParameter(1,hist_var_b[v].GetMean())
            my_back_var[v].SetParameter(2,hist_var_b[v].GetRMS())

        print "\n"
        print "fit data ", v
        #fit_data = TF1("fit_data","gaus",variable[jv]['min'],variable[jv]['max'])
        fit_data = TF1("fit_data","gaus",hist_var_d[v].GetMean()-2*hist_var_d[v].GetRMS(),hist_var_d[v].GetMean()+2*hist_var_d[v].GetRMS())
        fit_data.SetParameter(0,0.01)
        fit_data.SetParameter(1,hist_var_d[v].GetMean())
        fit_data.SetParameter(2,hist_var_d[v].GetRMS())
        if v=="photonEFrac":
            fit_data.SetParLimits(1,0.5,1)
        #fit_data.SetParameter(3,4)
        #fit_data.SetParameter(4,5)
        hist_var_d[v].Fit(fit_data,"ERBS")
        my_data_var[v] = hist_var_d[v].GetFunction("fit_data")
        hist_var_d[v].GetListOfFunctions().Remove(hist_var_d[v].GetFunction("fit_data"))
        if res.Status()!=0:
            print "Fit failed, using mean and RMS instead"
            my_data_var[v].SetParameter(1,hist_var_d[v].GetMean())
            my_data_var[v].SetParameter(2,hist_var_d[v].GetRMS())

        print "\n"
        print "fit sign ", v
        #fit_sign = TF1("fit_sign","gaus",variable[jv]['min'],variable[jv]['max'])
        fit_sign = TF1("fit_sign","gaus",hist_var_s[v].GetMean()-2*hist_var_s[v].GetRMS(),hist_var_s[v].GetMean()+2*hist_var_s[v].GetRMS())
        fit_sign.SetParameter(0,0.01)
        fit_sign.SetParameter(1,hist_var_s[v].GetMean())
        fit_sign.SetParameter(2,hist_var_s[v].GetRMS())
        #fit_data.SetParameter(3,4)
        #fit_data.SetParameter(4,5)
        #hist_var_s[v].Fit(fit_sign,"ERB")
        res = hist_var_s[v].Fit(fit_sign,"ERBS")
        my_sign_var[v] = hist_var_s[v].GetFunction("fit_sign")
        hist_var_s[v].GetListOfFunctions().Remove(hist_var_s[v].GetFunction("fit_sign"))
        if res.Status()!=0:
            print "Fit failed, using mean and RMS instead"
            my_sign_var[v].SetParameter(1,hist_var_s[v].GetMean())
            my_sign_var[v].SetParameter(2,hist_var_s[v].GetRMS())

        leg = TLegend(0.75, 0.79, 1., 0.99)
        #leg.SetFillStyle(0)
        hist_var_b[v].SetLineColor(samples[back[0]]['linecolor'])
        hist_var_b[v].SetMarkerColor(samples[back[0]]['linecolor'])
        hist_var_b[v].SetFillColorAlpha(samples[back[0]]['linecolor'],0.5)
        hist_var_s[v].SetLineColor(samples[sel_sign[0]]['linecolor'])
        hist_var_s[v].SetLineWidth(2)
        hist_var_s[v].SetMarkerColor(samples[sel_sign[0]]['linecolor'])
        hist_var_d[v].SetLineColor(1)
        hist_var_d[v].SetMarkerStyle(21)
        hist_var_d[v].SetMarkerColor(1)

        my_sign_var[v].SetLineColor(samples[sel_sign[0]]['linecolor'])
        my_back_var[v].SetLineColor(samples[back[0]]['linecolor'])
        my_data_var[v].SetLineColor(samples[data[0]]['linecolor'])
        my_back_var[v].SetLineStyle(2)
        my_data_var[v].SetLineStyle(2)
        my_sign_var[v].SetLineStyle(2)

        hist_var_b[v].Draw("HISTO,sames")
        hist_var_d[v].Draw("PE,sames")
        hist_var_s[v].Draw("HISTO,sames")
        my_back_var[v].Draw("L,sames")
        my_data_var[v].Draw("L,sames")
        my_sign_var[v].Draw("L,sames")
        leg.AddEntry(hist_var_b[v],samples[back[0]]['label'],"F")
        leg.AddEntry(hist_var_s[v],samples[sel_sign[0]]['label'],"L")
        leg.AddEntry(hist_var_d[v],samples[data[0]]['label'],"P")
        OUTSTRING = OUT
        OUTSTRING += v.replace(".","_")+"_to_model"
        drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True,left_marg_CMS=0.1,left_marg_LUMI=0.74)
        drawRegion(SEL)
        leg.Draw()
        can.Update()
        can.Print(OUTSTRING+label+'.png')
        can.Print(OUTSTRING+label+'.pdf')
        can.SetLogy()
        can.Update()
        can.Print(OUTSTRING+label+'_log.png')
        can.Print(OUTSTRING+label+'_log.pdf')
        can.Close()

    '''
    for v in var_to_model:
        can = TCanvas("can","can",900,800)
        can.SetRightMargin(0.05)
        can.cd()
        leg = TLegend(0.65, 0.6, 1., 0.9)
        hist_var_b[v].SetLineColor(samples[back[0]]['linecolor'])
        hist_var_b[v].SetMarkerColor(samples[back[0]]['linecolor'])
        hist_var_b[v].SetFillColorAlpha(samples[back[0]]['linecolor'],0.5)
        hist_var_s[v].SetLineColor(samples[sel_sign[0]]['linecolor'])
        hist_var_s[v].SetMarkerColor(samples[sel_sign[0]]['linecolor'])
        hist_var_d[v].SetLineColor(1)
        hist_var_d[v].SetMarkerStyle(21)
        hist_var_d[v].SetMarkerColor(1)
        hist_var_sb[v] = hist_var_b[v].Clone("sb_"+v)
        hist_var_sd[v] = hist_var_d[v].Clone("sd_"+v)
        hist_var_sb[v].Multiply(hist_var_rb[v])#On purpose! Scale to data!!
        hist_var_sd[v].Multiply(hist_var_rd[v])
        hist_var_sb[v].Draw("HISTO,sames")
        hist_var_s[v].Draw("HISTO,sames")
        hist_var_sd[v].Draw("PE,sames")
        leg.AddEntry(hist_var_b[v],samples[back[0]]['label'],"F")
        leg.AddEntry(hist_var_s[v],samples[sel_sign[0]]['label'],"L")
        leg.AddEntry(hist_var_d[v],samples[data[0]]['label'],"P")
        OUTSTRING = OUT
        OUTSTRING += v.replace(".","_")+"_rescaled"
        drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
        drawRegion(SEL)
        leg.Draw()
        can.Update()
        can.Print(OUTSTRING+label+'.png')
        can.Print(OUTSTRING+label+'.pdf')
        can.SetLogy()
        can.Update()
        can.Print(OUTSTRING+label+'_log.png')
        can.Print(OUTSTRING+label+'_log.pdf')
        can.Close()

    for v in var_to_model:
        can = TCanvas("can","can",900,800)
        can.SetRightMargin(0.05)
        can.cd()
        leg = TLegend(0.65, 0.6, 1., 0.9)
        hist_var_rb[v].SetLineColor(samples[back[0]]['linecolor'])
        hist_var_rb[v].SetMarkerColor(samples[back[0]]['linecolor'])
        hist_var_rb[v].SetFillColorAlpha(samples[back[0]]['linecolor'],0.5)
        hist_var_rd[v].SetLineColor(1)
        hist_var_rd[v].SetMarkerStyle(21)
        hist_var_rd[v].SetMarkerColor(1)
        hist_var_rb[v].Draw("HISTO,sames")
        hist_var_rd[v].Draw("PE,sames")
        leg.AddEntry(hist_var_rb[v],samples[back[0]]['label'],"F")
        leg.AddEntry(hist_var_rd[v],samples[data[0]]['label'],"P")
        OUTSTRING = OUT
        OUTSTRING += v.replace(".","_")+"_ratio"
        drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
        drawRegion(SEL)
        leg.Draw()
        can.Update()
        can.Print(OUTSTRING+label+'.png')
        can.Print(OUTSTRING+label+'.pdf')
        can.SetLogy()
        can.Update()
        can.Print(OUTSTRING+label+'_log.png')
        can.Print(OUTSTRING+label+'_log.pdf')
        can.Close()
        
    '''

    my_data.SetLineColor(1)
    my_data.SetLineStyle(2)

    b_h_smeared = TH1F("b_h_smeared",";"+variable[var]['title'],variable[var]['nbins'], variable[var]['min'],variable[var]['max'])
    b_h_smeared.Sumw2()

    b_h_shift0 = TH1F("b_h_shift0",";"+variable[var]['title'],100,-10,10)
    b_h_shift0.Sumw2()
    b_h_shift1 = TH1F("b_h_shift1",";"+variable[var]['title'],100,-10,10)
    b_h_shift1.Sumw2()
    b_h_shift2 = TH1F("b_h_shift2",";"+variable[var]['title'],100,-10,10)
    b_h_shift2.Sumw2()
    b_h_shift3 = TH1F("b_h_shift3",";"+variable[var]['title'],100,-10,10)
    b_h_shift3.Sumw2()
    b_h_shift4 = TH1F("b_h_shift4",";"+variable[var]['title'],100,-10,10)
    b_h_shift4.Sumw2()
    b_h_shift5 = TH1F("b_h_shift5",";"+variable[var]['title'],100,-10,10)
    b_h_shift5.Sumw2()
    b_h_shift6 = TH1F("b_h_shift6",";"+variable[var]['title'],100,-10,10)
    b_h_shift6.Sumw2()

    den_b_h_shift0 = TH1F("den_b_h_shift0","",len(bins_eta)-1,bins_eta)
    den_b_h_shift0.Sumw2()
    den_b_h_shift1 = TH1F("den_b_h_shift1","",len(bins_eta)-1,bins_eta)
    den_b_h_shift1.Sumw2()
    den_b_h_shift2 = TH1F("den_b_h_shift2","",len(bins_eta)-1,bins_eta)
    den_b_h_shift2.Sumw2()
    den_b_h_shift3 = TH1F("den_b_h_shift3","",len(bins_eta)-1,bins_eta)
    den_b_h_shift3.Sumw2()
    den_b_h_shift4 = TH1F("den_b_h_shift4","",len(bins_eta)-1,bins_eta)
    den_b_h_shift4.Sumw2()
    den_b_h_shift5 = TH1F("den_b_h_shift5","",len(bins_eta)-1,bins_eta)
    den_b_h_shift5.Sumw2()
    den_b_h_shift6 = TH1F("den_b_h_shift6","",len(bins_eta)-1,bins_eta)
    den_b_h_shift6.Sumw2()

    num_b_h_shift0 = TH1F("num_b_h_shift0","",len(bins_eta)-1,bins_eta)
    num_b_h_shift0.Sumw2()
    num_b_h_shift1 = TH1F("num_b_h_shift1","",len(bins_eta)-1,bins_eta)
    num_b_h_shift1.Sumw2()
    num_b_h_shift2 = TH1F("num_b_h_shift2","",len(bins_eta)-1,bins_eta)
    num_b_h_shift2.Sumw2()
    num_b_h_shift3 = TH1F("num_b_h_shift3","",len(bins_eta)-1,bins_eta)
    num_b_h_shift3.Sumw2()
    num_b_h_shift4 = TH1F("num_b_h_shift4","",len(bins_eta)-1,bins_eta)
    num_b_h_shift4.Sumw2()
    num_b_h_shift5 = TH1F("num_b_h_shift5","",len(bins_eta)-1,bins_eta)
    num_b_h_shift5.Sumw2()
    num_b_h_shift6 = TH1F("num_b_h_shift6","",len(bins_eta)-1,bins_eta)
    num_b_h_shift6.Sumw2()


    b_h_sigprob_shift0 = TH1F("b_h_sigprob_shift0",";"+variable[var]['title'],50,0,1)
    b_h_sigprob_shift0.Sumw2()
    b_h_sigprob_shift1 = TH1F("b_h_sigprob_shift1",";"+variable[var]['title'],50,0,1)
    b_h_sigprob_shift1.Sumw2()
    b_h_sigprob_shift2 = TH1F("b_h_sigprob_shift2",";"+variable[var]['title'],50,0,1)
    b_h_sigprob_shift2.Sumw2()
    b_h_sigprob_shift3 = TH1F("b_h_sigprob_shift3",";"+variable[var]['title'],50,0,1)
    b_h_sigprob_shift3.Sumw2()
    b_h_sigprob_shift4 = TH1F("b_h_sigprob_shift4",";"+variable[var]['title'],50,0,1)
    b_h_sigprob_shift4.Sumw2()
    b_h_sigprob_shift5 = TH1F("b_h_sigprob_shift5",";"+variable[var]['title'],50,0,1)
    b_h_sigprob_shift5.Sumw2()
    b_h_sigprob_shift6 = TH1F("b_h_sigprob_shift6",";"+variable[var]['title'],50,0,1)
    b_h_sigprob_shift6.Sumw2()


    #Data
    d_h_shift0 = TH1F("d_h_shift0",";"+variable[var]['title'],100,-10,10)
    d_h_shift0.Sumw2()
    d_h_shift1 = TH1F("d_h_shift1",";"+variable[var]['title'],100,-10,10)
    d_h_shift1.Sumw2()
    d_h_shift2 = TH1F("d_h_shift2",";"+variable[var]['title'],100,-10,10)
    d_h_shift2.Sumw2()
    d_h_shift3 = TH1F("d_h_shift3",";"+variable[var]['title'],100,-10,10)
    d_h_shift3.Sumw2()
    d_h_shift4 = TH1F("d_h_shift4",";"+variable[var]['title'],100,-10,10)
    d_h_shift4.Sumw2()
    d_h_shift5 = TH1F("d_h_shift5",";"+variable[var]['title'],100,-10,10)
    d_h_shift5.Sumw2()
    d_h_shift6 = TH1F("d_h_shift6",";"+variable[var]['title'],100,-10,10)
    d_h_shift6.Sumw2()

    den_d_h_shift0 = TH1F("den_d_h_shift0","",len(bins_eta)-1,bins_eta)
    den_d_h_shift0.Sumw2()
    den_d_h_shift1 = TH1F("den_d_h_shift1","",len(bins_eta)-1,bins_eta)
    den_d_h_shift1.Sumw2()
    den_d_h_shift2 = TH1F("den_d_h_shift2","",len(bins_eta)-1,bins_eta)
    den_d_h_shift2.Sumw2()
    den_d_h_shift3 = TH1F("den_d_h_shift3","",len(bins_eta)-1,bins_eta)
    den_d_h_shift3.Sumw2()
    den_d_h_shift4 = TH1F("den_d_h_shift4","",len(bins_eta)-1,bins_eta)
    den_d_h_shift4.Sumw2()
    den_d_h_shift5 = TH1F("den_d_h_shift5","",len(bins_eta)-1,bins_eta)
    den_d_h_shift5.Sumw2()
    den_d_h_shift6 = TH1F("den_d_h_shift6","",len(bins_eta)-1,bins_eta)
    den_d_h_shift6.Sumw2()

    num_d_h_shift0 = TH1F("num_d_h_shift0","",len(bins_eta)-1,bins_eta)
    num_d_h_shift0.Sumw2()
    num_d_h_shift1 = TH1F("num_d_h_shift1","",len(bins_eta)-1,bins_eta)
    num_d_h_shift1.Sumw2()
    num_d_h_shift2 = TH1F("num_d_h_shift2","",len(bins_eta)-1,bins_eta)
    num_d_h_shift2.Sumw2()
    num_d_h_shift3 = TH1F("num_d_h_shift3","",len(bins_eta)-1,bins_eta)
    num_d_h_shift3.Sumw2()
    num_d_h_shift4 = TH1F("num_d_h_shift4","",len(bins_eta)-1,bins_eta)
    num_d_h_shift4.Sumw2()
    num_d_h_shift5 = TH1F("num_d_h_shift5","",len(bins_eta)-1,bins_eta)
    num_d_h_shift5.Sumw2()
    num_d_h_shift6 = TH1F("num_d_h_shift6","",len(bins_eta)-1,bins_eta)
    num_d_h_shift6.Sumw2()

    d_h_sigprob_shift0 = TH1F("d_h_sigprob_shift0",";"+variable[var]['title'],50,0,1)
    d_h_sigprob_shift0.Sumw2()
    d_h_sigprob_shift1 = TH1F("d_h_sigprob_shift1",";"+variable[var]['title'],50,0,1)
    d_h_sigprob_shift1.Sumw2()
    d_h_sigprob_shift2 = TH1F("d_h_sigprob_shift2",";"+variable[var]['title'],50,0,1)
    d_h_sigprob_shift2.Sumw2()
    d_h_sigprob_shift3 = TH1F("d_h_sigprob_shift3",";"+variable[var]['title'],50,0,1)
    d_h_sigprob_shift3.Sumw2()
    d_h_sigprob_shift4 = TH1F("d_h_sigprob_shift4",";"+variable[var]['title'],50,0,1)
    d_h_sigprob_shift4.Sumw2()
    d_h_sigprob_shift5 = TH1F("d_h_sigprob_shift5",";"+variable[var]['title'],50,0,1)
    d_h_sigprob_shift5.Sumw2()
    d_h_sigprob_shift6 = TH1F("d_h_sigprob_shift6",";"+variable[var]['title'],50,0,1)
    d_h_sigprob_shift6.Sumw2()

    #Signal
    s_h_sigprob = {}
    s_h_time = {}
    num_s_h = {}
    den_s_h = {}
    s_graph = {}
    for s in sign:
        s_h_time[s] = TH1F("s_h_time_"+s,";"+variable[var]['title'],100,-10,10)
        s_h_sigprob[s] = TH1F("s_h_sigprob_"+s,";"+variable[var]['title'],50,0,1)
        num_s_h[s] = TH1F("num_s_h_"+s,"",len(bins_eta)-1,bins_eta)
        den_s_h[s] = TH1F("den_s_h_"+s,"",len(bins_eta)-1,bins_eta)
        s_graph[s] = TGraphAsymmErrors()
        s_h_time[s].Sumw2()
        s_h_sigprob[s].Sumw2()
        num_s_h[s].Sumw2()
        den_s_h[s].Sumw2()


    smear_cb = my_data.Clone("smear_cb")
    smear_cb.SetParameter(0,my_data.GetParameter(0))
    smear_cb.SetParameter(1,0)#do not modify the mean to match data!!!#my_data.GetParameter(1)-my_back.GetParameter(1)) 
    smear_cb.SetParameter(2, math.sqrt( abs(my_data.GetParameter(2)**2 - my_back.GetParameter(2)**2)) )
    smear_cb.SetParameter(3,my_data.GetParameter(3))
    smear_cb.SetParameter(4,my_data.GetParameter(4))
    smear_cb.SetLineColor(4)

    #Gaussian to shift/smear other inputs
    for v in var_to_model:
        shift_b[v] = my_back_var[v].Clone(v+"_b")
        shift_b[v].SetParameter(0,my_back_var[v].GetParameter(0))
        shift_b[v].SetParameter(1,my_sign_var[v].GetParameter(1)-my_back_var[v].GetParameter(1)) 
        shift_b[v].SetParameter(2, math.sqrt( abs(my_sign_var[v].GetParameter(2)**2 - my_back_var[v].GetParameter(2)**2)) )

        shift_d[v] = my_data_var[v].Clone(v+"_d")
        shift_d[v].SetParameter(0,my_data_var[v].GetParameter(0))
        shift_d[v].SetParameter(1,my_sign_var[v].GetParameter(1)-my_data_var[v].GetParameter(1)) 
        shift_d[v].SetParameter(2, math.sqrt( abs(my_sign_var[v].GetParameter(2)**2 - my_data_var[v].GetParameter(2)**2)) )


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

    s_timeRecHitsEB = {}
    s_EventWeight   = {}
    s_probs         = {}
    s_eta           = {}
    s_nTrackConstituents = {}
    s_nSelectedTracks = {}
    s_timeRecHitsEB = {}
    s_eFracRecHitsEB = {}
    s_nRecHitsEB = {}
    s_sig1EB = {}
    s_sig2EB = {}
    s_ptDEB = {}
    s_cHadEFrac = {}
    s_nHadEFrac = {}
    s_eleEFrac = {}
    s_photonEFrac = {}
    s_ptAllTracks = {}
    s_ptAllPVTracks = {}
    s_alphaMax = {}
    s_betaMax = {}
    s_gammaMax = {}
    s_gammaMaxEM = {}
    s_gammaMaxHadronic = {}
    s_gammaMaxET = {}
    s_minDeltaRAllTracks = {}
    s_minDeltaRPVTracks = {}

    for s in sign:
        s_timeRecHitsEB[s] = np.array([])
        s_EventWeight[s] = np.array([])
        s_probs[s] = np.array([])
        s_eta[s] = np.array([])
        s_nTrackConstituents[s] = np.array([])
        s_nSelectedTracks[s] = np.array([])
        s_timeRecHitsEB[s] = np.array([])
        s_eFracRecHitsEB[s] = np.array([])
        s_nRecHitsEB[s] = np.array([])
        s_sig1EB[s] = np.array([])
        s_sig2EB[s] = np.array([])
        s_ptDEB[s] = np.array([])
        s_cHadEFrac[s] = np.array([])
        s_nHadEFrac[s] = np.array([])
        s_eleEFrac[s] = np.array([])
        s_photonEFrac[s] = np.array([])
        s_ptAllTracks[s] = np.array([])
        s_ptAllPVTracks[s] = np.array([])
        s_alphaMax[s] = np.array([])
        s_betaMax[s] = np.array([])
        s_gammaMax[s] = np.array([])
        s_gammaMaxEM[s] = np.array([])
        s_gammaMaxHadronic[s] = np.array([])
        s_gammaMaxET[s] = np.array([])
        s_minDeltaRAllTracks[s] = np.array([])
        s_minDeltaRPVTracks[s] = np.array([])

    #TODO
    #Don't care about this MC, took signal
    #Do the smearing
    #Shift data time and check if the DNN score increases
    #Plot input variables in data and signal, check if they look any similar?
    #Tag eff as a function of eta?
    #Will absorb other discrepancies
    #Compare if similar in data/MC

    #Shift back
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

                for e in range(0,tree.GetEntries()):
                    tree.GetEntry(e)
                    neg_jets = getattr(tree,"JetsNegative")
                    met = getattr(tree,"MEt")

                    #ZtoLLPho selections
                    if abs(tree.Z_mass-91.2)<10.:
                        continue
                    if tree.Z_pho_tight==False:
                        continue

                    if tree.MinJetMetDPhi<=0.5:
                        continue

                    for n in range(min(1,neg_jets.size())):
                        if neg_jets[n].eta<=-1:
                            continue
                        if neg_jets[n].eta>=1:
                            continue
                        if ERA=="2017":
                            if neg_jets[n].phi>=2.7: continue
                        if ERA=="2018":
                            if neg_jets[n].phi>=0.4 and neg_jets[n].phi<0.9: continue

                        time_rnd = neg_jets[n].timeRecHitsEB + smear_cb.GetRandom()
                        #cHadEFrac_rnd = neg_jets[n].cHadEFrac + gRandom.Gaus(shift_b["cHadEFrac"].GetParameter(1),shift_b["cHadEFrac"].GetParameter(2))
                        nHadEFrac_rnd = neg_jets[n].nHadEFrac + gRandom.Gaus(shift_b["nHadEFrac"].GetParameter(1),shift_b["nHadEFrac"].GetParameter(2))
                        photonEFrac_rnd = neg_jets[n].photonEFrac + gRandom.Gaus(shift_b["photonEFrac"].GetParameter(1),shift_b["photonEFrac"].GetParameter(2))
                        eFracRecHitsEB_rnd = neg_jets[n].eFracRecHitsEB + gRandom.Gaus(shift_b["eFracRecHitsEB"].GetParameter(1),shift_b["eFracRecHitsEB"].GetParameter(2))#shift_b["eFracRecHitsEB"].GetRandom()

                        nRecHitsEB_rnd = neg_jets[n].nRecHitsEB + gRandom.Gaus(shift_b["nRecHitsEB"].GetParameter(1),shift_b["nRecHitsEB"].GetParameter(2))
                        ptDEB_rnd = neg_jets[n].ptDEB + gRandom.Gaus(shift_b["ptDEB"].GetParameter(1),shift_b["ptDEB"].GetParameter(2))
                        b_tmp_weight.append(tree.EventWeight * tree.PUReWeight * tree_weight)
                        b_tmp_time.append(time_rnd)
                        b_tmp_nTrackConstituents.append(neg_jets[n].nTrackConstituents)
                        b_tmp_nSelectedTracks.append(neg_jets[n].nSelectedTracks)
                        b_tmp_timeRecHitsEB.append(neg_jets[n].timeRecHitsEB)
                        #Scale:
                        b_tmp_eFracRecHitsEB.append(eFracRecHitsEB_rnd)#(neg_jets[n].eFracRecHitsEB)
                        b_tmp_nRecHitsEB.append(nRecHitsEB_rnd)#(neg_jets[n].nRecHitsEB)
                        b_tmp_sig1EB.append(neg_jets[n].sig1EB)
                        b_tmp_sig2EB.append(neg_jets[n].sig2EB)
                        b_tmp_ptDEB.append(ptDEB_rnd)#(neg_jets[n].ptDEB)
                        #Scale:
                        b_tmp_cHadEFrac.append(neg_jets[n].cHadEFrac)#(cHadEFrac_rnd)#
                        b_tmp_nHadEFrac.append(nHadEFrac_rnd)#(neg_jets[n].nHadEFrac)
                        b_tmp_eleEFrac.append(neg_jets[n].eleEFrac)
                        b_tmp_photonEFrac.append(photonEFrac_rnd)#(neg_jets[n].photonEFrac)
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
    b_probs_smear = model.predict(b_X_smear)
    b_probs_shift1 = model.predict(b_X_shift1)
    b_probs_shift2 = model.predict(b_X_shift2)
    b_probs_shift3 = model.predict(b_X_shift3)
    b_probs_shift4 = model.predict(b_X_shift4)
    b_probs_shift5 = model.predict(b_X_shift5)
    b_probs_shift6 = model.predict(b_X_shift6)

    m_b_time_smear  = (b_timeSmeared  > -1 )
    m_b_time_shift1 = (b_timeShift1   > -1)
    m_b_time_shift2 = (b_timeShift2   > -1)
    m_b_time_shift3 = (b_timeShift3   > -1)
    m_b_time_shift4 = (b_timeShift4   > -1)
    m_b_time_shift5 = (b_timeShift5   > -1)
    m_b_time_shift6 = (b_timeShift6   > -1)

    m_b_probs_smear  = (b_probs_smear[:,1]  > 0.996)
    m_b_probs_shift1 = (b_probs_shift1[:,1] > 0.996)
    m_b_probs_shift2 = (b_probs_shift2[:,1] > 0.996)
    m_b_probs_shift3 = (b_probs_shift3[:,1] > 0.996)
    m_b_probs_shift4 = (b_probs_shift4[:,1] > 0.996)
    m_b_probs_shift5 = (b_probs_shift5[:,1] > 0.996)
    m_b_probs_shift6 = (b_probs_shift6[:,1] > 0.996)


    root_numpy.fill_hist(b_h_sigprob_shift0,b_probs_smear[:,1][m_b_time_smear],   b_EventWeight[m_b_time_smear])
    root_numpy.fill_hist(b_h_sigprob_shift1,b_probs_shift1[:,1][m_b_time_shift1], b_EventWeight[m_b_time_shift1])
    root_numpy.fill_hist(b_h_sigprob_shift2,b_probs_shift2[:,1][m_b_time_shift2], b_EventWeight[m_b_time_shift2])
    root_numpy.fill_hist(b_h_sigprob_shift3,b_probs_shift3[:,1][m_b_time_shift3], b_EventWeight[m_b_time_shift3])
    root_numpy.fill_hist(b_h_sigprob_shift4,b_probs_shift4[:,1][m_b_time_shift4], b_EventWeight[m_b_time_shift4])
    root_numpy.fill_hist(b_h_sigprob_shift5,b_probs_shift5[:,1][m_b_time_shift5], b_EventWeight[m_b_time_shift5])
    root_numpy.fill_hist(b_h_sigprob_shift6,b_probs_shift6[:,1][m_b_time_shift6], b_EventWeight[m_b_time_shift6])

    root_numpy.fill_hist(b_h_smeared,  b_timeSmeared[m_b_time_smear], b_EventWeight[m_b_time_smear])
    root_numpy.fill_hist(b_h_shift0, b_timeSmeared[m_b_time_smear], b_EventWeight[m_b_time_smear])
    root_numpy.fill_hist(b_h_shift1, b_timeShift1[m_b_time_shift1], b_EventWeight[m_b_time_shift1])
    root_numpy.fill_hist(b_h_shift2, b_timeShift2[m_b_time_shift2], b_EventWeight[m_b_time_shift2])
    root_numpy.fill_hist(b_h_shift3, b_timeShift3[m_b_time_shift3], b_EventWeight[m_b_time_shift3])
    root_numpy.fill_hist(b_h_shift4, b_timeShift4[m_b_time_shift4], b_EventWeight[m_b_time_shift4])
    root_numpy.fill_hist(b_h_shift5, b_timeShift5[m_b_time_shift5], b_EventWeight[m_b_time_shift5])
    root_numpy.fill_hist(b_h_shift6, b_timeShift6[m_b_time_shift6], b_EventWeight[m_b_time_shift6])

    root_numpy.fill_hist(den_b_h_shift0, b_eta[m_b_time_smear],  b_EventWeight[m_b_time_smear])
    root_numpy.fill_hist(den_b_h_shift1, b_eta[m_b_time_shift1], b_EventWeight[m_b_time_shift1])
    root_numpy.fill_hist(den_b_h_shift2, b_eta[m_b_time_shift2], b_EventWeight[m_b_time_shift2])
    root_numpy.fill_hist(den_b_h_shift3, b_eta[m_b_time_shift3], b_EventWeight[m_b_time_shift3])
    root_numpy.fill_hist(den_b_h_shift4, b_eta[m_b_time_shift4], b_EventWeight[m_b_time_shift4])
    root_numpy.fill_hist(den_b_h_shift5, b_eta[m_b_time_shift5], b_EventWeight[m_b_time_shift5])
    root_numpy.fill_hist(den_b_h_shift6, b_eta[m_b_time_shift6], b_EventWeight[m_b_time_shift6])

    
    root_numpy.fill_hist(num_b_h_shift0, b_eta[np.logical_and(m_b_probs_smear,m_b_time_smear)],b_EventWeight[np.logical_and(m_b_probs_smear,m_b_time_smear)])
    root_numpy.fill_hist(num_b_h_shift1, b_eta[np.logical_and(m_b_probs_shift1,m_b_time_shift1)],b_EventWeight[np.logical_and(m_b_probs_shift1,m_b_time_shift1)])
    root_numpy.fill_hist(num_b_h_shift2, b_eta[np.logical_and(m_b_probs_shift2,m_b_time_shift2)],b_EventWeight[np.logical_and(m_b_probs_shift2,m_b_time_shift2)])
    root_numpy.fill_hist(num_b_h_shift3, b_eta[np.logical_and(m_b_probs_shift3,m_b_time_shift3)],b_EventWeight[np.logical_and(m_b_probs_shift3,m_b_time_shift3)])
    root_numpy.fill_hist(num_b_h_shift4, b_eta[np.logical_and(m_b_probs_shift4,m_b_time_shift4)],b_EventWeight[np.logical_and(m_b_probs_shift4,m_b_time_shift4)])
    root_numpy.fill_hist(num_b_h_shift5, b_eta[np.logical_and(m_b_probs_shift5,m_b_time_shift5)],b_EventWeight[np.logical_and(m_b_probs_shift5,m_b_time_shift5)])
    root_numpy.fill_hist(num_b_h_shift6, b_eta[np.logical_and(m_b_probs_shift6,m_b_time_shift6)],b_EventWeight[np.logical_and(m_b_probs_shift6,m_b_time_shift6)])

    #Shift data
    for b in data:
        for j, ss in enumerate(samples[b]['files']):
            print "Performing time shift for ", ss
            if 'Muon' in ss:
                d_tmp_file = TFile(MAIN_MM+ss+".root","READ")
            if 'Electron' in ss or 'EGamma' in ss:
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

            for e in range(0,tree.GetEntries()):
                tree.GetEntry(e)
                neg_jets = getattr(tree,"JetsNegative")
                met = getattr(tree,"MEt")

                if tree.MinJetMetDPhi<=0.5:
                    continue

                #ZtoLLPho selections
                if abs(tree.Z_mass-91.2)<10.:
                    continue
                if tree.Z_pho_tight==False:
                    continue

                for n in range(min(1,neg_jets.size())):
                    if neg_jets[n].eta<=-1:
                        continue
                    if neg_jets[n].eta>=1:
                        continue
                    if ERA=="2017":
                        if neg_jets[n].phi>=2.7: continue
                    if ERA=="2018":
                        if neg_jets[n].phi>=0.4 and neg_jets[n].phi<0.9: continue

                    #cHadEFrac_rnd = neg_jets[n].cHadEFrac + gRandom.Gaus(shift_d["cHadEFrac"].GetParameter(1),shift_d["cHadEFrac"].GetParameter(2))
                    nHadEFrac_rnd = neg_jets[n].nHadEFrac + gRandom.Gaus(shift_d["nHadEFrac"].GetParameter(1),shift_d["nHadEFrac"].GetParameter(2))
                    photonEFrac_rnd = neg_jets[n].photonEFrac + gRandom.Gaus(shift_d["photonEFrac"].GetParameter(1),shift_d["photonEFrac"].GetParameter(2))
                    eFracRecHitsEB_rnd = neg_jets[n].eFracRecHitsEB + gRandom.Gaus(shift_d["eFracRecHitsEB"].GetParameter(1),shift_d["eFracRecHitsEB"].GetParameter(2))#shift_d["eFracRecHitsEB"].GetRandom()
                    nRecHitsEB_rnd = neg_jets[n].nRecHitsEB + gRandom.Gaus(shift_d["nRecHitsEB"].GetParameter(1),shift_d["nRecHitsEB"].GetParameter(2))
                    ptDEB_rnd = neg_jets[n].ptDEB + gRandom.Gaus(shift_d["ptDEB"].GetParameter(1),shift_d["ptDEB"].GetParameter(2))


                    d_tmp_weight.append(tree.EventWeight * tree.PUReWeight * tree_weight)
                    d_tmp_nTrackConstituents.append(neg_jets[n].nTrackConstituents)
                    d_tmp_nSelectedTracks.append(neg_jets[n].nSelectedTracks)
                    d_tmp_timeRecHitsEB.append(neg_jets[n].timeRecHitsEB)
                    #Scale:
                    d_tmp_eFracRecHitsEB.append(eFracRecHitsEB_rnd)#(neg_jets[n].eFracRecHitsEB)
                    d_tmp_nRecHitsEB.append(nRecHitsEB_rnd)#(neg_jets[n].nRecHitsEB)
                    d_tmp_sig1EB.append(neg_jets[n].sig1EB)
                    d_tmp_sig2EB.append(neg_jets[n].sig2EB)
                    d_tmp_ptDEB.append(ptDEB_rnd)#(neg_jets[n].ptDEB)
                    #Scale:
                    d_tmp_cHadEFrac.append(neg_jets[n].cHadEFrac)#(cHadEFrac_rnd)#
                    d_tmp_nHadEFrac.append(nHadEFrac_rnd)#(neg_jets[n].nHadEFrac)
                    d_tmp_eleEFrac.append(neg_jets[n].eleEFrac)
                    d_tmp_photonEFrac.append(photonEFrac_rnd)#(neg_jets[n].photonEFrac)
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
    d_probs = model.predict(d_X)
    d_probs_shift1 = model.predict(d_X_shift1)
    d_probs_shift2 = model.predict(d_X_shift2)
    d_probs_shift3 = model.predict(d_X_shift3)
    d_probs_shift4 = model.predict(d_X_shift4)
    d_probs_shift5 = model.predict(d_X_shift5)
    d_probs_shift6 = model.predict(d_X_shift6)

    m_d_time        = (d_timeRecHitsEB > -1 )
    m_d_time_shift1 = (d_timeShift1    > -1)
    m_d_time_shift2 = (d_timeShift2    > -1)
    m_d_time_shift3 = (d_timeShift3    > -1)
    m_d_time_shift4 = (d_timeShift4    > -1)
    m_d_time_shift5 = (d_timeShift5    > -1)
    m_d_time_shift6 = (d_timeShift6    > -1)

    m_d_probs        = (d_probs[:,1]  > 0.996 )
    m_d_probs_shift1 = (d_probs_shift1[:,1] > 0.996)
    m_d_probs_shift2 = (d_probs_shift2[:,1] > 0.996)
    m_d_probs_shift3 = (d_probs_shift3[:,1] > 0.996)
    m_d_probs_shift4 = (d_probs_shift4[:,1] > 0.996)
    m_d_probs_shift5 = (d_probs_shift5[:,1] > 0.996)
    m_d_probs_shift6 = (d_probs_shift6[:,1] > 0.996)


    root_numpy.fill_hist(d_h_sigprob_shift0,d_probs[:,1][m_d_time],               d_EventWeight[m_d_time])
    root_numpy.fill_hist(d_h_sigprob_shift1,d_probs_shift1[:,1][m_d_time_shift1], d_EventWeight[m_d_time_shift1])
    root_numpy.fill_hist(d_h_sigprob_shift2,d_probs_shift2[:,1][m_d_time_shift2], d_EventWeight[m_d_time_shift2])
    root_numpy.fill_hist(d_h_sigprob_shift3,d_probs_shift3[:,1][m_d_time_shift3], d_EventWeight[m_d_time_shift3])
    root_numpy.fill_hist(d_h_sigprob_shift4,d_probs_shift4[:,1][m_d_time_shift4], d_EventWeight[m_d_time_shift4])
    root_numpy.fill_hist(d_h_sigprob_shift5,d_probs_shift5[:,1][m_d_time_shift5], d_EventWeight[m_d_time_shift5])
    root_numpy.fill_hist(d_h_sigprob_shift6,d_probs_shift6[:,1][m_d_time_shift6], d_EventWeight[m_d_time_shift6])

    root_numpy.fill_hist(d_h_shift0, d_timeRecHitsEB[m_d_time], d_EventWeight[m_d_time])
    root_numpy.fill_hist(d_h_shift1, d_timeShift1[m_d_time_shift1], d_EventWeight[m_d_time_shift1])
    root_numpy.fill_hist(d_h_shift2, d_timeShift2[m_d_time_shift2], d_EventWeight[m_d_time_shift2])
    root_numpy.fill_hist(d_h_shift3, d_timeShift3[m_d_time_shift3], d_EventWeight[m_d_time_shift3])
    root_numpy.fill_hist(d_h_shift4, d_timeShift4[m_d_time_shift4], d_EventWeight[m_d_time_shift4])
    root_numpy.fill_hist(d_h_shift5, d_timeShift5[m_d_time_shift5], d_EventWeight[m_d_time_shift5])
    root_numpy.fill_hist(d_h_shift6, d_timeShift6[m_d_time_shift6], d_EventWeight[m_d_time_shift6])

    mean_d_h_shift0 = d_h_shift0.GetMean()
    mean_d_h_shift1 = d_h_shift1.GetMean()
    mean_d_h_shift2 = d_h_shift2.GetMean()
    mean_d_h_shift3 = d_h_shift3.GetMean()
    mean_d_h_shift4 = d_h_shift4.GetMean()
    mean_d_h_shift5 = d_h_shift5.GetMean()
    mean_d_h_shift6 = d_h_shift6.GetMean()

    #print "mean_d_h_shift0 ", d_h_shift0.GetMean()
    #print "mean_d_h_shift1 ", d_h_shift1.GetMean()
    #print "mean_d_h_shift2 ", d_h_shift2.GetMean()
    #print "mean_d_h_shift3 ", d_h_shift3.GetMean()
    #print "mean_d_h_shift4 ", d_h_shift4.GetMean()
    #print "mean_d_h_shift5 ", d_h_shift5.GetMean()
    #print "mean_d_h_shift6 ", d_h_shift6.GetMean()

    root_numpy.fill_hist(den_d_h_shift0, d_eta[m_d_time],  d_EventWeight[m_d_time])
    root_numpy.fill_hist(den_d_h_shift1, d_eta[m_d_time_shift1], d_EventWeight[m_d_time_shift1])
    root_numpy.fill_hist(den_d_h_shift2, d_eta[m_d_time_shift2], d_EventWeight[m_d_time_shift2])
    root_numpy.fill_hist(den_d_h_shift3, d_eta[m_d_time_shift3], d_EventWeight[m_d_time_shift3])
    root_numpy.fill_hist(den_d_h_shift4, d_eta[m_d_time_shift4], d_EventWeight[m_d_time_shift4])
    root_numpy.fill_hist(den_d_h_shift5, d_eta[m_d_time_shift5], d_EventWeight[m_d_time_shift5])
    root_numpy.fill_hist(den_d_h_shift6, d_eta[m_d_time_shift6], d_EventWeight[m_d_time_shift6])

    root_numpy.fill_hist(num_d_h_shift0, d_eta[np.logical_and(m_d_probs,m_d_time)],d_EventWeight[np.logical_and(m_d_probs,m_d_time)])
    root_numpy.fill_hist(num_d_h_shift1, d_eta[np.logical_and(m_d_probs_shift1,m_d_time_shift1)],d_EventWeight[np.logical_and(m_d_probs_shift1,m_d_time_shift1)])
    root_numpy.fill_hist(num_d_h_shift2, d_eta[np.logical_and(m_d_probs_shift2,m_d_time_shift2)],d_EventWeight[np.logical_and(m_d_probs_shift2,m_d_time_shift2)])
    root_numpy.fill_hist(num_d_h_shift3, d_eta[np.logical_and(m_d_probs_shift3,m_d_time_shift3)],d_EventWeight[np.logical_and(m_d_probs_shift3,m_d_time_shift3)])
    root_numpy.fill_hist(num_d_h_shift4, d_eta[np.logical_and(m_d_probs_shift4,m_d_time_shift4)],d_EventWeight[np.logical_and(m_d_probs_shift4,m_d_time_shift4)])
    root_numpy.fill_hist(num_d_h_shift5, d_eta[np.logical_and(m_d_probs_shift5,m_d_time_shift5)],d_EventWeight[np.logical_and(m_d_probs_shift5,m_d_time_shift5)])
    root_numpy.fill_hist(num_d_h_shift6, d_eta[np.logical_and(m_d_probs_shift6,m_d_time_shift6)],d_EventWeight[np.logical_and(m_d_probs_shift6,m_d_time_shift6)])

    #Compute sign
    for s in sign:
        for j, ss in enumerate(samples[s]['files']):
            print "Performing computation for ", ss
            s_tmp_file = TFile(MAIN_MET+ss+".root","READ")
            tree = s_tmp_file.Get("tree")

            tree_weight = tree.GetWeight()
            s_tmp_weight = []
            s_tmp_timeRecHitsEB = []
            s_tmp_sigprob = []
            s_tmp_eta = []

            s_tmp_nTrackConstituents = []
            s_tmp_nSelectedTracks = []
            s_tmp_eFracRecHitsEB = []
            s_tmp_nRecHitsEB = []
            s_tmp_sig1EB = []
            s_tmp_sig2EB = []
            s_tmp_ptDEB = []
            s_tmp_cHadEFrac = []
            s_tmp_nHadEFrac = []
            s_tmp_eleEFrac = []
            s_tmp_photonEFrac = []
            s_tmp_ptAllTracks = []
            s_tmp_ptAllPVTracks = []
            s_tmp_alphaMax = []
            s_tmp_betaMax = []
            s_tmp_gammaMax = []
            s_tmp_gammaMaxEM = []
            s_tmp_gammaMaxHadronic = []
            s_tmp_gammaMaxET = []
            s_tmp_minDeltaRAllTracks = []
            s_tmp_minDeltaRPVTracks = []

            for e in range(0,tree.GetEntries()):
                tree.GetEntry(e)
                neg_jets = getattr(tree,"JetsNegative")
                met = getattr(tree,"MEt")

                if tree.MinJetMetDPhi<=0.5:
                    continue
                for n in range(min(1,neg_jets.size())):
                    if neg_jets[n].eta<=-1:
                        continue
                    if neg_jets[n].eta>=1:
                        continue
                    if ERA=="2017":
                        if neg_jets[n].phi>=2.7: continue
                    if ERA=="2018":
                        if neg_jets[n].phi>=0.4 and neg_jets[n].phi<0.9: continue
                    s_tmp_weight.append(tree.EventWeight * tree.PUReWeight * tree_weight)
                    s_tmp_timeRecHitsEB.append(neg_jets[n].timeRecHitsEB)
                    s_tmp_eta.append(neg_jets[n].eta)
                    s_tmp_sigprob.append(neg_jets[n].sigprob)
                    s_tmp_nTrackConstituents.append(neg_jets[n].nTrackConstituents)
                    s_tmp_nSelectedTracks.append(neg_jets[n].nSelectedTracks)
                    s_tmp_eFracRecHitsEB.append(neg_jets[n].eFracRecHitsEB)
                    s_tmp_nRecHitsEB.append(neg_jets[n].nRecHitsEB)
                    s_tmp_sig1EB.append(neg_jets[n].sig1EB)
                    s_tmp_sig2EB.append(neg_jets[n].sig2EB)
                    s_tmp_ptDEB.append(neg_jets[n].ptDEB)
                    s_tmp_cHadEFrac.append(neg_jets[n].cHadEFrac)
                    s_tmp_nHadEFrac.append(neg_jets[n].nHadEFrac)
                    s_tmp_eleEFrac.append(neg_jets[n].eleEFrac)
                    s_tmp_photonEFrac.append(neg_jets[n].photonEFrac)
                    s_tmp_ptAllTracks.append(neg_jets[n].ptAllTracks)
                    s_tmp_ptAllPVTracks.append(neg_jets[n].ptAllPVTracks)
                    s_tmp_alphaMax.append(neg_jets[n].alphaMax)
                    s_tmp_betaMax.append(neg_jets[n].betaMax)
                    s_tmp_gammaMax.append(neg_jets[n].gammaMax)
                    s_tmp_gammaMaxEM.append(neg_jets[n].gammaMaxEM)
                    s_tmp_gammaMaxHadronic.append(neg_jets[n].gammaMaxHadronic)
                    s_tmp_gammaMaxET.append(neg_jets[n].gammaMaxET)
                    s_tmp_minDeltaRAllTracks.append(neg_jets[n].minDeltaRAllTracks)
                    s_tmp_minDeltaRPVTracks.append(neg_jets[n].minDeltaRPVTracks)



            s_EventWeight[s] = np.concatenate((s_EventWeight[s], np.array(s_tmp_weight)))
            print "I should probably also smear signal time! But not shift the mean"
            s_timeRecHitsEB[s] = np.concatenate(( s_timeRecHitsEB[s], s_tmp_timeRecHitsEB))
            s_probs[s] = np.concatenate(( s_probs[s],s_tmp_sigprob ))
            s_eta[s] = np.concatenate(( s_eta[s],s_tmp_eta ))
            s_nTrackConstituents[s] = np.concatenate(( s_nTrackConstituents[s], s_tmp_nTrackConstituents ))
            s_nSelectedTracks[s] = np.concatenate(( s_nSelectedTracks[s], s_tmp_nSelectedTracks))
            s_eFracRecHitsEB[s] = np.concatenate(( s_eFracRecHitsEB[s], s_tmp_eFracRecHitsEB))
            s_nRecHitsEB[s] = np.concatenate(( s_nRecHitsEB[s], s_tmp_nRecHitsEB))
            s_sig1EB[s] = np.concatenate(( s_sig1EB[s], s_tmp_sig1EB))
            s_sig2EB[s] = np.concatenate(( s_sig2EB[s], s_tmp_sig2EB))
            s_ptDEB[s] = np.concatenate(( s_ptDEB[s], s_tmp_ptDEB))
            s_cHadEFrac[s] = np.concatenate(( s_cHadEFrac[s], s_tmp_cHadEFrac))
            s_nHadEFrac[s] = np.concatenate(( s_nHadEFrac[s], s_tmp_nHadEFrac))
            s_eleEFrac[s] = np.concatenate(( s_eleEFrac[s], s_tmp_eleEFrac))
            s_photonEFrac[s] = np.concatenate(( s_photonEFrac[s], s_tmp_photonEFrac))
            s_ptAllTracks[s] = np.concatenate(( s_ptAllTracks[s], s_tmp_ptAllTracks))
            s_ptAllPVTracks[s] = np.concatenate(( s_ptAllPVTracks[s], s_tmp_ptAllPVTracks))
            s_alphaMax[s] = np.concatenate(( s_alphaMax[s], s_tmp_alphaMax))
            s_betaMax[s] = np.concatenate(( s_betaMax[s], s_tmp_betaMax))
            s_gammaMax[s] = np.concatenate(( s_gammaMax[s], s_tmp_gammaMax))
            s_gammaMaxEM[s] = np.concatenate(( s_gammaMaxEM[s], s_tmp_gammaMaxEM))
            s_gammaMaxHadronic[s] = np.concatenate(( s_gammaMaxHadronic[s], s_tmp_gammaMaxHadronic))
            s_gammaMaxET[s] = np.concatenate(( s_gammaMaxET[s], s_tmp_gammaMaxET))
            s_minDeltaRAllTracks[s] = np.concatenate(( s_minDeltaRAllTracks[s], s_tmp_minDeltaRAllTracks))
            s_minDeltaRPVTracks[s] = np.concatenate(( s_minDeltaRPVTracks[s], s_tmp_minDeltaRPVTracks))


    m_s_time = {}
    m_s_probs = {}
    m_s_and = {}
    mean_s = {}

    for s in sign:
        m_s_time[s]   = (s_timeRecHitsEB[s] > -1 )
        m_s_probs[s]  = (s_probs[s]  > 0.996 )
        m_s_and[s] = np.logical_and(m_s_time[s],m_s_probs[s])
        root_numpy.fill_hist(s_h_time[s],    s_timeRecHitsEB[s][m_s_time[s]], s_EventWeight[s][m_s_time[s]])
        root_numpy.fill_hist(s_h_sigprob[s], s_probs[s][m_s_time[s]],         s_EventWeight[s][m_s_time[s]])
        root_numpy.fill_hist(den_s_h[s],     s_eta[s][m_s_time[s]],           s_EventWeight[s][m_s_time[s]])
        root_numpy.fill_hist(num_s_h[s],     s_eta[s][m_s_and[s]],            s_EventWeight[s][m_s_and[s]])
        s_graph[s].BayesDivide(num_s_h[s],den_s_h[s])
        mean_s[s] = s_h_time[s].GetMean()
        s_graph[s].SetMarkerStyle(45)
        s_graph[s].SetMarkerColor(samples[s]['linecolor'])
        s_graph[s].SetLineColor(samples[s]['linecolor'])
        s_graph[s].SetLineStyle(7)

    #Gaussian shift/smear
    #root_numpy.fill_hist(hist_var_sb["cHadEFrac"], b_cHadEFrac, b_EventWeight)
    root_numpy.fill_hist(hist_var_sb["nHadEFrac"], b_nHadEFrac, b_EventWeight)
    root_numpy.fill_hist(hist_var_sb["photonEFrac"], b_photonEFrac, b_EventWeight)
    root_numpy.fill_hist(hist_var_sb["eFracRecHitsEB"], b_eFracRecHitsEB, b_EventWeight)
    root_numpy.fill_hist(hist_var_sb["nRecHitsEB"], b_nRecHitsEB, b_EventWeight)
    root_numpy.fill_hist(hist_var_sb["ptDEB"], b_ptDEB, b_EventWeight)

    #root_numpy.fill_hist(hist_var_sd["cHadEFrac"], d_cHadEFrac, d_EventWeight)
    root_numpy.fill_hist(hist_var_sd["nHadEFrac"], d_nHadEFrac, d_EventWeight)
    root_numpy.fill_hist(hist_var_sd["photonEFrac"], d_photonEFrac, d_EventWeight)
    root_numpy.fill_hist(hist_var_sd["eFracRecHitsEB"], d_eFracRecHitsEB, d_EventWeight)
    root_numpy.fill_hist(hist_var_sd["nRecHitsEB"], d_nRecHitsEB, d_EventWeight)
    root_numpy.fill_hist(hist_var_sd["ptDEB"], d_ptDEB, d_EventWeight)

    #Plot variables shifted/smeared
    for v in var_to_model:
        jv = "Jets[0]."+v
        can = TCanvas("can","can",900,800)
        can.SetRightMargin(0.05)
        can.cd()
        
        hist_var_sb[v].Scale(1./hist_var_sb[v].Integral())
        hist_var_sd[v].Scale(1./hist_var_sd[v].Integral())

        leg = TLegend(0.75, 0.79, 1., 0.99)
        leg.SetFillStyle(0)
        hist_var_sb[v].SetLineColor(8)#(samples[back[0]]['linecolor']+2)
        hist_var_sb[v].SetLineWidth(2)
        hist_var_sb[v].SetMarkerColor(8)#(samples[back[0]]['linecolor']+2)
        #hist_var_sb[v].SetFillColorAlpha(samples[back[0]]['linecolor']+2,0.5)
        hist_var_sd[v].SetLineColor(4)
        hist_var_sd[v].SetMarkerStyle(22)
        hist_var_sd[v].SetMarkerColor(4)
        hist_var_sd[v].SetLineWidth(2)

        hist_var_b[v].Draw("HISTO,sames")
        hist_var_d[v].Draw("PE,sames")
        hist_var_s[v].Draw("HISTO,sames")
        hist_var_sb[v].Draw("HISTO,sames")
        hist_var_sd[v].Draw("PE,sames")

        leg.AddEntry(hist_var_s[v],samples[sel_sign[0]]['label'],"L")
        leg.AddEntry(hist_var_b[v],samples[back[0]]['label'],"F")
        leg.AddEntry(hist_var_sb[v],"MC shifted","L")
        leg.AddEntry(hist_var_d[v],samples[data[0]]['label'],"P")
        leg.AddEntry(hist_var_sd[v],"data shifted","P")
        OUTSTRING = OUT
        OUTSTRING += v.replace(".","_")+"_shifted"
        drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True,left_marg_CMS=0.1,left_marg_LUMI=0.75)
        drawRegion(SEL)
        leg.Draw()
        can.Update()
        can.Print(OUTSTRING+label+'.png')
        can.Print(OUTSTRING+label+'.pdf')
        can.SetLogy()
        can.Update()
        can.Print(OUTSTRING+label+'_log.png')
        can.Print(OUTSTRING+label+'_log.pdf')
        can.Close()


    #Plot back
    b_h_smeared.SetLineColor(856)
    b_h_smeared.SetLineWidth(2)
    colors = [2,801,856,602,920,881,824]
    colors = [1,801,856,602,2,881,824]

    b_h_shift0.SetMarkerColor(colors[0])
    b_h_shift1.SetMarkerColor(colors[1])
    b_h_shift2.SetMarkerColor(colors[2])
    b_h_shift3.SetMarkerColor(colors[3])
    b_h_shift4.SetMarkerColor(colors[4])
    b_h_shift5.SetMarkerColor(colors[5])
    b_h_shift6.SetMarkerColor(colors[6])
    b_h_shift0.SetLineColor(colors[0])
    b_h_shift1.SetLineColor(colors[1])
    b_h_shift2.SetLineColor(colors[2])
    b_h_shift3.SetLineColor(colors[3])
    b_h_shift4.SetLineColor(colors[4])
    b_h_shift5.SetLineColor(colors[5])
    b_h_shift6.SetLineColor(colors[6])
    b_h_shift0.SetLineWidth(2)
    b_h_shift1.SetLineWidth(2)
    b_h_shift2.SetLineWidth(2)
    b_h_shift3.SetLineWidth(2)
    b_h_shift4.SetLineWidth(2)
    b_h_shift5.SetLineWidth(2)
    b_h_shift6.SetLineWidth(2)

    b_h_sigprob_shift0.SetMarkerColor(colors[0])
    b_h_sigprob_shift1.SetMarkerColor(colors[1])
    b_h_sigprob_shift2.SetMarkerColor(colors[2])
    b_h_sigprob_shift3.SetMarkerColor(colors[3])
    b_h_sigprob_shift4.SetMarkerColor(colors[4])
    b_h_sigprob_shift5.SetMarkerColor(colors[5])
    b_h_sigprob_shift6.SetMarkerColor(colors[6])
    b_h_sigprob_shift0.SetMarkerStyle(22)
    b_h_sigprob_shift1.SetMarkerStyle(22)
    b_h_sigprob_shift2.SetMarkerStyle(22)
    b_h_sigprob_shift3.SetMarkerStyle(22)
    b_h_sigprob_shift4.SetMarkerStyle(22)
    b_h_sigprob_shift5.SetMarkerStyle(22)
    b_h_sigprob_shift6.SetMarkerStyle(22)
    b_h_sigprob_shift0.SetLineColor(colors[0])
    b_h_sigprob_shift1.SetLineColor(colors[1])
    b_h_sigprob_shift2.SetLineColor(colors[2])
    b_h_sigprob_shift3.SetLineColor(colors[3])
    b_h_sigprob_shift4.SetLineColor(colors[4])
    b_h_sigprob_shift5.SetLineColor(colors[5])
    b_h_sigprob_shift6.SetLineColor(colors[6])
    b_h_sigprob_shift0.SetLineWidth(2)
    b_h_sigprob_shift1.SetLineWidth(2)
    b_h_sigprob_shift2.SetLineWidth(2)
    b_h_sigprob_shift3.SetLineWidth(2)
    b_h_sigprob_shift4.SetLineWidth(2)
    b_h_sigprob_shift5.SetLineWidth(2)
    b_h_sigprob_shift6.SetLineWidth(2)


    if scale:
        b_h_smeared.Scale(1./b_h_smeared.Integral())
        #b_h_shift0.Scale(1./b_h_shift0.Integral())
        #b_h_shift1.Scale(1./b_h_shift1.Integral())
        #b_h_shift2.Scale(1./b_h_shift2.Integral())
        #b_h_shift3.Scale(1./b_h_shift3.Integral())
        #b_h_shift4.Scale(1./b_h_shift4.Integral())
        #b_h_shift5.Scale(1./b_h_shift5.Integral())
        #b_h_shift6.Scale(1./b_h_shift6.Integral())

    can = TCanvas("can","can",900,800)
    can.SetRightMargin(0.05)
    can.cd()
    leg = TLegend(0.65, 0.6, 1., 0.9)
    for s in back+data:
        if s in back:
            hist[s].Draw("HISTO,sames")
            leg.AddEntry(hist[s],samples[s]['label'],"F")
        if s in data:
            hist[s].Draw("PE,sames")
            leg.AddEntry(hist[s],samples[s]['label'],"P")
        hist[s].GetYaxis().SetTitle("Events")

    b_h_smeared.Draw("HISTO,sames")
    #b_h_shift1.Draw("HISTO,sames")
    #b_h_shift2.Draw("HISTO,sames")
    #b_h_shift3.Draw("HISTO,sames")
    #b_h_shift4.Draw("HISTO,sames")
    #b_h_shift5.Draw("HISTO,sames")
    #b_h_shift6.Draw("HISTO,sames")
    my_data.Draw("L,sames")
    my_back.Draw("L,sames")

    leg.AddEntry(my_back,"back fit: mean "+str(round(my_back.GetParameter(1),2))+", #sigma "+str(round(my_back.GetParameter(2),2)),"L")
    leg.AddEntry(my_data,"data fit: mean "+str(round(my_data.GetParameter(1),2))+", #sigma "+str(round(my_data.GetParameter(2),2)),"L")
    leg.AddEntry(b_h_smeared,"MC smeared","L")

    OUTSTRING = OUT
    OUTSTRING += "jet_time"
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    #drawAnalysis("LL"+CHAN)
    drawRegion(SEL)
    leg.Draw()
    can.Update()
    can.Print(OUTSTRING+label+'.png')
    can.Print(OUTSTRING+label+'.pdf')
    can.SetLogy()
    can.Update()
    can.Print(OUTSTRING+label+'_log.png')
    can.Print(OUTSTRING+label+'_log.pdf')
    can.Close()


    ##

    can2 = TCanvas("can2","can2",900,800)
    can2.SetRightMargin(0.05)
    can2.cd()
    leg2 = TLegend(0.1, 0.6, 0.4, 0.9)
    leg2.SetHeader(samples[back[0]]['label']+", MC")
    leg2.AddEntry(b_h_shift0,"time smeared","L")
    leg2.AddEntry(b_h_shift1,"+1 ns shift","L")
    leg2.AddEntry(b_h_shift2,"+2 ns shift","L")
    leg2.AddEntry(b_h_shift3,"+3 ns shift","L")
    leg2.AddEntry(b_h_shift4,"+4 ns shift","L")
    leg2.AddEntry(b_h_shift5,"+5 ns shift","L")
    leg2.AddEntry(b_h_shift6,"+6 ns shift","L")
    b_h_shift0.GetYaxis().SetTitle("Events")
    b_h_shift0.Draw("HISTO,sames")
    b_h_shift1.Draw("HISTO,sames")
    b_h_shift2.Draw("HISTO,sames")
    b_h_shift3.Draw("HISTO,sames")
    b_h_shift4.Draw("HISTO,sames")
    b_h_shift5.Draw("HISTO,sames")
    b_h_shift6.Draw("HISTO,sames")
    leg2.Draw()
    OUTSTRING = OUT
    OUTSTRING += "back_shifted_jet_time"
    #drawAnalysis("LL"+CHAN)
    drawCMS_simple(-1, "Simulation", ERA=ERA, onTop=True)
    drawRegion(SEL,top=0.5)
    can2.Update()
    can2.Print(OUTSTRING+label+'.png')
    can2.Print(OUTSTRING+label+'.pdf')
    can2.SetLogy()
    can2.Update()
    can2.Print(OUTSTRING+label+'_log.png')
    can2.Print(OUTSTRING+label+'_log.pdf')
    can2.Close()


    can3 = TCanvas("can3","can3",900,800)
    can3.SetRightMargin(0.05)
    can3.cd()
    leg2 = TLegend(0.1+0.2, 0.6, 0.4+0.2, 0.9)
    leg2.SetHeader(samples[back[0]]['label']+", MC")
    leg2.AddEntry(b_h_sigprob_shift0,"time smeared","L")
    leg2.AddEntry(b_h_sigprob_shift1,"+1 ns shift","L")
    leg2.AddEntry(b_h_sigprob_shift2,"+2 ns shift","L")
    leg2.AddEntry(b_h_sigprob_shift3,"+3 ns shift","L")
    leg2.AddEntry(b_h_sigprob_shift4,"+4 ns shift","L")
    leg2.AddEntry(b_h_sigprob_shift5,"+5 ns shift","L")
    leg2.AddEntry(b_h_sigprob_shift6,"+6 ns shift","L")
    b_h_sigprob_shift6.GetXaxis().SetTitle("jet DNN score")
    b_h_sigprob_shift6.GetYaxis().SetTitle("Events")
    b_h_sigprob_shift6.Draw("HIST,sames")
    b_h_sigprob_shift0.Draw("HIST,sames")
    b_h_sigprob_shift1.Draw("HIST,sames")
    b_h_sigprob_shift2.Draw("HIST,sames")
    b_h_sigprob_shift3.Draw("HIST,sames")
    b_h_sigprob_shift4.Draw("HIST,sames")
    b_h_sigprob_shift5.Draw("HIST,sames")
    leg2.Draw()
    OUTSTRING = OUT
    OUTSTRING += "back_sigprob_shift_jet_time"
    #drawAnalysis("LL"+CHAN)
    drawCMS_simple(-1, "Simulation", ERA=ERA, onTop=True)
    drawRegion(SEL)
    can3.Update()
    can3.Print(OUTSTRING+label+'.png')
    can3.Print(OUTSTRING+label+'.pdf')
    can3.SetLogy()
    can3.Update()
    can3.Print(OUTSTRING+label+'_log.png')
    can3.Print(OUTSTRING+label+'_log.pdf')
    can3.Close()


    #time_bins = np.array([-10.,-9.,-8.,-7.,-6.,-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
    b_graph0 = TGraphAsymmErrors()
    #num_b_h_shift0.Rebin(len(bins_eta)-1,"num_b_h_shift0_2",bins_eta)
    #den_b_h_shift0.Rebin(len(bins_eta)-1,"den_b_h_shift0_2",bins_eta)
    b_graph0.BayesDivide(num_b_h_shift0,den_b_h_shift0)

    b_graph1 = TGraphAsymmErrors()
    #num_b_h_shift1.Rebin(len(bins_eta)-1,"num_b_h_shift1_2",bins_eta)
    #den_b_h_shift1.Rebin(len(bins_eta)-1,"den_b_h_shift1_2",bins_eta)
    b_graph1.BayesDivide(num_b_h_shift1,den_b_h_shift1)

    b_graph2 = TGraphAsymmErrors()
    #num_b_h_shift2.Rebin(len(bins_eta)-1,"num_b_h_shift2_2",bins_eta)
    #den_b_h_shift2.Rebin(len(bins_eta)-1,"den_b_h_shift2_2",bins_eta)
    b_graph2.BayesDivide(num_b_h_shift2,den_b_h_shift2)

    b_graph3 = TGraphAsymmErrors()
    #num_b_h_shift3.Rebin(len(bins_eta)-1,"num_b_h_shift3_2",bins_eta)
    #den_b_h_shift3.Rebin(len(bins_eta)-1,"den_b_h_shift3_2",bins_eta)
    b_graph3.BayesDivide(num_b_h_shift3,den_b_h_shift3)

    b_graph4 = TGraphAsymmErrors()
    #num_b_h_shift4.Rebin(len(bins_eta)-1,"num_b_h_shift4_2",bins_eta)
    #den_b_h_shift4.Rebin(len(bins_eta)-1,"den_b_h_shift4_2",bins_eta)
    b_graph4.BayesDivide(num_b_h_shift4,den_b_h_shift4)

    b_graph5 = TGraphAsymmErrors()
    #num_b_h_shift5.Rebin(len(bins_eta)-1,"num_b_h_shift5_2",bins_eta)
    #den_b_h_shift5.Rebin(len(bins_eta)-1,"den_b_h_shift5_2",bins_eta)
    b_graph5.BayesDivide(num_b_h_shift5,den_b_h_shift5)

    b_graph6 = TGraphAsymmErrors()
    #num_b_h_shift6.Rebin(len(bins_eta)-1,"num_b_h_shift6_2",bins_eta)
    #den_b_h_shift6.Rebin(len(bins_eta)-1,"den_b_h_shift6_2",bins_eta)
    b_graph6.BayesDivide(num_b_h_shift6,den_b_h_shift6)


    b_graph0.SetMarkerStyle(20)
    b_graph1.SetMarkerStyle(20)
    b_graph2.SetMarkerStyle(20)
    b_graph3.SetMarkerStyle(20)
    b_graph4.SetMarkerStyle(20)
    b_graph5.SetMarkerStyle(20)
    b_graph6.SetMarkerStyle(20)
    b_graph0.SetMarkerColor(colors[0])
    b_graph1.SetMarkerColor(colors[1])
    b_graph2.SetMarkerColor(colors[2])
    b_graph3.SetMarkerColor(colors[3])
    b_graph4.SetMarkerColor(colors[4])
    b_graph5.SetMarkerColor(colors[5])
    b_graph6.SetMarkerColor(colors[6])
    b_graph0.SetLineColor(colors[0])
    b_graph1.SetLineColor(colors[1])
    b_graph2.SetLineColor(colors[2])
    b_graph3.SetLineColor(colors[3])
    b_graph4.SetLineColor(colors[4])
    b_graph5.SetLineColor(colors[5])
    b_graph6.SetLineColor(colors[6])
    b_graph0.SetLineWidth(2)
    b_graph1.SetLineWidth(2)
    b_graph2.SetLineWidth(2)
    b_graph3.SetLineWidth(2)
    b_graph4.SetLineWidth(2)
    b_graph5.SetLineWidth(2)
    b_graph6.SetLineWidth(2)

    can4 = TCanvas("can4","can4",900,800)
    can4.SetRightMargin(0.05)
    can4.cd()
    leg2 = TLegend(0.1+0.4, 0.6, 0.4+0.4, 0.9)
    leg2.SetHeader(samples[back[0]]['label']+", MC")
    leg2.AddEntry(b_graph0,"time smeared","PL")
    leg2.AddEntry(b_graph1,"+1 ns shift","PL")
    leg2.AddEntry(b_graph2,"+2 ns shift","PL")
    leg2.AddEntry(b_graph3,"+3 ns shift","PL")
    leg2.AddEntry(b_graph4,"+4 ns shift","PL")
    leg2.AddEntry(b_graph5,"+5 ns shift","PL")
    leg2.AddEntry(b_graph6,"+6 ns shift","PL")
    b_graph0.Draw("AP")
    b_graph1.Draw("P,sames")
    b_graph2.Draw("P,sames")
    b_graph3.Draw("P,sames")
    b_graph4.Draw("P,sames")
    b_graph5.Draw("P,sames")
    b_graph6.Draw("P,sames")
    b_graph0.SetMaximum(1.01)
    b_graph0.GetXaxis().SetTitle("jet #eta")
    b_graph0.GetYaxis().SetTitle("tag efficiency")
    leg2.Draw()
    OUTSTRING = OUT
    OUTSTRING += "back_tag_eff_shifted_jet_time"
    #drawAnalysis("LL"+CHAN)
    drawCMS_simple(-1, "Simulation", ERA=ERA, onTop=True)
    drawRegion(SEL)
    can4.Update()
    can4.Print(OUTSTRING+label+'.png')
    can4.Print(OUTSTRING+label+'.pdf')
    can4.Close()


    #Plot data
    colors = [2,801,856,602,920,881,824]
    colors = [1,801,856,602,2,881,824]

    d_h_shift0.SetMarkerColor(colors[0])
    d_h_shift1.SetMarkerColor(colors[1])
    d_h_shift2.SetMarkerColor(colors[2])
    d_h_shift3.SetMarkerColor(colors[3])
    d_h_shift4.SetMarkerColor(colors[4])
    d_h_shift5.SetMarkerColor(colors[5])
    d_h_shift6.SetMarkerColor(colors[6])
    d_h_shift0.SetLineColor(colors[0])
    d_h_shift1.SetLineColor(colors[1])
    d_h_shift2.SetLineColor(colors[2])
    d_h_shift3.SetLineColor(colors[3])
    d_h_shift4.SetLineColor(colors[4])
    d_h_shift5.SetLineColor(colors[5])
    d_h_shift6.SetLineColor(colors[6])
    d_h_shift0.SetLineWidth(2)
    d_h_shift1.SetLineWidth(2)
    d_h_shift2.SetLineWidth(2)
    d_h_shift3.SetLineWidth(2)
    d_h_shift4.SetLineWidth(2)
    d_h_shift5.SetLineWidth(2)
    d_h_shift6.SetLineWidth(2)

    d_h_sigprob_shift0.SetMarkerColor(colors[0])
    d_h_sigprob_shift1.SetMarkerColor(colors[1])
    d_h_sigprob_shift2.SetMarkerColor(colors[2])
    d_h_sigprob_shift3.SetMarkerColor(colors[3])
    d_h_sigprob_shift4.SetMarkerColor(colors[4])
    d_h_sigprob_shift5.SetMarkerColor(colors[5])
    d_h_sigprob_shift6.SetMarkerColor(colors[6])
    d_h_sigprob_shift0.SetMarkerStyle(22)
    d_h_sigprob_shift1.SetMarkerStyle(22)
    d_h_sigprob_shift2.SetMarkerStyle(22)
    d_h_sigprob_shift3.SetMarkerStyle(22)
    d_h_sigprob_shift4.SetMarkerStyle(22)
    d_h_sigprob_shift5.SetMarkerStyle(22)
    d_h_sigprob_shift6.SetMarkerStyle(22)
    d_h_sigprob_shift0.SetLineColor(colors[0])
    d_h_sigprob_shift1.SetLineColor(colors[1])
    d_h_sigprob_shift2.SetLineColor(colors[2])
    d_h_sigprob_shift3.SetLineColor(colors[3])
    d_h_sigprob_shift4.SetLineColor(colors[4])
    d_h_sigprob_shift5.SetLineColor(colors[5])
    d_h_sigprob_shift6.SetLineColor(colors[6])
    d_h_sigprob_shift0.SetLineWidth(2)
    d_h_sigprob_shift1.SetLineWidth(2)
    d_h_sigprob_shift2.SetLineWidth(2)
    d_h_sigprob_shift3.SetLineWidth(2)
    d_h_sigprob_shift4.SetLineWidth(2)
    d_h_sigprob_shift5.SetLineWidth(2)
    d_h_sigprob_shift6.SetLineWidth(2)


    can2 = TCanvas("can2","can2",900,800)
    can2.SetRightMargin(0.05)
    can2.cd()
    leg2 = TLegend(0.1, 0.6, 0.4, 0.9)
    leg2.SetHeader("Z #rightarrow ll, data")
    leg2.AddEntry(d_h_shift0,"time smeared","L")
    leg2.AddEntry(d_h_shift1,"+1 ns shift","L")
    leg2.AddEntry(d_h_shift2,"+2 ns shift","L")
    leg2.AddEntry(d_h_shift3,"+3 ns shift","L")
    leg2.AddEntry(d_h_shift4,"+4 ns shift","L")
    leg2.AddEntry(d_h_shift5,"+5 ns shift","L")
    leg2.AddEntry(d_h_shift6,"+6 ns shift","L")
    d_h_shift0.Draw("HISTO,sames")
    d_h_shift1.Draw("HISTO,sames")
    d_h_shift2.Draw("HISTO,sames")
    d_h_shift3.Draw("HISTO,sames")
    d_h_shift4.Draw("HISTO,sames")
    d_h_shift5.Draw("HISTO,sames")
    d_h_shift6.Draw("HISTO,sames")
    d_h_shift0.GetYaxis().SetTitle("Events")
    my_data.Draw("L,sames")
    my_back.Draw("L,sames")
    leg2.Draw()
    OUTSTRING = OUT
    OUTSTRING += "data_shifted_jet_time"
    #drawAnalysis("LL"+CHAN)
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawRegion(SEL,top=0.5)
    can2.Update()
    can2.Print(OUTSTRING+label+'.png')
    can2.Print(OUTSTRING+label+'.pdf')
    can2.SetLogy()
    can2.Update()
    can2.Print(OUTSTRING+label+'_log.png')
    can2.Print(OUTSTRING+label+'_log.pdf')
    can2.Close()


    can3 = TCanvas("can3","can3",900,800)
    can3.SetRightMargin(0.05)
    can3.cd()
    leg2 = TLegend(0.1+0.2, 0.6, 0.4+0.2, 0.9)
    leg2.SetHeader("Z #rightarrow ll, data")
    leg2.AddEntry(d_h_sigprob_shift0,"time smeared","L")
    leg2.AddEntry(d_h_sigprob_shift1,"+1 ns shift","L")
    leg2.AddEntry(d_h_sigprob_shift2,"+2 ns shift","L")
    leg2.AddEntry(d_h_sigprob_shift3,"+3 ns shift","L")
    leg2.AddEntry(d_h_sigprob_shift4,"+4 ns shift","L")
    leg2.AddEntry(d_h_sigprob_shift5,"+5 ns shift","L")
    leg2.AddEntry(d_h_sigprob_shift6,"+6 ns shift","L")
    d_h_sigprob_shift6.GetXaxis().SetTitle("jet DNN score")
    d_h_sigprob_shift6.GetXaxis().SetTitle("Events")
    d_h_sigprob_shift6.Draw("HIST,sames")
    d_h_sigprob_shift0.Draw("HIST,sames")
    d_h_sigprob_shift1.Draw("HIST,sames")
    d_h_sigprob_shift2.Draw("HIST,sames")
    d_h_sigprob_shift3.Draw("HIST,sames")
    d_h_sigprob_shift4.Draw("HIST,sames")
    d_h_sigprob_shift5.Draw("HIST,sames")
    leg2.Draw()
    OUTSTRING = OUT
    OUTSTRING += "data_sigprob_shifted_jet_time"
    #drawAnalysis("LL"+CHAN)
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawRegion(SEL)
    can3.Update()
    can3.Print(OUTSTRING+label+'.png')
    can3.Print(OUTSTRING+label+'.pdf')
    can3.SetLogy()
    can3.Update()
    can3.Print(OUTSTRING+label+'_log.png')
    can3.Print(OUTSTRING+label+'_log.pdf')
    can3.Close()


    #time_bins = np.array([-10.,-9.,-8.,-7.,-6.,-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
    d_graph0 = TGraphAsymmErrors()
    #num_d_h_shift0.Rebin(len(bins_eta)-1,"num_d_h_shift0_2",bins_eta)
    #den_d_h_shift0.Rebin(len(bins_eta)-1,"den_d_h_shift0_2",bins_eta)
    d_graph0.BayesDivide(num_d_h_shift0,den_d_h_shift0)

    d_graph1 = TGraphAsymmErrors()
    #num_d_h_shift1.Rebin(len(bins_eta)-1,"num_d_h_shift1_2",bins_eta)
    #den_d_h_shift1.Rebin(len(bins_eta)-1,"den_d_h_shift1_2",bins_eta)
    d_graph1.BayesDivide(num_d_h_shift1,den_d_h_shift1)

    d_graph2 = TGraphAsymmErrors()
    #num_d_h_shift2.Rebin(len(bins_eta)-1,"num_d_h_shift2_2",bins_eta)
    #den_d_h_shift2.Rebin(len(bins_eta)-1,"den_d_h_shift2_2",bins_eta)
    d_graph2.BayesDivide(num_d_h_shift2,den_d_h_shift2)

    d_graph3 = TGraphAsymmErrors()
    #num_d_h_shift3.Rebin(len(bins_eta)-1,"num_d_h_shift3_2",bins_eta)
    #den_d_h_shift3.Rebin(len(bins_eta)-1,"den_d_h_shift3_2",bins_eta)
    d_graph3.BayesDivide(num_d_h_shift3,den_d_h_shift3)

    d_graph4 = TGraphAsymmErrors()
    #num_d_h_shift4.Rebin(len(bins_eta)-1,"num_d_h_shift4_2",bins_eta)
    #den_d_h_shift4.Rebin(len(bins_eta)-1,"den_d_h_shift4_2",bins_eta)
    d_graph4.BayesDivide(num_d_h_shift4,den_d_h_shift4)

    d_graph5 = TGraphAsymmErrors()
    #num_d_h_shift5.Rebin(len(bins_eta)-1,"num_d_h_shift5_2",bins_eta)
    #den_d_h_shift5.Rebin(len(bins_eta)-1,"den_d_h_shift5_2",bins_eta)
    d_graph5.BayesDivide(num_d_h_shift5,den_d_h_shift5)

    d_graph6 = TGraphAsymmErrors()
    #num_d_h_shift6.Rebin(len(bins_eta)-1,"num_d_h_shift6_2",bins_eta)
    #den_d_h_shift6.Rebin(len(bins_eta)-1,"den_d_h_shift6_2",bins_eta)
    d_graph6.BayesDivide(num_d_h_shift6,den_d_h_shift6)

    d_graph0.SetMarkerStyle(24)
    d_graph1.SetMarkerStyle(24)
    d_graph2.SetMarkerStyle(24)
    d_graph3.SetMarkerStyle(24)
    d_graph4.SetMarkerStyle(24)
    d_graph5.SetMarkerStyle(24)
    d_graph6.SetMarkerStyle(24)

    d_graph0.SetLineStyle(2)
    d_graph1.SetLineStyle(2)
    d_graph2.SetLineStyle(2)
    d_graph3.SetLineStyle(2)
    d_graph4.SetLineStyle(2)
    d_graph5.SetLineStyle(2)
    d_graph6.SetLineStyle(2)

    d_graph0.SetMarkerColor(colors[0])
    d_graph1.SetMarkerColor(colors[1])
    d_graph2.SetMarkerColor(colors[2])
    d_graph3.SetMarkerColor(colors[3])
    d_graph4.SetMarkerColor(colors[4])
    d_graph5.SetMarkerColor(colors[5])
    d_graph6.SetMarkerColor(colors[6])
    d_graph0.SetLineColor(colors[0])
    d_graph1.SetLineColor(colors[1])
    d_graph2.SetLineColor(colors[2])
    d_graph3.SetLineColor(colors[3])
    d_graph4.SetLineColor(colors[4])
    d_graph5.SetLineColor(colors[5])
    d_graph6.SetLineColor(colors[6])
    d_graph0.SetLineWidth(2)
    d_graph1.SetLineWidth(2)
    d_graph2.SetLineWidth(2)
    d_graph3.SetLineWidth(2)
    d_graph4.SetLineWidth(2)
    d_graph5.SetLineWidth(2)
    d_graph6.SetLineWidth(2)

    can4 = TCanvas("can4","can4",900,800)
    can4.SetRightMargin(0.05)
    can4.cd()
    leg2 = TLegend(0.1+0.4, 0.6, 0.4+0.4, 0.9)
    leg2.SetHeader("Z #rightarrow ll, data")
    leg2.AddEntry(d_graph0,"time smeared","PL")
    leg2.AddEntry(d_graph1,"+1 ns shift","PL")
    leg2.AddEntry(d_graph2,"+2 ns shift","PL")
    leg2.AddEntry(d_graph3,"+3 ns shift","PL")
    leg2.AddEntry(d_graph4,"+4 ns shift","PL")
    leg2.AddEntry(d_graph5,"+5 ns shift","PL")
    leg2.AddEntry(d_graph6,"+6 ns shift","PL")
    d_graph0.Draw("AP")
    d_graph1.Draw("P,sames")
    d_graph2.Draw("P,sames")
    d_graph3.Draw("P,sames")
    d_graph4.Draw("P,sames")
    d_graph5.Draw("P,sames")
    d_graph6.Draw("P,sames")
    d_graph0.SetMaximum(1.01)
    d_graph0.GetXaxis().SetTitle("jet #eta")
    d_graph0.GetYaxis().SetTitle("tag efficiency")
    leg2.Draw()
    OUTSTRING = OUT
    OUTSTRING += "data_tag_eff_shifted_jet_time"
    #drawAnalysis("LL"+CHAN)
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    drawRegion(SEL)
    can4.Update()
    can4.Print(OUTSTRING+label+'.png')
    can4.Print(OUTSTRING+label+'.pdf')
    can4.Close()

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
    can5_bkg.SetRightMargin(0.05)
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
    leg2 = TLegend(0.3+0.15, 0.7, 1., 1.)
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
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True,left_marg_CMS=0.1,left_marg_LUMI=0.75)
    drawRegion(SEL)
    can5_sgn.Update()
    can5_sgn.Print(OUTSTRING+label+'.png')
    can5_sgn.Print(OUTSTRING+label+'.pdf')
    can5_sgn.Close()




    ##
    can5 = TCanvas("can5","can5",900,800)
    can5.SetRightMargin(0.05)
    can5.cd()
    leg2 = TLegend(0.1+0.55, 0.7-0.02, 0.99, 0.95-0.02)
    #leg2.AddEntry(b_graph0,"time smeared MC","PL")
    leg2.AddEntry(b_graph1,"+1 ns shift, MC bkg","PL")
    leg2.AddEntry(b_graph2,"+2 ns shift, MC bkg","PL")
    leg2.AddEntry(d_graph1,"+1 ns shift, data","PL")
    leg2.AddEntry(d_graph2,"+2 ns shift, data","PL")
    d_graph1.SetLineStyle(2)
    d_graph2.SetLineStyle(2)
    b_graph1.Draw("AP")
    b_graph2.Draw("P,sames")
    d_graph1.Draw("P,sames")
    d_graph2.Draw("P,sames")
    for s in ['SUSY_mh600_ctau500','SUSY_mh800_ctau500','SUSY_mh1800_ctau500']:
        leg2.AddEntry(s_graph[s],samples[s]['label'],"PL")
        s_graph[s].Draw("P,sames")
    b_graph1.SetMaximum(1.01)
    b_graph1.SetMinimum(0.5)
    b_graph1.GetXaxis().SetTitle("jet #eta")
    b_graph1.GetYaxis().SetTitle("tag efficiency")
    leg2.Draw()
    OUTSTRING = OUT
    OUTSTRING += "data_vs_back_vs_sign_tag_eff_shifted_jet_time"
    #drawAnalysis("LL"+CHAN)
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True,left_marg_CMS=0.1,left_marg_LUMI=0.75)
    drawRegion(SEL)
    can5.Update()
    can5.Print(OUTSTRING+label+'.png')
    can5.Print(OUTSTRING+label+'.pdf')
    can5.Close()
    ##


    #Redo ratio
    #xxxxx
    #b_graph0.BayesDivide(num_b_h_shift0,den_b_h_shift0)
    #hist_var_rb[v] = hist_var_s[v].Clone("rb_"+v)
    #hist_var_rb[v].Divide(hist_var_b[v])
    ratio1 = num_d_h_shift1.Clone("r1")
    ratio1.Divide(den_d_h_shift1)
    ratio1.Divide(num_b_h_shift1)
    ratio1.Multiply(den_b_h_shift1)

    ratio2 = num_d_h_shift2.Clone("r2")
    ratio2.Divide(den_d_h_shift2)
    ratio2.Divide(num_b_h_shift2)
    ratio2.Multiply(den_b_h_shift2)

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
    leg2 = TLegend(0.65, 0.7, 0.95, 0.9)
    #leg2.AddEntry(ratio0,"time smeared","PL")
    leg2.AddEntry(ratio1,"+1 ns time shift","PL")
    leg2.AddEntry(ratio2,"+2 ns time shift","PL")
    #leg2.AddEntry(ratio3,"+3 ns shift","PL")
    #leg2.AddEntry(ratio4,"+4 ns shift","PL")
    #leg2.AddEntry(ratio5,"+5 ns shift","PL")
    #leg2.AddEntry(ratio6,"+6 ns shift","PL")
    #ratio0.Draw("P")
    ratio1.Draw("P,sames")
    ratio2.Draw("P,sames")
    #ratio3.Draw("P,sames")
    #ratio4.Draw("P,sames")
    #ratio5.Draw("P,sames")
    #ratio6.Draw("P,sames")
    ratio0.GetXaxis().SetTitle("jet #eta")
    ratio0.GetYaxis().SetTitle("data/MC scale factor")
    ratio0.SetMinimum(0.5)
    ratio0.SetMaximum(1.5)
    ratio1.GetXaxis().SetTitle("jet #eta")
    ratio1.GetYaxis().SetTitle("data/MC scale factor")
    ratio1.SetMinimum(0.5)
    ratio1.SetMaximum(1.5)
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
    
    #Save ratio
    outfile = TFile(OUT+"data_MC_SF"+label+".root","RECREATE")
    outfile.cd()
    ratio1.Write("ratio_1ns")
    ratio2.Write("ratio_2ns")
    can.Write()
    print "Info in <TFile::Write>: root file "+OUT+"data_MC_SF"+label+".root has been created"
    outfile.Close()

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
    #leg2.AddEntry(diff0,"time smeared","PL")
    leg2.AddEntry(diff1,"+1 ns time shift","PL")
    leg2.AddEntry(diff2,"+2 ns time shift","PL")
    #leg2.AddEntry(diff3,"+3 ns time shift","PL")
    #leg2.AddEntry(diff4,"+4 ns time shift","PL")
    #leg2.AddEntry(diff5,"+5 ns shift","PL")
    #leg2.AddEntry(diff6,"+6 ns shift","PL")
    #diff0.SetTitle("")
    #diff0.Draw("APL")
    diff1.SetTitle("")
    diff1.Draw("APL")
    #diff1.Draw("PL,sames")
    diff2.Draw("PL,sames")
    #diff3.Draw("PL,sames")
    #diff4.Draw("PL,sames")
    #diff5.Draw("PL,sames")
    #diff6.Draw("PL,sames")
    #diff0.GetXaxis().SetTitle("jet #eta")
    #diff0.GetYaxis().SetTitle("data/MC tag efficiency difference (%)")
    #diff0.SetMinimum(0)
    #diff0.SetMaximum(200)
    diff1.GetXaxis().SetTitle("jet #eta")
    diff1.GetYaxis().SetTitle("data/MC tag efficiency difference (%)")
    #diff1.SetMinimum(0)
    #diff1.SetMaximum(200)
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
if SEL=="SR":
    #time_fit(var="JetsNegative[0].timeRecHitsEB",cut = "isSR",label=lab,scale=True,do_smear=False)
    #time_fit(var="JetsNegative.timeRecHitsEB",cut = "isSR && fabs(Jets.eta)<1 && MinJetMetDPhi>0.5",label=lab+"_all_jets",scale=True,do_smear=False)
    #signal_smearing(var="JetsNegative.timeRecHitsEB",cut = "isSR && fabs(Jets.eta)<1 && MinJetMetDPhi>0.5",label=lab+"_all_jets",scale=True,do_smear=True)
    draw_syst_unc()
if SEL=="ZtoLLPho":
    lab = "_G-H"
    calc_sf(var="JetsNegative[0].timeRecHitsEB",cut = "fabs(Z_mass-91.2)<10. && Z_pho_tight && MinJetMetDPhi>0.5",label=lab,scale=True,do_smear=False)
