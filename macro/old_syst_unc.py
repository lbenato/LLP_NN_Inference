#! /usr/bin/env python

import os, multiprocessing
import subprocess
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import json
import yaml
import time
import uproot
import pandas as pd
import gc
from array import array
from awkward import *
import root_numpy
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from prettytable import PrettyTable
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, Double
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, TH2D, THStack, TGraph, TGraphAsymmErrors, TF1, TEfficiency, TObjArray, TIter
from ROOT import TStyle, TCanvas, TPad, TProfile
from ROOT import TLegend, TLatex, TText, TLine, TBox
from ROOT import RDataFrame
from ctypes import c_double

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
##gROOT.ProcessLine('.L %s/src/NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v5.h+' % os.environ['CMSSW_BASE'])
##from ROOT import MEtType, JetType#LeptonType, JetType, FatJetType, MEtType, CandidateType, LorentzType
from collections import defaultdict, OrderedDict
from itertools import chain

########## SETTINGS ##########

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-v", "--variable", action="store", type="string", dest="variable", default="met_pt_nomu")#"nPV")
parser.add_option("-c", "--cut", action="store", type="string", dest="cut", default="met_test")
parser.add_option("-C", "--compare", action="store", type="string", dest="compare", default="")
parser.add_option("-D", "--drawsignal", action="store_true", dest="drawsignal", default=False)
parser.add_option("-n", "--normalized", action="store_true", dest="normalized", default=False)
parser.add_option("-d", "--dataset", action="store", type="string", dest="dataset", default="mu")#"mu3nPV"
parser.add_option("-r", "--run", action="store", type="string", dest="run", default="G")
parser.add_option("-e", "--efficiency", action="store_true", dest="efficiency", default=False)
parser.add_option("-s", "--sample", action="store", type="string", dest="sample", default="All")
parser.add_option("-g", "--goodplots", action="store_true", default=False, dest="goodplots")#not needed in 2016
parser.add_option("-a", "--all", action="store_true", default=False, dest="all")
parser.add_option("-b", "--bash", action="store_true", default=True, dest="bash")
parser.add_option("-B", "--blind", action="store_true", default=False, dest="blind")
parser.add_option("-f", "--final", action="store_true", default=False, dest="final")
parser.add_option("-R", "--rebin", action="store_true", default=False, dest="rebin")
parser.add_option("-p", "--public", action="store_true", default=False, dest="public")
(options, args) = parser.parse_args()
if options.bash: gROOT.SetBatch(True)

########## SETTINGS ##########

gStyle.SetOptStat(0)
#gStyle.SetPadTopMargin(-0.05)
#gStyle.SetPadBottomMargin(-0.05)
#gStyle.SetPadRightMargin(-0.2)
#gStyle.SetPadLeftMargin(-0.2)

ERA                = "2017"
REGION             = "SR"#"WtoMN"#"WtoEN"#"WtoMN"#"ZtoEE"#"JetHT"
CUT                = "isSR"#"isWtoMN"#"isWtoEN"#"isWtoMN"#"isZtoEE"#"isJetHT"#"isJetMET_low_dPhi_MET_200_Lep"#"isJetHT_unprescaled"#"isWtoEN"# && MT<100"#"isZtoMM"
KILL_QCD           = True
DO_ETA             = True
DO_PHI             = False#False#
if DO_PHI:
    DO_ETA = False
CUT_ETA            = False#False#True#True#False
CUT_PHI            = False

print "\n"
print "region: ", REGION
print "kill qcd: ", KILL_QCD
print "do eta: ", DO_ETA
print "do phi: ", DO_PHI
print "eta cut: ", CUT_ETA
print "phi cut: ", CUT_PHI
print "\n"



NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_August_2021/v5_calo_AOD_"+ERA+"_"+REGION+"_signal_uncertainties/"#"_2017_signal/"#
NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_August_2021/v5_calo_AOD_2017_SR_JER/"
NTUPLEDIR_Up          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_August_2021/v5_calo_AOD_2017_SR_JER_Up/"
NTUPLEDIR_Down          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_August_2021/v5_calo_AOD_2017_SR_JER_Down/"
#if REGION=="WtoEN":
#    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoEN_noMT/"
#if REGION=="WtoMN":
#    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoMN_noMT/"


PLOTDIR            = "plots/Efficiency/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
#OUTPUTDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"/"
OUTPUTDIR          = "plots/Uncertainties/v5_calo_AOD_"+ERA+"_"+REGION+"/"
OUTPUTDIR          = "plots/Uncertainties/v5_calo_AOD_"+ERA+"_"+REGION+"_JER/"
CHAN               = "SUSY"
particle           = "#chi"
ctaupoint          = 1000
signalMultFactor   = 0.001
PRELIMINARY        = True
TAGVAR             = "nTagJets_0p996_JJ"

REBIN              = options.rebin
SAVE               = True
#LUMI               = (2.090881769 + 3.121721023 + 2.909790309 + 0.385165352)*1000#8507.558453#59.74*1000#Run2018

sign = [
    "SUSY_mh400_pl1000","SUSY_mh300_pl1000","SUSY_mh250_pl1000","SUSY_mh200_pl1000","SUSY_mh175_pl1000","SUSY_mh150_pl1000","SUSY_mh127_pl1000",
]
data = ["HighMET"]

if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]
elif ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]
elif ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    if "_G-H" in jet_tag:
        print "Only GH!"
        LUMI  = lumi[ data[0] ]["G"]+lumi[ data[0] ]["H"]#["tot"]
    elif "_B-F" in jet_tag:
        LUMI  = lumi[ data[0] ]["B"]+lumi[ data[0] ]["C"]+lumi[ data[0] ]["D"]+lumi[ data[0] ]["E"]+lumi[ data[0] ]["F"]#["tot"]
    else:
        LUMI  = lumi[ data[0] ]["tot"]

print LUMI

COMPARE = options.compare
DRAWSIGNAL = options.drawsignal

########## SAMPLES ##########

colors = [856, 1,  634, 420, 806, 882, 401, 418, 881, 798, 602, 921]
colors_jj = [1,2,4,418,801,856]
colors = colors_jj + [881, 798, 602, 921]
lines = [1,1,1,1,1,2,2,2,2]
markers = [20,20,20,20,20,24,24,24,24]
siz = 1.3
marker_sizes = [siz,siz,siz,siz,siz,siz,siz,siz,siz]
#markers = [20,21]#[24,25,21,20]
########## ######## ##########

#gROOT.SetBatch(True)

#cut_den = CUT
#cut_num = cut_den + " && "
chain = {}
filename = {}
hist_den = {}
hist_num = {}
hist_num_cutbased = {}
eff = {} 
eff_cutbased = {} 
dnn_bins = array('d', [0.,.00001,.0001,0.001,.01,.05,.1,.25,.5,.75,1.,])
#less_bins = array('d', [1,10,20,30,40,50,60,70,80,90,100,10000])
#less_bins_plot = array('d', [1,10,20,30,40,50,60,70,80,90,100,1000])
#New version from Jiajing
less_bins = array('d', [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,10000])
less_bins_pt = array('d', [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,10000])
less_bins_plot = array('d', [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,10000])

print "Warning, overflow causing nans, remove last bin"
np_bins = np.array(less_bins)
np_bins = np_bins[np_bins>=30]#only interested in non-zero bins
np_bins = np_bins[np_bins<10000]#only interested in non-nan bins

#bins=array('d', [1,10,20,30,40,50,60,70,80,90,100,1000])
#bins=np.array([1,10,20,30,40,50,60,70,80,90,100,1000])
#bins = bins.astype(np.float32)

bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,150,200,300,500,10000])
more_bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,350,400,450,500,550,600,650,700,800,900,1000,2500,5000,10000,100000])
np_more_bins = np.array(more_bins)
#np_more_bins = np_more_bins[np_more_bins>=30]#only interested in non-zero bins

#more_bins_eta = array('d', [-1.5,-1.4,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])
more_bins_eta = array('d',[-1.5,-1.45,-1.4,-1.35,-1.3,-1.25,-1.2,-1.15,-1.1,-1.05,-1.,-0.95,-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5 ])
#less_bins_eta = array('d',[-1.5, -1.25, -1., -0.5, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.85, 1., 1.25, 1.5])
#homogeneous:
less_bins_eta = array('d',[-1.5, -1.25, -1., -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5])
#more_bins=np.array([1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,350,400,450,500,550,600,650,700,800,900,1000])
#more_bins = more_bins.astype(np.float32)
###more_bins = bins

more_bins_phi = array('d',[-3.2 , -3.04, -2.88, -2.72, -2.56, -2.4 , -2.24, -2.08, -1.92,
                           -1.76, -1.6 , -1.44, -1.28, -1.12, -0.96, -0.8 , -0.64, -0.48,
                           -0.32, -0.16,  0.  ,  0.16,  0.32,  0.48,  0.64,  0.8 ,  0.96,
                           1.12,  1.28,  1.44,  1.6 ,  1.76,  1.92,  2.08,  2.24,  2.4 ,
                           2.56,  2.72,  2.88,  3.04,  3.2                       ])
less_bins_phi = array('d',[
    -3.2 , -2.56, -1.92, -1.28, -0.64,  0.  ,  0.64,  1.28,  1.92,
    2.56,  3.2
])

np_bins_eta = np.array(less_bins_eta)
np_bins_eta = np_bins_eta[0:-1]

maxeff = 0.002#15

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def deltaPhi( p1, p2):
    '''Computes delta phi, handling periodic limit conditions.'''
    res = abs(p1 - p2)
    if res > math.pi:
        res -= 2*math.pi
    return res

def deltaR( e1, p1, e2, p2):
    de = e1 - e2
    dp = deltaPhi(p1, p2)
    return math.sqrt(de*de + dp*dp)


def get_tree_weights(sample_list):
    
    tree_w_dict = defaultdict(dict)
    for i, s in enumerate(sample_list):
        for l, ss in enumerate(samples[s]['files']):
            #Tree weight
            if ('Run201') in ss:
                t_w = 1.
            else:
                filename = TFile(NTUPLEDIR+ss+'.root', "READ")
                nevents = filename.Get("c_nEvents").GetBinContent(1)
                b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
                n_pass      = filename.Get("n_pass").GetBinContent(1)
                n_odd       = filename.Get("n_odd").GetBinContent(1)
                filename.Close()
                if('GluGluH2_H2ToSSTobbbb') in ss:
                    xs = 1.
                elif('XXTo4J') in ss:
                    xs = 1.*0.001
                elif('gluinoGMSB') in ss:
                    xs = 1.*0.001
                elif('GluinoGluinoToNeutralinoNeutralinoTo2T2B2S') in ss:
                    xs = 1.*0.001
                elif ('n3n2-n1-hbb-hbb') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                elif ('TChiHH') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                else:
                    xs = sample[ss]['xsec'] * sample[ss]['kfactor']
                print "LUMI ", LUMI
                print "xs ", xs
                print "nevents ", nevents
                t_w = LUMI * xs / nevents
                if(b_skipTrain>0):
                    #print("Skip even events: ")
                    #print "n_pass: ", n_pass
                    #print "n_odd: ", n_odd
                    #if(n_pass>0):
                    #    print "ratio: ", float(n_odd/n_pass)
                    if(n_odd>0):
                        #print "even/odd weight: ", float(n_pass/n_odd)
                        t_w *= float(n_pass/n_odd)
            print("%s has tree weight %f")%(ss,t_w)
            tree_w_dict[s][ss] = t_w

    return tree_w_dict
    

def calc_unc(tree_weight_dict,sign,comb_fold_label="",add_label="",label_2="",pred_unc=0,eta=False,phi=False,eta_cut=False,phi_cut=False,scale_mu=1.,unc_set=[]):

    y_2 = defaultdict(dict)
    unc_2 = defaultdict(dict)
    #y_2_up = defaultdict(dict)
    #y_2_down = defaultdict(dict)

    if not os.path.isdir(OUTPUTDIR): os.mkdir(OUTPUTDIR)
    UNCDIR = OUTPUTDIR+CHAN+"/"
    if not os.path.isdir(UNCDIR): os.mkdir(UNCDIR)

    UNCDIR+=TAGVAR
    if eta_cut:
        UNCDIR += "_eta_1p0"
    if phi_cut:
        UNCDIR += "_phi_cut"

    if eta:
        UNCDIR += "_vs_eta"+comb_fold_label
    else:
        if phi:
            UNCDIR += "_vs_phi"+comb_fold_label
        else:
            UNCDIR += comb_fold_label

    UNCDIR += "/"
    if not os.path.isdir(UNCDIR):
        os.mkdir(UNCDIR)

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
        print '-'*11*2

    mass = []
    print "tree_weight_dict"
    print tree_weight_dict
    chainSignal = {}
    list_of_variables = ["nTagJets*","isMC","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight*","TriggerWeight*",CUT]#"nLeptons"
    if ERA=="2018":
        list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]

    for i,s in enumerate(sign):
        print "m chi: ",samples[s]['mass']
        print samples[s]['ctau']
        mass.append(samples[s]['mass'])
        chainSignal[s] = TChain("tree")
        tree_weights = {}
        chain_entries_cumulative = {}
        chain_entries = {}
        array_size_tot = 0

        for l, ss in enumerate(samples[s]['files']):
            print "ss", ss
            chainSignal[s].Add(NTUPLEDIR + ss + '.root')
            tree_weights[l] = tree_weight_dict[s][ss]
            chain_entries_cumulative[l] = chainSignal[s].GetEntries()
            if l==0:
                chain_entries[l]=chainSignal[s].GetEntries()
            else:
                chain_entries[l]=chain_entries_cumulative[l] - chain_entries_cumulative[l-1]
                print "Entries per sample ", ss, " : ", chain_entries[l]

            chunk_size = 100000
            n_iter = int(float(chain_entries[l])/float(chunk_size))
            c = 0
            if chain_entries[l]%chunk_size!=0:
                n_iter += 1

            print "chain_entries: ", chain_entries[l]
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter

            #Looping three times is inefficient but it allows to not modify the code
            for unc in unc_set:            
                bin2 = np.array([])
                #print "Doing this uncertainty: ", unc
                gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
                for arrays in gen:
                    st_it = time.time()
                    key_list = arrays.keys()
                    array_size_tot+=len(arrays[ key_list[0] ])
                    tree_w_array = tree_weight_dict[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                    cut_mask = (arrays[CUT]>0)

                    if eta_cut and phi_cut==False:
                        cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                        cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                        cut_mask = (cut_mask_eta.any()==True)

                    if phi_cut and eta_cut==False:
                        cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                        cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                        cut_mask = (cut_mask_phi.any()==True)

                    if phi_cut and eta_cut:
                        cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                        cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                        cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                        cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                        #This is needed to guarantee the shapes are consistent
                        cut_mask = (cut_mask_phi_eta.any()==True)

                    ### Cuts below depend on uncertainty
                    ### need to define a dictionary of yields per sample
                
                    if KILL_QCD:
                        ##Here effect of cuts on unc!
                        minjetmet_var = "MinJetMetDPhi"+unc
                        print "minjetmet_var: ", minjetmet_var
                        cut_mask = np.logical_and( cut_mask, arrays[minjetmet_var]>0.5)

                    #SR cut
                    tag_var = "nTagJets_0p996"+unc
                    tr_weight = "TriggerWeight"+unc
                    if unc=="":
                        tag_var = "nTagJets_0p996_JJ"
                        tr_weight = "TriggerWeight"
                    print "tag_var: ", tag_var
                    print "tr_weight: ", tr_weight
                    cut_mask = np.logical_and( cut_mask, arrays[tag_var]>1 )
                    tag = arrays[tag_var][cut_mask] !=0
                    eventweight = arrays["EventWeight"][cut_mask]
                    pureweight = arrays["PUReWeight"][cut_mask]
                    triggerweight = arrays[tr_weight][cut_mask]
                    del arrays
                    if scale_mu!=1.:
                        print "Scaling mu up by factor", scale_mu
                        weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[s][ss]*signalMultFactor*scale_mu
                    else:
                        #w/o trigger:
                        #weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]*signalMultFactor
                        weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[s][ss]*signalMultFactor
                    bin2 = np.concatenate( (bin2,np.multiply(tag,weight)) )

                del gen
                #Here: inside the loop of signal samples ss, and inside the loop of syst uncertainties
                if unc=="":
                    lab = "nominal"
                else:
                    lab=unc.replace("_","")
                y_2[ samples[s]['mass'] ][lab] = np.sum(bin2)#*tree_weight --> already in w
                #e_2 = np.sqrt( sum(x*x for x in bin2) ).sum()#*tree_weight --> already in w
                
        for unc in unc_set:
            if unc!="":
                lab = unc.replace("_","")
                unc_2[ samples[s]['mass'] ][lab] = 100.*abs(y_2[ samples[s]['mass'] ][lab] - y_2[ samples[s]['mass'] ]["nominal"])/(y_2[ samples[s]['mass'] ]["nominal"])
                print s, lab, unc_2[ samples[s]['mass'] ][lab]
            else:
                print s, "nominal", y_2[ samples[s]['mass'] ]["nominal"]

        #print y_2
        #print unc_2
        #here compute y_2 differences...
        #outname = "some_name"
        #print "Info: " , outname, " written"

    mass = np.array(mass)
    mass = np.sort(mass)

    graph_yield = {}#TGraph()
    graph_unc = {}#TGraph()
    n = 0
    for m in mass:
        #print m, y_2[m], unc_2[m]
        for unc in unc_set:
            lab = ""
            if unc!="":
                lab = unc.replace("_","")
                graph_unc[lab] = TGraph()
            else:
                lab = "nominal"
            graph_yield[lab] = TGraph()


    for m in mass:
        #print m, y_2[m], unc_2[m]
        for unc in unc_set:
            lab = ""
            if unc!="":
                lab = unc.replace("_","")
                graph_unc[lab].SetPoint(n,float(m),unc_2[m][lab])
            else:
                lab = "nominal"
            graph_yield[lab].SetPoint(n,float(m),y_2[m][lab])
            print lab
            print y_2[m].keys()
        n+=1

    
    #now plot
    colors = [801,856,825,881,2,602,880,798,856,5]
    lines = [1,2,2,3,3,4,4,5,5]
    gStyle.SetLegendFillColor(0)

    c1 = TCanvas("c1", "c1", 800, 600)
    c1.cd()
    c1.SetGrid()
    c1.GetPad(0).SetTopMargin(0.06)
    c1.GetPad(0).SetRightMargin(0.05)
    c1.GetPad(0).SetTicks(1, 1)
    c1.GetPad(0).SetGridx()
    c1.GetPad(0).SetGridy()
    c1.GetPad(0).SetLogy()
    c1.GetPad(0).SetLogx()
    top = 0.9
    nitems = len(unc_set)
    leg1 = TLegend(0.5,0.2+0.5,0.9,0.4+0.5)
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetTextSize(0.04)

    l = 0
    for unc in unc_set:
        lab = ""
        if unc!="":
            lab = unc.replace("_","")
            #do unc
        else:
            lab = "nominal"
        graph_yield[lab].SetLineStyle(lines[l])
        graph_yield[lab].SetLineWidth(3)
        graph_yield[lab].SetLineColor(colors[l])
        graph_yield[lab].SetMarkerColor(colors[l])
        graph_yield[lab].SetFillColorAlpha(colors[l],0.5)
        graph_yield[lab].SetMinimum(0.05)
        graph_yield[lab].SetMaximum(2.5)
        graph_yield[lab].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
        graph_yield[lab].GetYaxis().SetTitle("Event yield in bin2")
        graph_yield[lab].GetXaxis().SetNoExponent(True)
        graph_yield[lab].GetXaxis().SetMoreLogLabels(True)
        graph_yield[lab].GetXaxis().SetTitleSize(0.048)
        graph_yield[lab].GetYaxis().SetTitleSize(0.048)
        graph_yield[lab].GetYaxis().SetTitleOffset(0.8)
        graph_yield[lab].GetXaxis().SetTitleOffset(0.9)
        if l == 0:
            graph_yield[lab].SetMarkerStyle(20)
            graph_yield[lab].Draw("APL3")
        else:
            graph_yield[lab].SetMarkerStyle(20)
            graph_yield[lab].Draw("SAME,PL3")
        leg1.AddEntry(graph_yield[lab],lab,"L")
        l+=1
    leg1.Draw()

    if PRELIMINARY:
        drawCMS(samples, LUMI, "Preliminary",onTop=True,left_marg_CMS=0.1)
    else:
        drawCMS(samples, LUMI, "",left_marg_CMS=0.32)

    drawRegion(CHAN,top=0.7)
    drawAnalysis("LL"+CHAN)
    drawTagVar(TAGVAR)

    OUTSTRING = UNCDIR+"/Event_yield_ctau"+str(ctaupoint)+"_"+CHAN
    c1.Print(OUTSTRING+".png")
    c1.Print(OUTSTRING+".pdf")
    c1.Close()



    c2 = TCanvas("c2", "c2", 800, 600)
    c2.cd()
    c2.SetGrid()
    c2.GetPad(0).SetTopMargin(0.06)
    c2.GetPad(0).SetRightMargin(0.05)
    c2.GetPad(0).SetTicks(1, 1)
    c2.GetPad(0).SetGridx()
    c2.GetPad(0).SetGridy()
    #c2.GetPad(0).SetLogy()
    c2.GetPad(0).SetLogx()
    leg2 = TLegend(0.5,0.2+0.5,0.9,0.4+0.5)
    leg2.SetBorderSize(0)
    leg2.SetFillStyle(0)
    leg2.SetTextSize(0.04)

    l = 0

    colors = [856,825,881,2,602,880,798,856,5]
    lines = [2,2,3,3,4,4,5,5]

    for unc in unc_set:
        lab = ""
        if unc!="":
            lab = unc.replace("_","")
            graph_unc[lab].SetLineStyle(lines[l])
            graph_unc[lab].SetLineWidth(3)
            graph_unc[lab].SetLineColor(colors[l])
            graph_unc[lab].SetMarkerColor(colors[l])
            graph_unc[lab].SetFillColorAlpha(colors[l],0.5)
            graph_unc[lab].SetMinimum(-0.5)
            graph_unc[lab].SetMaximum(10.)
            graph_unc[lab].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
            graph_unc[lab].GetYaxis().SetTitle("Uncertainty on yield in bin2 (%)")
            graph_unc[lab].GetXaxis().SetNoExponent(True)
            graph_unc[lab].GetXaxis().SetMoreLogLabels(True)
            graph_unc[lab].GetXaxis().SetTitleSize(0.048)
            graph_unc[lab].GetYaxis().SetTitleSize(0.048)
            graph_unc[lab].GetYaxis().SetTitleOffset(0.8)
            graph_unc[lab].GetXaxis().SetTitleOffset(0.9)
            if l == 0:
                graph_unc[lab].SetMarkerStyle(20)
                graph_unc[lab].Draw("APL3")
            else:
                graph_unc[lab].SetMarkerStyle(20)
                graph_unc[lab].Draw("SAME,PL3")
            leg2.AddEntry(graph_unc[lab],lab,"L")
            l+=1
    leg2.Draw()

    if PRELIMINARY:
        drawCMS(samples, LUMI, "Preliminary",onTop=True,left_marg_CMS=0.1)
    else:
        drawCMS(samples, LUMI, "",left_marg_CMS=0.32)

    drawRegion(CHAN,top=0.7)
    drawAnalysis("LL"+CHAN)
    drawTagVar(TAGVAR)

    OUTSTRING = UNCDIR+"/Uncertainty_yield_ctau"+str(ctaupoint)+"_"+CHAN
    c2.Print(OUTSTRING+".png")
    c2.Print(OUTSTRING+".pdf")
    c2.Close()


def calc_unc_JER(tree_weight_dict,sign,comb_fold_label="",add_label="",label_2="",pred_unc=0,eta=False,phi=False,eta_cut=False,phi_cut=False,scale_mu=1.,unc_set=[]):

    y_2 = defaultdict(dict)
    unc_2 = defaultdict(dict)

    if not os.path.isdir(OUTPUTDIR): os.mkdir(OUTPUTDIR)
    UNCDIR = OUTPUTDIR+CHAN+"/"
    if not os.path.isdir(UNCDIR): os.mkdir(UNCDIR)

    UNCDIR+=TAGVAR
    if eta_cut:
        UNCDIR += "_eta_1p0"
    if phi_cut:
        UNCDIR += "_phi_cut"

    if eta:
        UNCDIR += "_vs_eta"+comb_fold_label
    else:
        if phi:
            UNCDIR += "_vs_phi"+comb_fold_label
        else:
            UNCDIR += comb_fold_label

    UNCDIR += "/"
    if not os.path.isdir(UNCDIR):
        os.mkdir(UNCDIR)

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
        print '-'*11*2

    mass = []
    print "tree_weight_dict"
    print tree_weight_dict
    chainSignal = {}
    chainSignal_Up = {}
    chainSignal_Down = {}
    list_of_variables = ["nTagJets*","isMC","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight*","TriggerWeight*",CUT]#"nLeptons"
    if ERA=="2018":
        list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]

    for i,s in enumerate(sign):
        print "m chi: ",samples[s]['mass']
        print samples[s]['ctau']
        mass.append(samples[s]['mass'])
        chainSignal[s] = TChain("tree")
        chainSignal_Up[s] = TChain("tree")
        chainSignal_Down[s] = TChain("tree")
        tree_weights = {}
        chain_entries_cumulative = {}
        chain_entries = {}
        array_size_tot = 0

        for l, ss in enumerate(samples[s]['files']):
            print "ss", ss
            chainSignal[s].Add(NTUPLEDIR + ss + '.root')
            chainSignal_Up[s].Add(NTUPLEDIR + ss + '.root')
            chainSignal_Down[s].Add(NTUPLEDIR + ss + '.root')
            tree_weights[l] = tree_weight_dict[s][ss]
            chain_entries_cumulative[l] = chainSignal[s].GetEntries()
            if l==0:
                chain_entries[l]=chainSignal[s].GetEntries()
            else:
                chain_entries[l]=chain_entries_cumulative[l] - chain_entries_cumulative[l-1]
                print "Entries per sample ", ss, " : ", chain_entries[l]

            chunk_size = 100000
            n_iter = int(float(chain_entries[l])/float(chunk_size))
            c = 0
            if chain_entries[l]%chunk_size!=0:
                n_iter += 1

            print "chain_entries: ", chain_entries[l]
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter

            #Looping smarter
            bin2 = np.array([])
            bin2_Up = np.array([])
            bin2_Down = np.array([])

            for unc in unc_set:
                if unc=="":
                    print "\n"
                    print "Doing nominal JER: ", NTUPLEDIR
                    gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
                    for arrays in gen:
                        st_it = time.time()
                        key_list = arrays.keys()
                        array_size_tot+=len(arrays[ key_list[0] ])
                        tree_w_array = tree_weight_dict[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                        cut_mask = (arrays[CUT]>0)

                        if eta_cut and phi_cut==False:
                            cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                            cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                            cut_mask = (cut_mask_eta.any()==True)

                        if phi_cut and eta_cut==False:
                            cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                            cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                            cut_mask = (cut_mask_phi.any()==True)

                        if phi_cut and eta_cut:
                            cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                            cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                            cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                            cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                            #This is needed to guarantee the shapes are consistent
                            cut_mask = (cut_mask_phi_eta.any()==True)

                        ### Cuts below depend on uncertainty
                        ### need to define a dictionary of yields per sample
                
                        if KILL_QCD:
                            ##Here effect of cuts on unc!
                            minjetmet_var = "MinJetMetDPhi"+unc
                            print "minjetmet_var: ", minjetmet_var
                            cut_mask = np.logical_and( cut_mask, arrays[minjetmet_var]>0.5)

                        #SR cut
                        tag_var = "nTagJets_0p996"+unc
                        tr_weight = "TriggerWeight"+unc
                        if unc=="":
                            tag_var = "nTagJets_0p996_JJ"
                            tr_weight = "TriggerWeight"
                        print "tag_var: ", tag_var
                        print "tr_weight: ", tr_weight
                        cut_mask = np.logical_and( cut_mask, arrays[tag_var]>1 )
                        tag = arrays[tag_var][cut_mask] !=0
                        eventweight = arrays["EventWeight"][cut_mask]
                        pureweight = arrays["PUReWeight"][cut_mask]
                        triggerweight = arrays[tr_weight][cut_mask]
                        del arrays
                        if scale_mu!=1.:
                            print "Scaling mu up by factor", scale_mu
                            weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[s][ss]*signalMultFactor*scale_mu
                        else:
                            #w/o trigger:
                            #weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]*signalMultFactor
                            weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[s][ss]*signalMultFactor
                        bin2 = np.concatenate( (bin2,np.multiply(tag,weight)) )

                    del gen
                    #Here: inside the loop of signal samples ss, and inside the loop of syst uncertainties
                    if unc=="":
                        lab = "nominal"
                    else:
                        lab=unc.replace("_","")
                    y_2[ samples[s]['mass'] ][lab] = np.sum(bin2)#*tree_weight --> already in w
                    #e_2 = np.sqrt( sum(x*x for x in bin2) ).sum()#*tree_weight --> already in w

                elif unc=="_JERUp":
                    print "\n"
                    print "Doing JER Up: ", NTUPLEDIR_Up
                    gen = uproot.iterate([NTUPLEDIR_Up + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
                    for arrays in gen:
                        st_it = time.time()
                        key_list = arrays.keys()
                        array_size_tot+=len(arrays[ key_list[0] ])
                        tree_w_array = tree_weight_dict[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                        cut_mask = (arrays[CUT]>0)

                        if eta_cut and phi_cut==False:
                            cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                            cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                            cut_mask = (cut_mask_eta.any()==True)

                        if phi_cut and eta_cut==False:
                            cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                            cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                            cut_mask = (cut_mask_phi.any()==True)

                        if phi_cut and eta_cut:
                            cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                            cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                            cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                            cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                            #This is needed to guarantee the shapes are consistent
                            cut_mask = (cut_mask_phi_eta.any()==True)

                        ### Cuts below depend on uncertainty
                        ### need to define a dictionary of yields per sample
                
                        if KILL_QCD:
                            ##Here effect of cuts on unc!
                            minjetmet_var = "MinJetMetDPhi"+unc
                            print "minjetmet_var: ", minjetmet_var
                            cut_mask = np.logical_and( cut_mask, arrays[minjetmet_var]>0.5)

                        #SR cut
                        tag_var = "nTagJets_0p996"+unc
                        tr_weight = "TriggerWeight"+unc
                        if unc=="":
                            tag_var = "nTagJets_0p996_JJ"
                            tr_weight = "TriggerWeight"
                        print "tag_var: ", tag_var
                        print "tr_weight: ", tr_weight
                        cut_mask = np.logical_and( cut_mask, arrays[tag_var]>1 )
                        tag = arrays[tag_var][cut_mask] !=0
                        eventweight = arrays["EventWeight"][cut_mask]
                        pureweight = arrays["PUReWeight"][cut_mask]
                        triggerweight = arrays[tr_weight][cut_mask]
                        del arrays
                        if scale_mu!=1.:
                            print "Scaling mu up by factor", scale_mu
                            weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[s][ss]*signalMultFactor*scale_mu
                        else:
                            #w/o trigger:
                            #weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]*signalMultFactor
                            weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[s][ss]*signalMultFactor
                        bin2_Up = np.concatenate( (bin2_Up,np.multiply(tag,weight)) )

                    del gen
                    #Here: inside the loop of signal samples ss, and inside the loop of syst uncertainties
                    if unc=="":
                        lab = "nominal"
                    else:
                        lab=unc.replace("_","")
                    y_2[ samples[s]['mass'] ][lab] = np.sum(bin2_Up)#*tree_weight --> already in w
                    #print "Written: ", y_2

                elif unc=="_JERDown":
                    print "\n"
                    print "Doing JER Down: ", NTUPLEDIR_Down
                    gen = uproot.iterate([NTUPLEDIR_Down + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
                    for arrays in gen:
                        st_it = time.time()
                        key_list = arrays.keys()
                        array_size_tot+=len(arrays[ key_list[0] ])
                        tree_w_array = tree_weight_dict[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                        cut_mask = (arrays[CUT]>0)

                        if eta_cut and phi_cut==False:
                            cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                            cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                            cut_mask = (cut_mask_eta.any()==True)

                        if phi_cut and eta_cut==False:
                            cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                            cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                            cut_mask = (cut_mask_phi.any()==True)

                        if phi_cut and eta_cut:
                            cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                            cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                            cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                            cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                            #This is needed to guarantee the shapes are consistent
                            cut_mask = (cut_mask_phi_eta.any()==True)

                        ### Cuts below depend on uncertainty
                        ### need to define a dictionary of yields per sample
                
                        if KILL_QCD:
                            ##Here effect of cuts on unc!
                            minjetmet_var = "MinJetMetDPhi"+unc
                            print "minjetmet_var: ", minjetmet_var
                            cut_mask = np.logical_and( cut_mask, arrays[minjetmet_var]>0.5)

                        #SR cut
                        tag_var = "nTagJets_0p996"+unc
                        tr_weight = "TriggerWeight"+unc
                        if unc=="":
                            tag_var = "nTagJets_0p996_JJ"
                            tr_weight = "TriggerWeight"
                        print "tag_var: ", tag_var
                        print "tr_weight: ", tr_weight
                        cut_mask = np.logical_and( cut_mask, arrays[tag_var]>1 )
                        tag = arrays[tag_var][cut_mask] !=0
                        eventweight = arrays["EventWeight"][cut_mask]
                        pureweight = arrays["PUReWeight"][cut_mask]
                        triggerweight = arrays[tr_weight][cut_mask]
                        del arrays
                        if scale_mu!=1.:
                            print "Scaling mu up by factor", scale_mu
                            weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[s][ss]*signalMultFactor*scale_mu
                        else:
                            #w/o trigger:
                            #weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]*signalMultFactor
                            weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[s][ss]*signalMultFactor
                        bin2_Down = np.concatenate( (bin2_Down,np.multiply(tag,weight)) )

                    del gen
                    #Here: inside the loop of signal samples ss, and inside the loop of syst uncertainties
                    if unc=="":
                        lab = "nominal"
                    else:
                        lab=unc.replace("_","")
                    y_2[ samples[s]['mass'] ][lab] = np.sum(bin2_Down)#*tree_weight --> already in w
                    #print "Written: ", y_2
                else:
                    print "\n"
                    print "Wrong JER, aborting... "
                    exit()

        #This part can be kept, looping inside samples
        for unc in unc_set:
            if unc!="":
                lab = unc.replace("_","")
                unc_2[ samples[s]['mass'] ][lab] = 100.*abs(y_2[ samples[s]['mass'] ][lab] - y_2[ samples[s]['mass'] ]["nominal"])/(y_2[ samples[s]['mass'] ]["nominal"])
                print s, lab, unc_2[ samples[s]['mass'] ][lab]
            else:
                print s, "nominal", y_2[ samples[s]['mass'] ]["nominal"]

        #print y_2
        #print unc_2
        #here compute y_2 differences...
        #outname = "some_name"
        #print "Info: " , outname, " written"

    mass = np.array(mass)
    mass = np.sort(mass)

    graph_yield = {}#TGraph()
    graph_unc = {}#TGraph()
    n = 0
    for m in mass:
        #print m, y_2[m], unc_2[m]
        for unc in unc_set:
            lab = ""
            if unc!="":
                lab = unc.replace("_","")
                graph_unc[lab] = TGraph()
            else:
                lab = "nominal"
            graph_yield[lab] = TGraph()


    for m in mass:
        #print m, y_2[m], unc_2[m]
        for unc in unc_set:
            lab = ""
            if unc!="":
                lab = unc.replace("_","")
                graph_unc[lab].SetPoint(n,float(m),unc_2[m][lab])
            else:
                lab = "nominal"
            graph_yield[lab].SetPoint(n,float(m),y_2[m][lab])
            print lab
            print y_2[m].keys()
        n+=1

    
    #now plot
    colors = [801,856,825,881,2,602,880,798,856,5]
    lines = [1,2,2,3,3,4,4,5,5]
    gStyle.SetLegendFillColor(0)

    c1 = TCanvas("c1", "c1", 800, 600)
    c1.cd()
    c1.SetGrid()
    c1.GetPad(0).SetTopMargin(0.06)
    c1.GetPad(0).SetRightMargin(0.05)
    c1.GetPad(0).SetTicks(1, 1)
    c1.GetPad(0).SetGridx()
    c1.GetPad(0).SetGridy()
    c1.GetPad(0).SetLogy()
    c1.GetPad(0).SetLogx()
    top = 0.9
    nitems = len(unc_set)
    leg1 = TLegend(0.5,0.2+0.5,0.9,0.4+0.5)
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetTextSize(0.04)

    l = 0
    for unc in unc_set:
        lab = ""
        if unc!="":
            lab = unc.replace("_","")
            #do unc
        else:
            lab = "nominal"
        graph_yield[lab].SetLineStyle(lines[l])
        graph_yield[lab].SetLineWidth(3)
        graph_yield[lab].SetLineColor(colors[l])
        graph_yield[lab].SetMarkerColor(colors[l])
        graph_yield[lab].SetFillColorAlpha(colors[l],0.5)
        graph_yield[lab].SetMinimum(0.05)
        graph_yield[lab].SetMaximum(2.5)
        graph_yield[lab].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
        graph_yield[lab].GetYaxis().SetTitle("Event yield in bin2")
        graph_yield[lab].GetXaxis().SetNoExponent(True)
        graph_yield[lab].GetXaxis().SetMoreLogLabels(True)
        graph_yield[lab].GetXaxis().SetTitleSize(0.048)
        graph_yield[lab].GetYaxis().SetTitleSize(0.048)
        graph_yield[lab].GetYaxis().SetTitleOffset(0.8)
        graph_yield[lab].GetXaxis().SetTitleOffset(0.9)
        if l == 0:
            graph_yield[lab].SetMarkerStyle(20)
            graph_yield[lab].Draw("APL3")
        else:
            graph_yield[lab].SetMarkerStyle(20)
            graph_yield[lab].Draw("SAME,PL3")
        leg1.AddEntry(graph_yield[lab],lab,"L")
        l+=1
    leg1.Draw()

    if PRELIMINARY:
        drawCMS(samples, LUMI, "Preliminary",onTop=True,left_marg_CMS=0.1)
    else:
        drawCMS(samples, LUMI, "",left_marg_CMS=0.32)

    drawRegion(CHAN,top=0.7)
    drawAnalysis("LL"+CHAN)
    drawTagVar(TAGVAR)

    OUTSTRING = UNCDIR+"/Event_yield_ctau"+str(ctaupoint)+"_"+CHAN
    c1.Print(OUTSTRING+"_JER_Up_Down.png")
    c1.Print(OUTSTRING+"_JER_Up_Down.pdf")
    c1.Close()



    c2 = TCanvas("c2", "c2", 800, 600)
    c2.cd()
    c2.SetGrid()
    c2.GetPad(0).SetTopMargin(0.06)
    c2.GetPad(0).SetRightMargin(0.05)
    c2.GetPad(0).SetTicks(1, 1)
    c2.GetPad(0).SetGridx()
    c2.GetPad(0).SetGridy()
    #c2.GetPad(0).SetLogy()
    c2.GetPad(0).SetLogx()
    leg2 = TLegend(0.5,0.2+0.5,0.9,0.4+0.5)
    leg2.SetBorderSize(0)
    leg2.SetFillStyle(0)
    leg2.SetTextSize(0.04)

    l = 0

    colors = [602,880]#[856,825,881,2,602,880,798,856,5]
    lines = [2,2,3,3,4,4,5,5]

    for unc in unc_set:
        lab = ""
        if unc!="":
            lab = unc.replace("_","")
            graph_unc[lab].SetLineStyle(lines[l])
            graph_unc[lab].SetLineWidth(3)
            graph_unc[lab].SetLineColor(colors[l])
            graph_unc[lab].SetMarkerColor(colors[l])
            graph_unc[lab].SetFillColorAlpha(colors[l],0.5)
            graph_unc[lab].SetMinimum(-0.5)
            graph_unc[lab].SetMaximum(10.)
            graph_unc[lab].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
            graph_unc[lab].GetYaxis().SetTitle("Uncertainty on yield in bin2 (%)")
            graph_unc[lab].GetXaxis().SetNoExponent(True)
            graph_unc[lab].GetXaxis().SetMoreLogLabels(True)
            graph_unc[lab].GetXaxis().SetTitleSize(0.048)
            graph_unc[lab].GetYaxis().SetTitleSize(0.048)
            graph_unc[lab].GetYaxis().SetTitleOffset(0.8)
            graph_unc[lab].GetXaxis().SetTitleOffset(0.9)
            if l == 0:
                graph_unc[lab].SetMarkerStyle(20)
                graph_unc[lab].Draw("APL3")
            else:
                graph_unc[lab].SetMarkerStyle(20)
                graph_unc[lab].Draw("SAME,PL3")
            leg2.AddEntry(graph_unc[lab],lab,"L")
            l+=1
    leg2.Draw()

    if PRELIMINARY:
        drawCMS(samples, LUMI, "Preliminary",onTop=True,left_marg_CMS=0.1)
    else:
        drawCMS(samples, LUMI, "",left_marg_CMS=0.32)

    drawRegion(CHAN,top=0.7)
    drawAnalysis("LL"+CHAN)
    drawTagVar(TAGVAR)

    OUTSTRING = UNCDIR+"/Uncertainty_yield_ctau"+str(ctaupoint)+"_"+CHAN
    c2.Print(OUTSTRING+"_JER_Up_Down.png")
    c2.Print(OUTSTRING+"_JER_Up_Down.pdf")
    c2.Close()



def calc_unc_PU_weight(tree_weight_dict,sign,comb_fold_label="",add_label="",label_2="",pred_unc=0,eta=False,phi=False,eta_cut=False,phi_cut=False,scale_mu=1.,unc_set=[]):

    y_2 = defaultdict(dict)
    unc_2 = defaultdict(dict)
    #y_2_up = defaultdict(dict)
    #y_2_down = defaultdict(dict)

    if not os.path.isdir(OUTPUTDIR): os.mkdir(OUTPUTDIR)
    UNCDIR = OUTPUTDIR+CHAN+"/"
    if not os.path.isdir(UNCDIR): os.mkdir(UNCDIR)

    UNCDIR+=TAGVAR
    if eta_cut:
        UNCDIR += "_eta_1p0"
    if phi_cut:
        UNCDIR += "_phi_cut"

    if eta:
        UNCDIR += "_vs_eta"+comb_fold_label
    else:
        if phi:
            UNCDIR += "_vs_phi"+comb_fold_label
        else:
            UNCDIR += comb_fold_label

    UNCDIR += "/"
    if not os.path.isdir(UNCDIR):
        os.mkdir(UNCDIR)

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
        print '-'*11*2

    mass = []
    print "tree_weight_dict"
    print tree_weight_dict
    chainSignal = {}
    list_of_variables = ["nTagJets*","isMC","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight*","TriggerWeight*",CUT]#"nLeptons"
    if ERA=="2018":
        list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]

    for i,s in enumerate(sign):
        print "m chi: ",samples[s]['mass']
        print samples[s]['ctau']
        mass.append(samples[s]['mass'])
        chainSignal[s] = TChain("tree")
        tree_weights = {}
        chain_entries_cumulative = {}
        chain_entries = {}
        array_size_tot = 0

        for l, ss in enumerate(samples[s]['files']):
            print "ss", ss
            chainSignal[s].Add(NTUPLEDIR + ss + '.root')
            tree_weights[l] = tree_weight_dict[s][ss]
            chainSignal[s].Add(NTUPLEDIR + ss + ".root")
            chain_entries_cumulative[l] = chainSignal[s].GetEntries()
            if l==0:
                chain_entries[l]=chainSignal[s].GetEntries()
            else:
                chain_entries[l]=chain_entries_cumulative[l] - chain_entries_cumulative[l-1]
                print "Entries per sample ", ss, " : ", chain_entries[l]

            chunk_size = 100000
            n_iter = int(float(chain_entries[l])/float(chunk_size))
            c = 0
            if chain_entries[l]%chunk_size!=0:
                n_iter += 1

            print "chain_entries: ", chain_entries[l]
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter

            #Looping three times is inefficient but it allows to not modify the code
            for unc in unc_set:            
                bin2 = np.array([])
                #print "Doing this uncertainty: ", unc
                gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
                for arrays in gen:
                    st_it = time.time()
                    key_list = arrays.keys()
                    array_size_tot+=len(arrays[ key_list[0] ])
                    tree_w_array = tree_weight_dict[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                    cut_mask = (arrays[CUT]>0)

                    if eta_cut and phi_cut==False:
                        cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                        cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                        cut_mask = (cut_mask_eta.any()==True)

                    if phi_cut and eta_cut==False:
                        cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                        cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                        cut_mask = (cut_mask_phi.any()==True)

                    if phi_cut and eta_cut:
                        cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                        cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                        cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                        cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                        #This is needed to guarantee the shapes are consistent
                        cut_mask = (cut_mask_phi_eta.any()==True)

                    ### Cuts below depend on uncertainty
                    ### need to define a dictionary of yields per sample
                
                    if KILL_QCD:
                        ##Here effect of cuts on unc!
                        minjetmet_var = "MinJetMetDPhi"#+unc
                        print "minjetmet_var: ", minjetmet_var
                        cut_mask = np.logical_and( cut_mask, arrays[minjetmet_var]>0.5)

                    #SR cut
                    tag_var = "nTagJets_0p996_JJ"#+unc
                    tr_weight = "TriggerWeight"#always the same, not affected by PU
                    #print "tag_var: ", tag_var
                    cut_mask = np.logical_and( cut_mask, arrays[tag_var]>1 )
                    tag = arrays[tag_var][cut_mask] !=0
                    eventweight = arrays["EventWeight"][cut_mask]
                    triggerweight = arrays[tr_weight][cut_mask]
                    if unc=="":
                        pureweight = arrays["PUReWeight"][cut_mask]
                    else:
                        pureweight = arrays[unc][cut_mask]
                    del arrays
                    if scale_mu!=1.:
                        print "Scaling mu up by factor", scale_mu
                        weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[s][ss]*signalMultFactor*scale_mu
                    else:
                        weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[s][ss]*signalMultFactor
                    bin2 = np.concatenate( (bin2,np.multiply(tag,weight)) )

                del gen
                #Here: inside the loop of signal samples ss, and inside the loop of syst uncertainties
                if unc=="":
                    lab = "nominal"
                else:
                    lab=unc.replace("_","")
                y_2[ samples[s]['mass'] ][lab] = np.sum(bin2)#*tree_weight --> already in w
                #e_2 = np.sqrt( sum(x*x for x in bin2) ).sum()#*tree_weight --> already in w
                
        for unc in unc_set:
            if unc!="":
                lab = unc.replace("_","")
                #print s, lab, y_2[s][lab]
                unc_2[ samples[s]['mass'] ][lab] = 100.*abs(y_2[ samples[s]['mass'] ][lab] - y_2[ samples[s]['mass'] ]["nominal"])/(y_2[ samples[s]['mass'] ]["nominal"])
                print s, "uncert ", unc, unc_2[ samples[s]['mass'] ][lab]
            else:
                print s, "nominal", y_2[ samples[s]['mass'] ]["nominal"]

        #print y_2
        #print unc_2
        #here compute y_2 differences...
        #outname = "some_name"
        #print "Info: " , outname, " written"

    mass = np.array(mass)
    mass = np.sort(mass)

    graph_yield = {}#TGraph()
    graph_unc = {}#TGraph()
    n = 0
    for m in mass:
        #print m, y_2[m], unc_2[m]
        for unc in unc_set:
            lab = ""
            if unc!="":
                lab = unc.replace("_","")
                graph_unc[lab] = TGraph()
            else:
                lab = "nominal"
            graph_yield[lab] = TGraph()


    for m in mass:
        #print m, y_2[m], unc_2[m]
        for unc in unc_set:
            lab = ""
            if unc!="":
                lab = unc.replace("_","")
                graph_unc[lab].SetPoint(n,float(m),unc_2[m][lab])
            else:
                lab = "nominal"
            graph_yield[lab].SetPoint(n,float(m),y_2[m][lab])
            print lab
            print y_2[m].keys()
        n+=1

    
    #now plot
    colors = [801,856,825,881,2,602,880,798,856,5]
    lines = [1,2,2,3,3,4,4,5,5]
    gStyle.SetLegendFillColor(0)

    c1 = TCanvas("c1", "c1", 800, 600)
    c1.cd()
    c1.SetGrid()
    c1.GetPad(0).SetTopMargin(0.06)
    c1.GetPad(0).SetRightMargin(0.05)
    c1.GetPad(0).SetTicks(1, 1)
    c1.GetPad(0).SetGridx()
    c1.GetPad(0).SetGridy()
    c1.GetPad(0).SetLogy()
    c1.GetPad(0).SetLogx()
    top = 0.9
    nitems = len(unc_set)
    leg1 = TLegend(0.5,0.2+0.5,0.9,0.4+0.5)
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetTextSize(0.04)

    l = 0
    for unc in unc_set:
        lab = ""
        if unc!="":
            lab = unc.replace("_","")
            #do unc
        else:
            lab = "nominal"
        graph_yield[lab].SetLineStyle(lines[l])
        graph_yield[lab].SetLineWidth(3)
        graph_yield[lab].SetLineColor(colors[l])
        graph_yield[lab].SetMarkerColor(colors[l])
        graph_yield[lab].SetFillColorAlpha(colors[l],0.5)
        graph_yield[lab].SetMinimum(0.05)
        graph_yield[lab].SetMaximum(2.5)
        graph_yield[lab].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
        graph_yield[lab].GetYaxis().SetTitle("Event yield in bin2")
        graph_yield[lab].GetXaxis().SetNoExponent(True)
        graph_yield[lab].GetXaxis().SetMoreLogLabels(True)
        graph_yield[lab].GetXaxis().SetTitleSize(0.048)
        graph_yield[lab].GetYaxis().SetTitleSize(0.048)
        graph_yield[lab].GetYaxis().SetTitleOffset(0.8)
        graph_yield[lab].GetXaxis().SetTitleOffset(0.9)
        if l == 0:
            graph_yield[lab].SetMarkerStyle(20)
            graph_yield[lab].Draw("APL3")
        else:
            graph_yield[lab].SetMarkerStyle(20)
            graph_yield[lab].Draw("SAME,PL3")
        leg1.AddEntry(graph_yield[lab],lab,"L")
        l+=1
    leg1.Draw()

    if PRELIMINARY:
        drawCMS(samples, LUMI, "Preliminary",onTop=True,left_marg_CMS=0.1)
    else:
        drawCMS(samples, LUMI, "",left_marg_CMS=0.32)

    drawRegion(CHAN,top=0.7)
    drawAnalysis("LL"+CHAN)
    #drawTagVar(TAGVAR)

    OUTSTRING = UNCDIR+"/Event_yield_PU_weight_ctau"+str(ctaupoint)+"_"+CHAN
    c1.Print(OUTSTRING+".png")
    c1.Print(OUTSTRING+".pdf")
    c1.Close()



    c2 = TCanvas("c2", "c2", 800, 600)
    c2.cd()
    c2.SetGrid()
    c2.GetPad(0).SetTopMargin(0.06)
    c2.GetPad(0).SetRightMargin(0.05)
    c2.GetPad(0).SetTicks(1, 1)
    c2.GetPad(0).SetGridx()
    c2.GetPad(0).SetGridy()
    #c2.GetPad(0).SetLogy()
    c2.GetPad(0).SetLogx()
    leg2 = TLegend(0.5,0.2+0.5,0.9,0.4+0.5)
    leg2.SetBorderSize(0)
    leg2.SetFillStyle(0)
    leg2.SetTextSize(0.04)

    l = 0

    colors = [856,825,881,2,602,880,798,856,5]
    lines = [2,2,3,3,4,4,5,5]

    for unc in unc_set:
        lab = ""
        if unc!="":
            lab = unc.replace("_","")
            graph_unc[lab].SetLineStyle(lines[l])
            graph_unc[lab].SetLineWidth(3)
            graph_unc[lab].SetLineColor(colors[l])
            graph_unc[lab].SetMarkerColor(colors[l])
            graph_unc[lab].SetFillColorAlpha(colors[l],0.5)
            graph_unc[lab].SetMinimum(-0.5)
            graph_unc[lab].SetMaximum(10.)
            graph_unc[lab].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
            graph_unc[lab].GetYaxis().SetTitle("Uncertainty on yield in bin2 (%)")
            graph_unc[lab].GetXaxis().SetNoExponent(True)
            graph_unc[lab].GetXaxis().SetMoreLogLabels(True)
            graph_unc[lab].GetXaxis().SetTitleSize(0.048)
            graph_unc[lab].GetYaxis().SetTitleSize(0.048)
            graph_unc[lab].GetYaxis().SetTitleOffset(0.8)
            graph_unc[lab].GetXaxis().SetTitleOffset(0.9)
            if l == 0:
                graph_unc[lab].SetMarkerStyle(20)
                graph_unc[lab].Draw("APL3")
            else:
                graph_unc[lab].SetMarkerStyle(20)
                graph_unc[lab].Draw("SAME,PL3")
            leg2.AddEntry(graph_unc[lab],lab,"L")
            l+=1
    leg2.Draw()

    if PRELIMINARY:
        drawCMS(samples, LUMI, "Preliminary",onTop=True,left_marg_CMS=0.1)
    else:
        drawCMS(samples, LUMI, "",left_marg_CMS=0.32)

    drawRegion(CHAN,top=0.7)
    drawAnalysis("LL"+CHAN)
    #drawTagVar(TAGVAR)

    OUTSTRING = UNCDIR+"/Uncertainty_yield_PU_weight_ctau"+str(ctaupoint)+"_"+CHAN
    c2.Print(OUTSTRING+".png")
    c2.Print(OUTSTRING+".pdf")
    c2.Close()



        
def evaluate_median_expected_difference(file_names,labels,plot_label="",combination=False,eta=False,phi=False,eta_cut=False,phi_cut=False,LUMI=LUMI):

    if combination:
        print "COMMENTED because crashes, fixmeee"
        #exit()
        PLOTLIMITSDIR      = "plots/Limits/v5_calo_AOD_combination/"#"/"#"_2017_signal/"#
        #exit()
        LUMI = 137.4*1000
    else:
        PLOTLIMITSDIR      = "plots/Limits/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
    #    LUMI = LUMI

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()

    colors = [801,856,825,881,2,602,880,798,856,5]
    lines = [1,2,1,2,1,2,1,2,1,3]
    gStyle.SetLegendFillColor(0)
    c2 = TCanvas("c2", "c2", 800, 600)
    #c2 = TCanvas("c2", "c2", 1200, 600)
    c2.cd()
    c2.SetGrid()
    c2.GetPad(0).SetTopMargin(0.06)
    c2.GetPad(0).SetRightMargin(0.05)
    c2.GetPad(0).SetTicks(1, 1)
    c2.GetPad(0).SetGridx()
    c2.GetPad(0).SetGridy()
    #c2.GetPad(0).SetLogy()
    c2.GetPad(0).SetLogx()
    top = 0.9
    nitems = len(file_names)
    leg = TLegend(0.5,0.2+0.5,0.9,0.4+0.5)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("95% CL limits, c#tau_{"+particle+"}="+str(ctaupoint)+" mm")
    leg.SetTextSize(0.04)    
    graph_exp = {}
    graph_1sigma = {}

    x = {}
    y = {}

    for i,l in enumerate(file_names):
        filename = TFile(l, "READ")
        #graph_1sigma[i] = TGraph()
        #graph_exp[i] = TGraph()
        graph_1sigma[i] = filename.Get("pl"+str(ctaupoint)+"_1sigma")
        graph_exp[i] = filename.Get("pl"+str(ctaupoint)+"_exp")
        x[i] = []
        y[i] = []

        for r in range(graph_exp[i].GetN()):
            a = c_double()
            b = c_double()
            graph_exp[i].GetPoint(r,a,b)
            x[i].append(a.value)
            y[i].append(b.value)
        #print x[i]
        #print y[i]
        #print graph_exp[i].Print()
    #print y
    pair_diff = {}
    pair_graph = {}
    for l in range(0,len(x.keys()),2):
        pair_diff[l] = 100.*(np.array(y[l+1])-np.array(y[l]))/( (np.array(y[l])+np.array(y[l+1]))/2.  )
        pair_graph[l] = TGraph()
        #print l, y[l], y[l+1], pair_diff[l]
        for point in range(len(y[l])):
            pair_graph[l].SetPoint(point, x[l][point], pair_diff[l][point])
        pair_graph[l].SetLineStyle(lines[l])
        pair_graph[l].SetLineWidth(3)
        pair_graph[l].SetLineColor(colors[l])
        pair_graph[l].SetMarkerColor(colors[l])
        pair_graph[l].SetFillColorAlpha(colors[l],0.5)
        pair_graph[l].SetMinimum(-.1)
        pair_graph[l].SetMaximum(3.5)
        pair_graph[l].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
        pair_graph[l].GetYaxis().SetTitle("Impact on median expected limit (%)")
        pair_graph[l].GetXaxis().SetNoExponent(True)
        pair_graph[l].GetXaxis().SetMoreLogLabels(True)
        pair_graph[l].GetXaxis().SetTitleSize(0.048)
        pair_graph[l].GetYaxis().SetTitleSize(0.048)
        pair_graph[l].GetYaxis().SetTitleOffset(0.8)
        pair_graph[l].GetXaxis().SetTitleOffset(0.9)
        if l == 0:
            pair_graph[l].SetMarkerStyle(20)
            pair_graph[l].Draw("APL3")
        else:
            pair_graph[l].SetMarkerStyle(20)
            pair_graph[l].Draw("SAME,PL3")
        leg.AddEntry(pair_graph[l],labels[l+1],"L")
        filename.Close()
    leg.Draw()

    if PRELIMINARY:
        drawCMS(samples, LUMI, "Preliminary",onTop=True,left_marg_CMS=0.1)
    else:
        drawCMS(samples, LUMI, "",left_marg_CMS=0.32)

    drawRegion(CHAN,top=0.7)
    drawAnalysis("LL"+CHAN)
    drawTagVar(TAGVAR)

    OUTSTRING = PLOTLIMITSDIR+"/Expected_median_difference_ctau"+str(ctaupoint)+"_combination_"+CHAN
    c2.Print(OUTSTRING+plot_label+".png")
    c2.Print(OUTSTRING+plot_label+".pdf")
    c2.Close()

        
samples_to_run = sign#data#back#data#back#data#back#data#sign#back#data#back#data#back+data#data#data+back#+data
jet_tag = ""#+
clos = False#True#True#False#True

if KILL_QCD:
    jet_tag += "_MinDPhi_0p5"

if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]
    if CUT_PHI==True:
        print "Here phi cuts 2018"
        MINPHI = 1.0 #min allowed
        MAXPHI = 0.3 #max allowed
elif ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]
    if CUT_PHI==True:
        print "Here phi cuts 2017"
        MINPHI = 3.5 #min allowed
        MAXPHI = 2.5 #max allowed
elif ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    if CUT_PHI==True:
        print "Here phi cuts 2016 --> to be re-set as false"
        CUT_PHI=False
    if "_G-H" in jet_tag:
        print "Only GH!"
        LUMI  = lumi[ data[0] ]["G"]+lumi[ data[0] ]["H"]#["tot"]
    elif "_B-F" in jet_tag:
        LUMI  = lumi[ data[0] ]["B"]+lumi[ data[0] ]["C"]+lumi[ data[0] ]["D"]+lumi[ data[0] ]["E"]+lumi[ data[0] ]["F"]#["tot"]
    else:
        LUMI  = lumi[ data[0] ]["tot"]

#print "Warning!!!!!!!"
#print "Artificially setting lumi to 137*1000 /pb"
#LUMI = 137*1000
#print "Warning!!!!!!!"


print "Ntupledir: ", NTUPLEDIR
print "Luminosity: ", data[0], LUMI


mu_scale = 1.

calc_unc_PU_weight(
    get_tree_weights(sign),
    sign,
    comb_fold_label = jet_tag,
    add_label="",
    label_2=jet_tag,
    pred_unc = 1.,#here to boost it if needed
    eta = DO_ETA,
    phi = DO_PHI,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
    scale_mu=mu_scale,
    unc_set=["","PUReWeightUp","PUReWeightDown",],
)

#exit()


calc_unc_JER(
    get_tree_weights(sign),
    sign,
    comb_fold_label = jet_tag,
    add_label="",
    label_2=jet_tag,
    pred_unc = 1.,#here to boost it if needed
    eta = DO_ETA,
    phi = DO_PHI,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
    scale_mu=mu_scale,
    unc_set=["","_JERUp","_JERDown"],#"_JESUp","_JESDown","_UnclusteredEnUp","_UnclusteredEnDown"],#"_JER","_JESUp","_JESDown","_JERUp","_JERDown","_UnclusteredEnUp","_UnclusteredEnDown"],
)
#exit()

calc_unc(
    get_tree_weights(sign),
    sign,
    comb_fold_label = jet_tag,
    add_label="",
    label_2=jet_tag,
    pred_unc = 1.,#here to boost it if needed
    eta = DO_ETA,
    phi = DO_PHI,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
    scale_mu=mu_scale,
    unc_set=["","_JESUp","_JESDown","_UnclusteredEnUp","_UnclusteredEnDown"],#"_JER","_JESUp","_JESDown","_JERUp","_JERDown","_UnclusteredEnUp","_UnclusteredEnDown"],
)
exit()



evaluate_median_expected_difference(
    #signal contamination
    file_names = [
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_contamination.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_0p1.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_0p1_contamination.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_10p0.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_10p0_contamination.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_50p0.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_50p0_contamination.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_100p0.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_100p0_contamination.root",
    ],
    labels = ["no S contam. (#mu = 1.)", "with S contam. (#mu = 1.)","no S contam. (#mu = 0.1)", "with S contam. (#mu = 0.1)","no S contam. (#mu = 10.)", "with S contam. (#mu = 10.)","no S contam. (#mu = 50.)", "with S contam. (#mu = 50.)","no S contam. (#mu = 100.)", "with S contam. (#mu = 100.)",],
    plot_label = "_signal_contamination",
)
exit()
