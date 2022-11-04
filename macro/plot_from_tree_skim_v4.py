#! /usr/bin/env python

import os, multiprocessing
import copy
import math
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, gPad
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, THStack, TGraph, TGraphAsymmErrors
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TBox, TGaxis

#### IMPORT SAMPLES AND VARIABLES DICTIONARIES ####

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *


#### PARSER ####

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
parser.add_option("-v", "--variable", action="store", type="string", dest="variable", default="")
parser.add_option("-c", "--cut", action="store", type="string", dest="cut", default="")
parser.add_option("-s", "--cut_s", action="store", type="string", dest="cut_s", default="")
parser.add_option("-r", "--region", action="store", type="string", dest="region", default="")
parser.add_option("-f", "--formula", action="store", type="string", dest="formula", default="")
parser.add_option("-t", "--treename", action="store", type="string", dest="treename", default="ntuple/tree")
parser.add_option("-B", "--blind", action="store_true", default=False, dest="blind")
(options, args) = parser.parse_args()
if options.bash: gROOT.SetBatch(True)
gStyle.SetOptStat(0)

#### NTUPLE, PLOT DIRECTORIES ####

#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_all_events/"
#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged/"
#PLOTDIR     = "plots/v4_calo_AOD_2018_tf_and_skim_all_events/"
#PLOTDIR     = "plots/v4_calo_AOD_2018_tf_and_skim/"

#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_HT_cut/"
#PLOTDIR     = "plots/v4_calo_AOD_2018_tf_and_skim_no_HT_cut/"

#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ/"
#PLOTDIR     = "plots/v4_calo_AOD_2018_tf_and_skim_no_cuts_compare_JJ/"

#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ_tagger_AK4_v2_LUMI_JJ/"
#PLOTDIR     = "plots/v4_calo_AOD_2018_tf_and_skim_no_cuts_compare_JJ_tagger_AK4_v2_LUMI_JJ/"

#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ_tagger_v3/"
#PLOTDIR     = "plots/v4_calo_AOD_2018_tf_and_skim_no_cuts_compare_JJ_tagger_v3/"

NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_AK4_v3__AK8_v2/"
PLOTDIR     = "plots/v4_calo_AOD_2018_tf_and_skim_AK4_v3__AK8_v2/"

NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_debug_old/"
PLOTDIR     = "plots/v4_calo_AOD_2018_debug_old/"

#v4_calo_AOD_2018_synch
#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_synch/"
#PLOTDIR     = "plots/v4_calo_AOD_2018_synch/"

#from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018_skim import sample, samples
from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples

LUMI = 13967.832433416#35867
LUMI = 59.74*1000
LUMI = 95.9*1000

#SIGNAL      = 10000######LLL
SIGNAL = 100#good for met
#SIGNAL = 100000#good for ZH
#SIGNAL = 10
SIGNAL = 1#now!

POISSON     = False

#### SAMPLES ####

data = ["data_obs"]
#back = ["VV","WJetsToQQ","WJetsToLNu","DYJetsToQQ","DYJetsToLL","ZJetsToNuNu","ST","TTbar","QCD"]#
back = ["VV","WJetsToLNu","ZJetsToNuNu","TTbar","QCD"]
#back = ["VV","WJetsToLNu","ZJetsToNuNu","TTbar"]
back = ["VV","TTbar","WJetsToLNu","QCD","ZJetsToNuNu"]
#back = ["VV","TTbar","QCD","ZJetsToNuNu"]
back = ["QCD"]
#data = back = []
sign = []

sign_b = ["VBFH_M20_ctau0","VBFH_M20_ctau0p1", "VBFH_M20_ctau1"]#
sign_track = ["VBFH_M20_ctau5","VBFH_M20_ctau10", "VBFH_M20_ctau100"]#
#sign_calo = []
sign_calo = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh200_pl1000']
sign_calo = ['ggH_MH600_MS50_ctau1000','ggH_MH600_MS150_ctau1000','ggH_MH125_MS25_ctau1000','ggH_MH125_MS55_ctau1000',]
##sign_calo = ['SUSY_mh400_pl1000','ggH_MH1000_MS400_ctau1000','ggH_MH1000_MS400_ctau10000','ggH_MH1000_MS150_ctau500','ggH_MH1000_MS150_ctau1000',]

sign_calo = ['SUSY_mh400_pl1000_XL','ggH_MH2000_MS250_ctau1000','ggH_MH1000_MS150_ctau1000_XL','ggH_MH600_MS50_ctau1000_XL']

sign_calo = ['SUSY_mh400_pl1000_XL','SUSY_mh200_pl1000']

#boosted
#sign_calo = ['ggH_MH2000_MS250_ctau1000','ggH_MH1500_MS200_ctau1000','ggH_MH1000_MS150_ctau1000_XL','ggH_MH600_MS50_ctau1000_XL','ggH_MH400_MS50_ctau1000']
sign_calo = ['ggH_MH2000_MS250_ctau1000','ggH_MH1500_MS200_ctau1000','ggH_MH1000_MS150_ctau1000','ggH_MH600_MS50_ctau1000']

sign_calo = ['ggH_MH2000_MS250_ctau1000','ggH_MH2000_MS50_ctau1000']
sign_calo = ['ggH_MH2000_MS250_ctau1000','ggH_MH2000_MS50_ctau1000','ggH_MH2000_MS50_ctau100']


#resolved
#sign_calo = ['ggH_MH2000_MS600_ctau1000','ggH_MH1500_MS500_ctau1000','ggH_MH1000_MS400_ctau1000','ggH_MH600_MS150_ctau1000_XL','ggH_MH400_MS100_ctau1000']
sign_calo = ['SUSY_mh200_pl1000']

#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v0_gen_studies_calo_AOD_HeavyHiggsSUSY/"
#PLOTDIR     = "plots/v0_gen_studies_calo_AOD_SUSY/"
#back = data = []

#back = ["WJetsToLNuIncl"]
#sign_calo = ["WJetsToLNuRed"]

SIGNAL = 1
if options.region == "b":
    sign = sign_b
    SIGNAL      = 100000
elif options.region == "track":
    sign = sign_track
elif options.region == "calo":
    sign = sign_calo
#    if "METTrigger" in options.cut:
    #SIGNAL = 1#SIGNAL/10
elif options.region == "calo_ggH":
    sign = sign_calo_ggH
elif options.region == "all":#pick all signals
    for a in samples.keys():
        if "VBFH_M" in a and a!="VBFH_M25_ctau5":
            #print a
            sign.append(a)
    SIGNAL      = 100
else:
    print "No region specified, plotting without signal!"
    print "Aborting!"
    exit()


#SIGNAL = SIGNAL*10

#print sign
#exit()

#### Function not working!
#def signal_matching_string(var):
#    signal_cut_additional_string = ""
#    if "Jets.Jets" in var:
#        rest = var[0:var.index(']')+1]
#        signal_cut_additional_string = " && ( " + rest + ".pt==MatchedCHSJet1.pt || " + rest + ".pt==MatchedCHSJet2.pt || " + rest + ".pt==MatchedCHSJet3.pt || " + rest + ".pt==MatchedCHSJet4.pt ) "
#    return signal_cut_additional_string
####


def plot(var, cut, cut_s, tree_name="tree",norm=False):
    ### Preliminary Operations ###
    
    # Substitute cut
    pd = ""
    channel = ""
    plotdir = ""
    shortcut = cut
    shortcut_s = cut_s
    longcut = longcut_s = ""
    if cut in selection:
        plotdir = cut
        longcut = selection[cut]
    if cut_s in selection:
        #The function does not work.
        ################longcut_s = "Jets.Jets[0].pt>30 && Jets.Jets[0].eta<2 && Jets.Jets[0].isGenMatched>-5 && Jets.Jets[0].isMatchedToMatchedCHSJet>-2 && Jets.Jets[0].isMatchedToMatchedCHSJet<=1 "#Jets.Jets[0].isMatchedToMatchedCHSJet"#selection[cut_s] #+ " && Jets.Jets[0].isMatchedToMatchedCHSJet "# + signal_matching_string(var)
        #longcut_s = selection[cut_s] + signal_matching_string(var)
        #print "VERIFY: " , longcut_s
        longcut_s = selection[cut_s]

    # Determine Primary Dataset
    pd = getPrimaryDataset(samples, longcut)
    if len(data)>0 and len(pd)==0: raw_input("Warning: Primary Dataset not recognized, continue?")
    
    # Determine weight
    print "!!!!!!!!!"
    #print "Artificially enhancing by factor 2 for comparison!!!!!!"
    #weight = "EventWeight*2"
    #weight = "2"
    weight = "EventWeight"
    print weight

    print "Considered ntuples: ", NTUPLEDIR
    print "Plotting", var#, "in", channel, "channel with:"
    print "  dataset:", pd
    print "  weight :", weight
    print "  cut    :", longcut
    print "  cut on signal    :", longcut_s
    suffix = ""

    for i, s in enumerate(back):
        print "back sample: ", s


    ### Create and fill MC histograms ###
    print "doing project . . . "
    hist = project(samples, var, longcut, longcut_s, weight, data+back+sign, pd, NTUPLEDIR, treename=tree_name,formula=options.formula,alpha=.6)
    
    # Background sum
    if len(back)>0:
        if options.blind: RATIO = False
        else: RATIO = 4
        hist['BkgSum'] = hist['data_obs'].Clone("BkgSum") if 'data_obs' in hist else hist[back[0]].Clone("BkgSum")
        hist['BkgSum'].Reset("MICES")
        hist['BkgSum'].SetFillStyle(3003)
        hist['BkgSum'].SetFillColor(1)
        for i, s in enumerate(back):
            hist['BkgSum'].Add(hist[s])
    
    if len(back)==0 and len(data)==0:
        suffix = ''
        RATIO = False
        for i, s in enumerate(sign):
            print "I won't scale signal!"
            #hist[s].Scale(1./hist[s].Integral())
            hist[s].SetFillStyle(0)
    
    if norm:
        sfnorm = hist['data_obs'].Integral()/hist['BkgSum'].Integral()
        for i, s in enumerate(back+['BkgSum']): hist[s].Scale(sfnorm)
        
    ### Plot ###

    if len(data+back)>0:
        if options.blind: RATIO = 0
        else: RATIO = 4
        out = draw(samples, hist, data if not options.blind else [], back, sign, SIGNAL, RATIO, POISSON, variable[var]['log'])
    else:
        out = drawSignal(samples, hist, sign,variable[var]['log'])
        out[0].SetGrid()

    # Other plot operations
    out[0].cd(1)
    drawCMS(samples, LUMI, "Preliminary" if len(data+back)>0 else "Simulation",onTop=True if len(data+back)>0 else False,data_obs=data)
    #drawCMS(samples, LUMI, "Work in Progress",data_obs=data)
    drawRegion(shortcut)
    drawAnalysis("LLSUSY")
    #drawAnalysis("LLggHeavyHiggs")   
    #drawAnalysis("LLZH")
    out[0].Update()
    
    # Save
    SAVE = True
    pathname = PLOTDIR+plotdir
    #if gROOT.IsBatch() and SAVE:
    if SAVE:
        if not os.path.exists(pathname): os.makedirs(pathname)
        suffix+= "_"+str(options.region)
        if len(data+back)>0:
            if var=="@CSCSegments.size()":
                var = "nCSCSegments"
            if var=="@PFCandidatesAK8.size()":
                var = "nPFCandidatesAK8"
            out[0].Print(pathname+"/"+var.replace('.', '_').replace('/','_div_')+suffix+".png")
            out[0].Print(pathname+"/"+var.replace('.', '_').replace('/','_div_')+suffix+".pdf")
        else:
            if var=="@PFCandidatesAK8.size()":
                var = "nPFCandidatesAK8"
            out[0].Print(pathname+"/"+var.replace('.', '_')+suffix+"_signal.png")
            out[0].Print(pathname+"/"+var.replace('.', '_')+suffix+"_signal.pdf")    
    ### Other operations ###
    # Print table
    if len(data+back)>0: printTable(samples, hist, sign, SIGNAL)
    
    if not gROOT.IsBatch(): raw_input("Press Enter to continue...")


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

##plot(options.variable, options.cut, options.cut_s, "skim")
plot(options.variable, options.cut, options.cut_s, "tree")
