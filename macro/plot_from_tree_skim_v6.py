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
parser.add_option("-d", "--cut_d", action="store", type="string", dest="cut_d", default="")
parser.add_option("-r", "--region", action="store", type="string", dest="region", default="calo")
parser.add_option("-f", "--formula", action="store", type="string", dest="formula", default="")
parser.add_option("-t", "--treename", action="store", type="string", dest="treename", default="ntuple/tree")
parser.add_option("-B", "--blind", action="store_true", default=False, dest="blind")
(options, args) = parser.parse_args()
if options.bash: gROOT.SetBatch(True)
gStyle.SetOptStat(0)

#### NTUPLE, PLOT DIRECTORIES ####

ERA = "2016"
NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(ERA)+"_SR_HBHE/"
NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(ERA)+"_SR/"
NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+str(ERA)+"_SR/"
NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(ERA)+"_Gen/"
#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(ERA)+"_time_smearing_no_cuts/"
#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(ERA)+"_SR/"

#PLOTDIR     = "plots/v6_calo_AOD_"+str(ERA)+"_BeamHalo_studies_updated/"
PLOTDIR     = "plots/v6_calo_AOD_"+str(ERA)+"_SR_HBHE/"
PLOTDIR     = "plots/v6_calo_AOD_"+str(ERA)+"_SR/"
#PLOTDIR     = "plots/v5_calo_AOD_"+str(ERA)+"_SR/"
PLOTDIR     = "plots/v6_calo_AOD_"+str(ERA)+"_Gen/"

#PLOTDIR     = "plots/v6_calo_AOD_"+str(ERA)+"_BeamHalo_AN/"
#PLOTDIR     = "plots/v6_calo_AOD_"+str(ERA)+"_BeamHalo_AN_preappr/"

#PLOTDIR     = "plots/v6_calo_AOD_"+str(ERA)+"_cosmic_veto/"
#PLOTDIR     = "plots/v6_calo_AOD_"+str(ERA)+"_BeamHalo_studies/"

SIGNAL = 1#000#000#now!
POISSON     = False

#### SAMPLES ####

#data = ["data_obs"]
#data = ["SingleMuon"]
#data_tag =  "SingleMuon"#+"BH"#"MET"#"data_obs"
#data_tag = "SingleElectron"
#data_tag = "EGamma"
#data_tag = "MuonEG"
data_tag = "HighMET"#+"BH"
#data_tag = "JetHT"
data = [data_tag]
#back = ["VV","WJetsToQQ","WJetsToLNu","DYJetsToQQ","DYJetsToLL","ZJetsToNuNu","ST","TTbar","QCD"]#
#back = ["VV","WJetsToLNu","ZJetsToNuNu","TTbar"]
#back = [data_tag]#
#back = ["JetHTMC"]
#back = ["QCD"]
#back=["DYJetsToLL"]#,"TTbarGenMET","WJetsToLNu","ZJetsToNuNu"]
#back = ["WJetsToLNu"]
#back = ["VV","WJetsToLNu","ZJetsToNuNu","TTbarGenMET","QCD"]#good order for JetHT

back = [data_tag+"BH"]

#back = ["VV","TTbarGenMET","WJetsToLNu","QCD","ZJetsToNuNu"]#,"DYJetsToLL"]
#back = ["All"]
#back = [data_tag]
#back = [data_tag+"Copy"]
#back = [data_tag+"BH"]
#back = ["TTbarGenMET","WJetsToLNu","QCD","ZJetsToNuNu"]
#back = ["TTbarGenMET"]

if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    #LUMI  = lumi[ data[0] ]["tot"]#["tot"]
    if "BH" in data[0]:
        name = data[0].replace("BH","")
        LUMI  = lumi[ name ]["tot"]
    else:
        LUMI  = lumi[ data[0] ]["tot"]
elif ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    if "BH" in data[0]:
        name = data[0].replace("BH","")
        LUMI  = lumi[ name ]["tot"]
    else:
        LUMI  = lumi[ data[0] ]["tot"]
elif ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    #LUMI  = lumi[ data[0] ]["tot"]#["tot"]
    if "BH" in data[0]:
        name = data[0].replace("BH","")
        LUMI  = lumi[ name ]["tot"]
    else:
        LUMI  = lumi[ data[0] ]["tot"]

print "Luminosity: ", data[0], LUMI


sign = []
#sign = ['SUSY_mh400_pl1000']#,'SUSY_mh300_pl1000','SUSY_mh200_pl1000']#,'Event277096']#,'SUSY_mh200_pl1000','SUSY_mh150_pl1000']
#These: for gen studies
#sign += ['SUSY_mh127_ctau500','SUSY_mh300_ctau500','SUSY_mh600_ctau500','SUSY_mh1000_ctau500','SUSY_mh1800_ctau500']
#sign += ['SUSY_mh127_ctau3000','SUSY_mh300_ctau3000','SUSY_mh600_ctau3000','SUSY_mh1000_ctau3000','SUSY_mh1800_ctau3000']
sign = ['SUSY_mh400_ctau500_HH','SUSY_mh400_ctau3000_HH','SUSY_mh1800_ctau500_HH','SUSY_mh1800_ctau3000_HH']

sign = ['SUSY_mh1800_ctau500_HH',]#'SUSY_mh1800_ctau500_HH']
#back = ['dummy_1800']#,'SUSY_mh1800_ctau500_HH_tag']

sign = ['SUSY_mh400_ctau500_HH','SUSY_mh1000_ctau500_HH']
sign = ['SUSY_mh1000_ctau500_HH']
#data = back = []
#back = ["fake"]

#data = back = []
#sign = ["HighMETHBHE"]

#sign = [
#    'SUSY_mh400_ctau500_HH','SUSY_mh400_ctau500_HZ','SUSY_mh400_ctau500_ZZ',
#    'SUSY_mh400_ctau3000_HH','SUSY_mh400_ctau3000_HZ','SUSY_mh400_ctau3000_ZZ',
#]
#sign = ['SUSY_mh1800_ctau3000']
sign = ['SUSY_mh200_pl1000','SUSY_mh400_pl1000']

sign = ['SUSY_mh400_ctau500_HH','SUSY_mh1800_ctau500_HH','SUSY_mh400_ctau3000_HH','SUSY_mh1800_ctau3000_HH']
#sign = ['SUSY_mh400_ctau500','SUSY_mh1800_ctau500']

sign = []
sign = ['SUSY_mh400_ctau500_HH']
sign = ['SUSY_mh127_ctau500_HH','SUSY_mh150_ctau500_HH','SUSY_mh175_ctau500_HH','SUSY_mh200_ctau500_HH','SUSY_mh300_ctau500_HH','SUSY_mh600_ctau500_HH']
#sign = ['SUSY_mh127_ctau3000_HH','SUSY_mh150_ctau3000_HH','SUSY_mh175_ctau3000_HH','SUSY_mh200_ctau3000_HH','SUSY_mh300_ctau3000_HH','SUSY_mh600_ctau3000_HH']
sign = []

data = back = []
sign = ['SUSY_mh127_ctau500_HH','SUSY_mh150_ctau500_HH','SUSY_mh175_ctau500_HH','SUSY_mh200_ctau500_HH','SUSY_mh300_ctau500_HH','SUSY_mh400_ctau500_HH','SUSY_mh600_ctau500_HH','SUSY_mh1000_ctau500_HH','SUSY_mh1800_ctau500_HH']
sign = ['SUSY_mh127_ctau500_HH','SUSY_mh200_ctau500_HH','SUSY_mh300_ctau500_HH','SUSY_mh600_ctau500_HH','SUSY_mh1000_ctau500_HH','SUSY_mh1800_ctau500_HH']



def plot(var, cut, cut_s, cut_d="", tree_name="tree",norm=False):
    ### Preliminary Operations ###
    
    # Substitute cut
    pd = ""
    channel = ""
    plotdir = ""
    shortcut = cut
    shortcut_s = cut_s
    if cut_d=="":
        cut_d = cut
    longcut = longcut_s = longcut_d= ""
    if cut in selection:
        plotdir = cut
        longcut = selection[cut]
    if cut_s in selection:
        #The function does not work.
        ################longcut_s = "Jets.Jets[0].pt>30 && Jets.Jets[0].eta<2 && Jets.Jets[0].isGenMatched>-5 && Jets.Jets[0].isMatchedToMatchedCHSJet>-2 && Jets.Jets[0].isMatchedToMatchedCHSJet<=1 "#Jets.Jets[0].isMatchedToMatchedCHSJet"#selection[cut_s] #+ " && Jets.Jets[0].isMatchedToMatchedCHSJet "# + signal_matching_string(var)
        #longcut_s = selection[cut_s] + signal_matching_string(var)
        #print "VERIFY: " , longcut_s
        longcut_s = selection[cut_s]
    if cut_d in selection:
        longcut_d = selection[cut_d]

    # Determine Primary Dataset
    pd = getPrimaryDataset(samples, longcut_d, data_tag=data_tag)
    if len(data)>0 and len(pd)==0: raw_input("Warning: Primary Dataset not recognized, continue?")
    
    # Determine weight
    print "!!!!!!!!!"
    #print "Artificially enhancing by factor 2 for comparison!!!!!!"
    #weight = "EventWeight*2"
    #weight = "2"
    #weight = "1"
    #weight = "EventWeight*PUReWeight*TriggerWeight"
    #weight = "EventWeight*PUWeight"
    weight = "EventWeight"
    print weight

    print "Considered ntuples: ", NTUPLEDIR
    print "Plotting", var#, "in", channel, "channel with:"
    print "  dataset:", pd
    print "  weight :", weight
    print "  cut    :", longcut
    print "  cut on signal    :", longcut_s
    print "  cut on data    :", longcut_d
    suffix = ""

    for i, s in enumerate(back):
        print "back sample: ", s


    ### Create and fill MC histograms ###
    print "doing project . . . "
    hist = project(samples, var, longcut, longcut_s, longcut_d, weight, data+back+sign, pd, NTUPLEDIR, treename=tree_name,formula=options.formula,alpha=.4)
    
    # Background sum
    if len(back)>0:
        if options.blind: RATIO = False
        else: RATIO = 4
        hist['BkgSum'] = hist[data_tag].Clone("BkgSum") if data_tag in hist else hist[back[0]].Clone("BkgSum")
        hist['BkgSum'].Reset("MICES")
        hist['BkgSum'].SetFillStyle(3003)
        hist['BkgSum'].SetFillColor(1)
        for i, s in enumerate(back):
            hist['BkgSum'].Add(hist[s])
    
    if len(back)==0 and len(data)==0:
        suffix = ''
        RATIO = False
        for i, s in enumerate(sign):
            #print "scaling: ",  s
            print "I won't scale signal!"
            #hist[s].Scale(1./hist[s].Integral())
            hist[s].SetFillStyle(0)
    
    if norm and len(data+back)>0:
        sfnorm = hist[data_tag].Integral()/hist['BkgSum'].Integral()
        for i, s in enumerate(back+['BkgSum']): hist[s].Scale(sfnorm)
        
    ### Plot ###

    if len(data+back)>0:
        if options.blind: RATIO = 0
        else: RATIO = 4
        out = draw(samples, hist, data if not options.blind else [], back, sign, SIGNAL, RATIO, POISSON, variable[var]['log'],data_tag=data_tag)
    else:
        out = drawSignal(samples, hist, sign,variable[var]['log'])
        #out[0].SetGrid()

    # Other plot operations
    out[0].cd(1)
    drawCMS(samples, LUMI, "Preliminary" if len(data+back)>0 else "Simulation",onTop=True if len(data+back)>0 else False,data_obs=data)
    #Avoid overlapping region
    #drawCMS(samples, LUMI, "Preliminary" if len(data+back)>0 else "Simulation",onTop=False,data_obs=data)

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
    if len(data+back)>0: printTable(samples, hist, sign, SIGNAL,  data_tag=data_tag)
    
    if not gROOT.IsBatch(): raw_input("Press Enter to continue...")


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

##plot(options.variable, options.cut, options.cut_s, "skim")
plot(options.variable, options.cut, options.cut_s, options.cut_d, "tree",norm=True)
