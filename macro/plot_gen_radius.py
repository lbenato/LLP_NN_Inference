#! /usr/bin/env python

import os, multiprocessing
import copy
import math
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, THStack, TGraph, TMultiGraph, TGraphAsymmErrors, TSpline, TSpline3
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TBox

#from NNIferenceCMSSW.LLP_NN_Inference.samples_v3 import sample, samples
#from NNInferenceCMSSW.LLP_NN_Inference.samples import sample, samples
from NNInferenceCMSSW.LLP_NN_Inference.selections import selection
from NNInferenceCMSSW.LLP_NN_Inference.variables import *

gROOT.SetBatch(True)
gStyle.SetOptStat(0)
ERA = "2017"
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_Gen/"

#NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v0_SUSY_calo_MINIAOD_2018/"
from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
OUTPUTDIR = "plots/v5_calo_AOD_"+ERA+"_Gen/"


def plot_2D(sign,var,nbins=50,minimum=0,maximum=2000,bins=np.array([]),filename="",string="",part_var="GenBquarks",particle="#pi",norm=False):
    chain = {}
    hist = {}
    r_ecal = 129
    r_hcal = 179
    r_magnet = 295
    r_mb1 = 402
    r_mb4 = 738

    z_ecal = 300
    z_hcal = 376
    z_magnet = 0
    z_mb1 = 560

    if var=="radius2D":
        v_ecal = TLine(r_ecal,minimum,r_ecal,maximum)
        v_hcal = TLine(r_hcal,minimum,r_hcal,maximum)
        v_magnet = TLine(r_magnet,minimum,r_magnet,maximum)
        v_mb1 = TLine(r_mb1,minimum,r_mb1,maximum)
        v_mb4 = TLine(r_mb4,minimum,r_mb4,maximum)
        h_ecal = TLine(minimum,r_ecal,maximum,r_ecal)
        h_hcal = TLine(minimum,r_hcal,maximum,r_hcal)
        h_magnet = TLine(minimum,r_magnet,maximum,r_magnet)
        h_mb1 = TLine(minimum,r_mb1,maximum,r_mb1)
        h_mb4 = TLine(minimum,r_mb4,maximum,r_mb4)
    elif var=="z":
        v_ecal = TLine(z_ecal,minimum,z_ecal,maximum)
        v_hcal = TLine(z_hcal,minimum,z_hcal,maximum)
        v_magnet = TLine(z_magnet,minimum,z_magnet,maximum)
        v_mb1 = TLine(z_mb1,minimum,z_mb1,maximum)
        h_ecal = TLine(minimum,z_ecal,maximum,z_ecal)
        h_hcal = TLine(minimum,z_hcal,maximum,z_hcal)
        h_magnet = TLine(minimum,z_magnet,maximum,z_magnet)
        h_mb1 = TLine(minimum,z_mb1,maximum,z_mb1)
    else:
        v_ecal = TLine(r_ecal,minimum,r_ecal,maximum)
        v_hcal = TLine(r_hcal,minimum,r_hcal,maximum)
        v_magnet = TLine(r_magnet,minimum,r_magnet,maximum)
        v_mb1 = TLine(r_mb1,minimum,r_mb1,maximum)
        v_mb4 = TLine(r_mb4,minimum,r_mb4,maximum)
        h_ecal = TLine(minimum,r_ecal,maximum,r_ecal)
        h_hcal = TLine(minimum,r_hcal,maximum,r_hcal)
        h_magnet = TLine(minimum,r_magnet,maximum,r_magnet)
        h_mb1 = TLine(minimum,r_mb1,maximum,r_mb1)
        h_mb4 = TLine(minimum,r_mb4,maximum,r_mb4)

    v_ecal.SetLineColor(2)
    h_ecal.SetLineColor(2)
    v_hcal.SetLineColor(881)
    h_hcal.SetLineColor(881)
    v_magnet.SetLineColor(1)
    h_magnet.SetLineColor(1)
    v_mb1.SetLineColor(801)
    v_mb4.SetLineColor(4)
    h_mb1.SetLineColor(801)
    h_mb4.SetLineColor(4)

    v_ecal.SetLineWidth(4)
    h_ecal.SetLineWidth(4)
    v_hcal.SetLineWidth(4)
    h_hcal.SetLineWidth(4)
    v_magnet.SetLineWidth(4)
    h_magnet.SetLineWidth(4)
    v_mb1.SetLineWidth(4)
    h_mb1.SetLineWidth(4)
    v_mb4.SetLineWidth(3)
    h_mb4.SetLineWidth(3)

    v_ecal.SetLineStyle(3)
    h_ecal.SetLineStyle(3)
    v_hcal.SetLineStyle(2)
    h_hcal.SetLineStyle(2)
    v_magnet.SetLineStyle(4)
    h_magnet.SetLineStyle(4)
    v_mb1.SetLineStyle(8)
    h_mb1.SetLineStyle(8)
    v_mb4.SetLineStyle(9)
    h_mb4.SetLineStyle(9)

    leg = TLegend(1-0.9, 0.75, 1-0.75, 0.9)
    leg.AddEntry(v_ecal,"ECAL","L")
    leg.AddEntry(v_hcal,"HCAL","L")
    leg.AddEntry(v_magnet,"solenoid","L")
    leg.AddEntry(v_mb1,"MB1","L")
    leg.AddEntry(v_mb4,"MB4","L")

    #pal= 68 #kAvocado
    #pal= 64 #kAquamarine, very readable
    #pal= 75 #kCherry 75, awful
    #pal= 85 #kIsland 85, not beautiful but readable
    #pal= 86 #kLake 86, too violet
    #pal= 87 #kLightTemperature 87, used for trigger
    #pal= 91 #kPastel 91, too purple
    #pal= 100 #kSolar 100, very red and orange
    pal= 98 #kSandyTerrain 98, quite fine
    #pal= 99 #kSienna 99, a bit hard to read
    gStyle.SetPalette(pal)
    gStyle.SetPaintTextFormat(".f")

    if part_var=="GenBquarks":
        cutstring = "(EventWeight * PUReWeight) * ( (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v && Flag2_globalSuperTightHalo2016Filter && Flag2_goodVertices && Flag2_EcalDeadCellTriggerPrimitiveFilter && Flag2_HBHENoiseFilter && Flag2_HBHEIsoNoiseFilter && Flag2_ecalBadCalibFilter && Flag2_eeBadScFilter && Flag2_BadPFMuonFilter) && nMuonsPassing==0 && nElectronsPassing==0 && nPhotonsPassing==0 && nTausPassing==0 && MEt.pt>200 )"
        cutstring = "(EventWeight * PUReWeight) * (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v && MEt.pt>200 && nMuonsPassing==0 && nElectronsPassing==0 && nPhotonsPassing==0 && nTausPassing==0)"
        #cutstring = "(EventWeight * PUReWeight)"
    else:
        cutstring = "(EventWeight * PUReWeight) * ( (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v && Flag2_globalSuperTightHalo2016Filter && Flag2_goodVertices && Flag2_EcalDeadCellTriggerPrimitiveFilter && Flag2_HBHENoiseFilter && Flag2_HBHEIsoNoiseFilter && Flag2_ecalBadCalibFilter && Flag2_eeBadScFilter && Flag2_BadPFMuonFilter) && nMuonsPassing==0 && nElectronsPassing==0 && nPhotonsPassing==0 && nTausPassing==0 && MEt.pt>200 )"

    for i, s in enumerate(sign):
        chain[s] = TChain("tree")
        if filename=="":
            for p, ss in enumerate(samples[s]['files']):
                chain[s].Add(NTUPLEDIR + ss + ".root")
        else:
            chain[s].Add(NTUPLEDIR + filename+".root")
        print "Entries: ", chain[s].GetEntries()
        #filename[s] = TFile("VBFH_HToSSTobbbb_MH-125_MS-30_ctauS-1000.root", "READ")
        if len(bins) ==0:
            hist[s] = TH2F(s, "", nbins, minimum, maximum, nbins, minimum, maximum)
        else:
            hist[s] = TH2F(s, "", len(bins)-1, bins, len(bins)-1, bins)
        hist[s].Sumw2()
        if var=="z":
            #sign of eta for getting the right z value!
            chain[s].Project(s, "sqrt(pow("+part_var+"[0].radius,2) - pow("+part_var+"[0].radius2D,2)):sqrt(pow("+part_var+"[2].radius,2) - pow("+part_var+"[2].radius2D,2))", cutstring)
        else:
            if part_var=="GenBquarks":
                chain[s].Project(s, ""+part_var+"[0]."+var+":"+part_var+"[2]."+var+"", cutstring)
            else:
                chain[s].Project(s, ""+part_var+"[0]."+var+":"+part_var+"[1]."+var+"", cutstring)
        hist[s].SetOption("%s" % chain[s].GetTree().GetEntriesFast())
        if norm:
            hist[s].Scale(100./hist[s].Integral())
            gStyle.SetPaintTextFormat('5.1f')
        c1 = TCanvas("c1", "c1", 1000, 1000)
        c1.cd()
        #c1.SetGrid()
        #c1.SetLogz()
        c1.SetLogx()
        c1.SetLogy()
        hist[s].GetYaxis().SetTitle("Leading "+particle+" transverse decay length (cm)")#(""+part_var+"[0] "+var+" (cm)")
        hist[s].GetYaxis().SetTitleOffset(1.4)
        hist[s].GetXaxis().SetTitle("Sub-leading "+particle+" transverse decay length (cm)")#(""+part_var+"[2] "+var+" (cm)")
        hist[s].SetTitle(samples[s]['label'] if filename=="" else filename)
        hist[s].SetMarkerColor(0)#(2)#
        hist[s].Draw("colz")#()#
        v_ecal.Draw("sames")
        h_ecal.Draw("sames")
        v_hcal.Draw("sames")
        h_hcal.Draw("sames")
        v_magnet.Draw("sames")
        h_magnet.Draw("sames")
        v_mb1.Draw("sames")
        h_mb1.Draw("sames")
        v_mb4.Draw("sames")
        h_mb4.Draw("sames")
        hist[s].SetMarkerSize(1.2)#(2)#
        hist[s].Draw("text,sames")#()#
        leg.Draw("sames")
        c1.Print(OUTPUTDIR+"2D_gen_b_quark_"+var+"_"+(s if filename=="" else filename)+string+".png")
        c1.Print(OUTPUTDIR+"2D_gen_b_quark_"+var+"_"+(s if filename=="" else filename)+string+".pdf")

        if not gROOT.IsBatch(): raw_input("Press Enter to continue...")
        c1.Close()

taglio = "_preselections"
#taglio = "_nocuts"
#taglio = "_acceptance"
#taglio = "_MET_trigger"
#taglio = "_displaced_dijet_trigger"
#taglio = "_single_lepton_trigger"
#taglio = "_VBF_displaced_jet_trigger"
signal_15 = {
#    'ZH_M15_ctau100' :
#        {
#        'max' : 50,
#        },
    'VBFH_M15_ctau100' :
        {
        'max' : 50,
        },
#    'ggH_M15_ctau100' :
#        {
#        'max' : 50,
#        },
#    'ZH_M15_ctau1000' :
#        {
#        'max' : 500,
#        },
    'VBFH_M15_ctau1000' :
        {
        'max' : 500,
        },
#    'ggH_M15_ctau1000' :
#        {
#        'max' : 500,
#        },
#    'ZH_M15_ctau10000' :
#        {
#        'max' : 4000,
#        },
    'VBFH_M15_ctau10000' :
        {
        'max' : 4000,
        },
#    'ggH_M15_ctau10000' :
#        {
#        'max' : 4000,
#        },
}

signal = {
#    'ZH_M40_ctau100' :
#        {
#        'max' : 50,
#        },
    'VBFH_M30_ctau100' :
        {
        'max' : 50,
        },
#    'ggH_M30_ctau100' :
#        {
#        'max' : 50,
#        },
#    'ZH_M40_ctau1000' :
#        {
#        'max' : 500,
#        },
    'VBFH_M30_ctau1000' :
        {
        'max' : 500,
        },
#    'ggH_M30_ctau1000' :
#        {
#        'max' : 500,
#        },
#    'ZH_M40_ctau10000' :
#        {
#        'max' : 4000,
#        },
    'VBFH_M30_ctau10000' :
        {
        'max' : 4000,
        },
#    'ggH_M30_ctau10000' :
#        {
#        'max' : 4000,
#        },
}

signal_50 = {
    'ZH_M55_ctau100' :
        {
        'max' : 30,
        },
    'VBFH_M50_ctau100' :
        {
        'max' : 30,
        },
    'ggH_M50_ctau100' :
        {
        'max' : 30,
        },
    'ZH_M55_ctau1000' :
        {
        'max' : 400,
        },
    'VBFH_M50_ctau1000' :
        {
        'max' : 400,
        },
    'ggH_M50_ctau1000' :
        {
        'max' : 400,
        },
    'ZH_M55_ctau10000' :
        {
        'max' : 2000,
        },
    'VBFH_M50_ctau10000' :
        {
        'max' : 2000,
        },
    'ggH_M50_ctau10000' :
        {
        'max' : 2000,
        },
}

#for a in signal.keys():
#    print a
#    plot_2D([a],"radius2D",nbins=20,minimum=0,maximum=signal[a]['max'],bins=np.array([]),filename="",string=taglio)


#SUSY
signal = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh200_pl1000']#["mchi200_pl1000","mchi300_pl1000","mchi400_pl1000"]

plot_2D(signal,"travelRadiusLLP",nbins=50,minimum=9.9,maximum=50000,bins=np.array([9.9,25,50,100,250,500,1000,5000,10000,50000]),filename="",string=taglio,part_var="GenBquarks")

#for a in signal:
#    plot_2D([a],"radius2D",nbins=50,minimum=9.9,maximum=50000,bins=np.array([9.9,25,50,100,250,500,1000,5000,10000,50000]),filename=a,string=taglio,part_var="GenBquarks")

exit()


'''
#Heavy Higgs
signal = ["GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000",
"GluGluH2_H2ToSSTobbbb_MH-125_MS-8_ctauS-1000",
"GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-1000",
"GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-10000",
"GluGluH2_H2ToSSTobbbb_MH-125_MS-8_ctauS-10000",
"GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-10000",
"GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-1000",
"GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-1000",
"GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-1000",
"GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-10000",
"GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-10000",
"GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-10000",
"GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-1000",
"GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-1000",
"GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-1000",
"GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-10000",    "GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-10000",   "GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-10000","GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-1000",     "GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-1000","GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-10000",    "GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-10000",
"GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-500", 
"GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-2000", 
"GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-5000", 
"GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-500", 
"GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-2000", 
"GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-5000", 
"GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-500", 
"GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-2000", 
"GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-5000", 
"GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-500", 
"GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-2000", 
"GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-5000", 
"GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-500", 
"GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-2000", 
"GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-5000", 
"GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-500", 
"GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-2000", 
"GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-5000", 
"GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-500", 
"GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-2000", 
"GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-5000", 
"GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-500", 
"GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-2000", 
"GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-5000", 
"GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-500", 
"GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-2000", 
"GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-5000", 
"GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-500", 
"GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-2000", 
"GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-5000", 
]

for a in signal:
    plot_2D([a],"radius2D",nbins=50,minimum=9.9,maximum=50000,bins=np.array([9.9,50,100,250,500,1000,5000,10000,50000]),filename=a,string=taglio,part_var="GenBquarks")

'''

'''
##Heavy Higgs gen sim
signal = [
    "GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-500_TuneCP5_13TeV-pythia8",
    "GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-1000_TuneCP5_13TeV-pythia8",
    "GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-2000_TuneCP5_13TeV-pythia8",    
    "GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-5000_TuneCP5_13TeV-pythia8",
    "GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-10000_TuneCP5_13TeV-pythia8",
    "GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-500_TuneCP5_13TeV-pythia8",
    "GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8",
    "GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-2000_TuneCP5_13TeV-pythia8",    
    "GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-5000_TuneCP5_13TeV-pythia8",
    #"GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-10000_TuneCP5_13TeV-pythia8",
    ]
for a in signal:
    plot_2D([a],"radius2D",nbins=50,minimum=1,maximum=5000,bins=np.array([1.,5.,10.,50,100,250,500,1000,5000]),filename=a,string=taglio,part_var="GenBquarks", particle="S", norm=True)
'''

##SUSY

'''
signal = ['n3n2-n1-hbb-hbb_mh400_pl1000','n3n2-n1-hbb-hbb_mh300_pl1000','n3n2-n1-hbb-hbb_mh200_pl1000']

for a in signal:
    plot_2D([a],"radius2D",nbins=50,minimum=1,maximum=5000,bins=np.array([1.,5.,10.,50,100,250,500,1000,5000]),filename=a,string=taglio,part_var="GenHiggs", particle="b", norm=True)
exit()
'''

#HEAVY HIGGS

signal = ['ggH_MH1000_MS400_ctau500',
	'ggH_MH1000_MS400_ctau1000',
	'ggH_MH1000_MS400_ctau2000',
	'ggH_MH1000_MS400_ctau5000',
	'ggH_MH1000_MS400_ctau10000',
		
	'ggH_MH1000_MS150_ctau500',
	'ggH_MH1000_MS150_ctau1000',
	'ggH_MH1000_MS150_ctau2000',
	'ggH_MH1000_MS150_ctau5000',
	'ggH_MH1000_MS150_ctau10000']

plot_2D(signal,"radius2D",nbins=50,minimum=1,maximum=5000,bins=np.array([1.,5.,10.,50,100,250,500,1000,5000]),filename="",string=taglio,part_var="GenBquarks", particle="S", norm=True)

exit()
for a in signal:
    print a
    print samples[a]['files']
    plot_2D([a],"radius2D",nbins=50,minimum=1,maximum=5000,bins=np.array([1.,5.,10.,50,100,250,500,1000,5000]),filename=a,string=taglio,part_var="GenBquarks", particle="S", norm=True)


#plot_2D(a,"radius2D",nbins=20,minimum=0,maximum=500,bins=np.array([]),filename=a,string=taglio)

#plot_2D("radius2D",nbins=20,minimum=0,maximum=1000,bins=np.array([]),filename="VBFH_HToSSTobbbb_MH-125_MS-30_ctauS-1000",string=taglio)
#plot_2D("radius2D",nbins=20,minimum=0,maximum=2000,bins=np.array([]),filename="ZH_HToSSTobbbb_ZToLL_MH-125_MS-15_ctauS-10000",string=taglio)
#plot_2D("z",nbins=10,minimum=0,maximum=1000,bins=np.array([]),filename="VBFH_HToSSTobbbb_MH-125_MS-30_ctauS-1000")

