#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import numpy as np
from array import array
from collections import defaultdict
#from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, THnF
#from ROOT import TStyle, TCanvas, TPad
#from ROOT import TLegend, TLatex, TText, TLine, TProfile
import uproot
#import root_numpy
import yaml
from prettytable import PrettyTable

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
(options, args) = parser.parse_args()
##if options.bash: 
#gROOT.SetBatch(True)
#gStyle.SetOptStat(0)

from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.reinterpretation_variables import *

ERA = "2018"
LUMI = -1

if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI  = lumi[ "HighMET" ]["tot"]
elif ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI  = lumi[ "HighMET" ]["tot"]
elif ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI  = lumi[ "HighMET" ]["tot"]

NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_reinterpretation/"
MAIN_OUT = "plots/v6_calo_AOD_"+ERA+"_reinterpretation_v1/"

#new ntuples aftrer T2 accident
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v7_calo_AOD/v7_calo_AOD_"+ERA+"_reinterpretation/"
MAIN_OUT = "plots/v7_calo_AOD_"+ERA+"_reinterpretation/"


#new ntuples after post-processing
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v7_calo_AOD/v7_calo_AOD_"+ERA+"_reinterpretation/post_processed/"
MAIN_OUT = "plots/v7_calo_AOD_"+ERA+"_reinterpretation/"

sign = [
    "SUSY_mh1800_ctau500_ZZ",
    "SUSY_mh1000_ctau500_ZZ",
    "SUSY_mh400_ctau500_ZZ",
    "SUSY_mh127_ctau500_ZZ",
    "SUSY_mh1800_ctau3000_ZZ",
    "SUSY_mh1000_ctau3000_ZZ",
    "SUSY_mh400_ctau3000_ZZ",
    "SUSY_mh127_ctau3000_ZZ",
]

gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)
gStyle.SetPadTopMargin(0.06)
gStyle.SetPadRightMargin(0.05)
gStyle.SetPadRightMargin(0.15)

h = {}
num = {}
den = {}
eff = {}
tree = {}
WEIGHT = "EventWeight"#"EventWeight*TriggerWeight*PUReWeight"

def pl(sign, tree_name="tree",var="Jets.sigprob",cut_d="",cut_n="",do_radius=False,add_label="",eff_mode=True,max_eff=1.01,norm=False):

    print "cut den: ", cut_d

    min_x = 0.
    max_x = 0.
    for s in sign:
        if variable[var]["nbins"]!=0:
            print var
            print "nbins: ", variable[var]['nbins']
            num[s] = TH1F("num"+s+var,"num"+s+var,variable[var]["nbins"],variable[var]["min"],variable[var]["max"])
            den[s] = TH1F("den"+s+var,"den"+s+var,variable[var]["nbins"],variable[var]["min"],variable[var]["max"])
            min_x = variable[var]["min"]
            max_x = variable[var]["max"]
        else:
            print "bins: ", variable[var]['bins']
            num[s] = TH1F("num"+s+var,"num"+s+var,len(variable[var]['bins'])-1, array('f', variable[var]['bins']))
            den[s] = TH1F("den"+s+var,"den"+s+var,len(variable[var]['bins'])-1, array('f', variable[var]['bins']))
            min_x = variable[var]["bins"][0]
            max_x = variable[var]["bins"][-1]
        num[s].Sumw2()
        den[s].Sumw2()
        eff[s] = TGraphAsymmErrors()


    #list_of_variables = ["nTagJets_0p996","isMC","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB","JetsNegative.*","JetsNegative.pt","JetsNegative.phi","JetsNegative.eta","JetsNegative.sigprob","JetsNegative.timeRMSRecHitsEB","JetsNegative.nRecHitsEB","JetsNegative.timeRecHitsEB","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","TriggerWeight","PUWeight","PUReWeight","GenLLPs.*","MeanNumInteractions","HT","Jets.nTrackConstituents","isLLP*","nLLP*","nJets*","nTagJets*"]

    #list_of_variables_data = list_of_variables  + ["dt_ecal_dist","min_dPhi_jets*"]

    tree_weight_dict = {}
    #first round, get tree weight from mc
    for b in sign:
        for i, ss in enumerate(samples[b]['files']):
            if ('Run201') in ss:
                t_w = 1.
            else:
                if not os.path.isfile(NTUPLEDIR+ss+'.root'):
                    print("!!!File ", NTUPLEDIR+ss+'.root', " does not exist! Continuing")
                    continue
                filename = TFile(NTUPLEDIR+ss+'.root', "READ")
                if filename.GetListOfKeys().Contains("c_nEvents"):
                    nevents = filename.Get("c_nEvents").GetBinContent(1)
                    tree_entries = filename.Get("tree").GetEntries()
                    print "tree entries: ", tree_entries
                    if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                        print "SUSY central, consider sample dictionary for nevents!"
                        nevents = sample[ss]['nevents']
                        print "n gen events: ", nevents

                    if nevents!=tree_entries:
                        print "There are some selections applied! Aborting...."
                        exit()
                    b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
                    n_pass      = filename.Get("n_pass").GetBinContent(1)
                    n_odd       = filename.Get("n_odd").GetBinContent(1)
                    filename.Close()
                    xs = sample[ss]['xsec'] * sample[ss]['kfactor']
                    if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                        print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                        xs = 1.
                    t_w = LUMI * xs / nevents
                    if(b_skipTrain>0):
                        if(n_odd>0):
                            t_w *= float(n_pass/n_odd)
                else:
                    t_w = 0

            tree_weight_dict[ss] = t_w

    chunk_size = 100000


    print tree_weight_dict


    #Now: 
    #Efficiency of creating at least 1 tagged jet when exactly 1 LLPs in acceptance
    #Efficiency of creating at least 2 tagged jet when exactly 2 LLPs in acceptance

    can = TCanvas("can","can",1000,800)
    can.cd()
    can.SetGrid()
    #can.SetLogy()
    can.SetLogz()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)


    leg = TLegend(0.55, 0.7, 0.95, 0.95)
    leg.SetTextSize(0.035)

    max_y = 0
    for i,s in enumerate(sign):
        print "Filling ", s, i
        tree[s] = TChain("tree")
        for j, ss in enumerate(samples[s]['files']):
            tree[s].Add(NTUPLEDIR + ss + '.root')

        tree[s].Project("den"+s+var,var,"("+cut_d+")*"+WEIGHT)
        tree[s].Project("num"+s+var,var,"("+cut_n+")*"+WEIGHT)
        if norm:
            num[s].Scale(1./num[s].Integral())
            den[s].Scale(1./den[s].Integral())
        max_y = max(max_y,num[s].GetMaximum())
        max_y = max(max_y,den[s].GetMaximum())

        addOverflow(num[s], True)
        addOverflow(den[s], True)

    if eff_mode:
        max_y = max_eff#1.01
    else:
        max_y *=1.01

    min_y = -0.01
    #if "travelRadius" in var:
    #    min_y = 0.00001

    r_ecal = 129
    r_hcal = 179
    r_magnet = 295
    r_mb1 = 402
    r_mb4 = 738

    v_ecal = TLine(r_ecal,min_y,r_ecal,max_y)
    v_hcal = TLine(r_hcal,min_y,r_hcal,max_y)
    v_magnet = TLine(r_magnet,min_y,r_magnet,max_y)
    v_mb1 = TLine(r_mb1,min_y,r_mb1,max_y)
    v_mb4 = TLine(r_mb4,min_y,r_mb4,max_y)


    v_ecal.SetLineColor(801)
    v_hcal.SetLineColor(881)
    v_magnet.SetLineColor(16)
    v_mb1.SetLineColor(8)
    v_mb4.SetLineColor(8)

    v_ecal.SetLineWidth(6)
    v_hcal.SetLineWidth(6)
    v_magnet.SetLineWidth(6)
    v_mb1.SetLineWidth(6)
    v_mb4.SetLineWidth(6)

    v_ecal.SetLineStyle(3)
    v_hcal.SetLineStyle(2)
    v_magnet.SetLineStyle(1)
    v_mb1.SetLineStyle(8)
    v_mb4.SetLineStyle(9)

    leg_l = TLegend(0.15, 0.7, 0.3, 0.92)
    leg_l.AddEntry(v_ecal,"ECAL","L")
    leg_l.AddEntry(v_hcal,"HCAL","L")
    leg_l.AddEntry(v_magnet,"solenoid","L")
    leg_l.AddEntry(v_mb1,"MB1","L")
    leg_l.AddEntry(v_mb4,"MB4","L")


    for i,s in enumerate(sign):
        if eff_mode:
            eff[s].BayesDivide(num[s],den[s])
            eff[s].GetYaxis().SetTitle('Efficiency')
            fake_num = TGraph(num[s])
            fake_den = TGraph(den[s])
            print s, " num: ", fake_num.Print()
            print s, " den: ", fake_den.Print()
            del fake_num
            del fake_den
            print s, " graph: ", eff[s].Print()

        else:
            eff[s] = TGraphAsymmErrors(den[s])#
            eff[s].GetYaxis().SetTitle('Events')
        eff[s].GetXaxis().SetTitle(variable[var]['title'])
        eff[s].GetXaxis().SetRangeUser(min_x,max_x)
        eff[s].SetMarkerColor(samples[s]['linecolor'])
        if samples[s]['ctau']==500:
            eff[s].SetMarkerStyle(20)
        if samples[s]['ctau']==3000:
            eff[s].SetMarkerStyle(25)
        eff[s].SetLineColor(samples[s]['linecolor'])
        eff[s].SetLineStyle(samples[s]['linestyle'])
        eff[s].SetLineWidth(2)
        eff[s].SetMaximum(max_y)
        eff[s].SetMinimum(min_y)
        if i==0:
            if eff_mode:
                eff[s].Draw("APL")
            else:
                eff[s].Draw("AP")
        else:
            if eff_mode:
                eff[s].Draw("PL,sames")
            else:
                eff[s].Draw("P,sames")

        leg.AddEntry(eff[s],samples[s]['label'],"PL")

        
    if "travelRadius" in var:
        v_ecal.Draw("sames")
        v_hcal.Draw("sames")
        v_magnet.Draw("sames")
        v_mb1.Draw("sames")
        v_mb4.Draw("sames")

        for i,s in enumerate(sign):
            if eff_mode:
                eff[s].Draw("PL,sames")
            else:
                eff[s].Draw("P,sames")

        leg_l.Draw()
        #can.SetLogx()
        #eff[s].SetMinimum(0.00001)


    leg.Draw()
    drawCMS_simple(-1, "Simulation", ERA=ERA, onTop=True)
    add_eff_string=""
    if eff_mode:
        add_eff_string = "Eff_"
    can.Print(MAIN_OUT+add_eff_string+var.replace('.', '_').replace('/','_div_')+add_label+'.png')
    can.Print(MAIN_OUT+add_eff_string+var.replace('.', '_').replace('/','_div_')+add_label+'.pdf')
    can.Close()

    #at den: 1 LLP in acceptance (if there are 2: sum up)
    #num: the LLP made at least 1 tagged jet, if there are 2 jets associated to the same particle:
    #if there are no tagged jets: still have to fill denominator

def create_maps(sign, skip_even=False):

    list_of_variables = [
        "GenLLPs*","GenBquarks*","GenHiggs*","nTagJets_0p996","nTagJets_to_LLP0","nTagJets_to_LLP1","nJets_to_LLP0","nJets_to_LLP1","isPreselected","pt","EventNumber",
    ]

    maps = defaultdict(dict)

    '''
    tree_weight_dict = {}
    #first round, get tree weight from mc
    for b in sign:
        for i, ss in enumerate(samples[b]['files']):
            if ('Run201') in ss:
                t_w = 1.
            else:
                if not os.path.isfile(NTUPLEDIR+ss+'.root'):
                    print("!!!File ", NTUPLEDIR+ss+'.root', " does not exist! Continuing")
                    continue
                filename = TFile(NTUPLEDIR+ss+'.root', "READ")
                if filename.GetListOfKeys().Contains("c_nEvents"):
                    nevents = filename.Get("c_nEvents").GetBinContent(1)
                    tree_entries = filename.Get("tree").GetEntries()
                    print "tree entries: ", tree_entries
                    if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                        print "SUSY central, consider sample dictionary for nevents!"
                        nevents = sample[ss]['nevents']
                        print "n gen events: ", nevents

                    if nevents!=tree_entries:
                        print "There are some selections applied! Aborting...."
                        exit()
                    b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
                    n_pass      = filename.Get("n_pass").GetBinContent(1)
                    n_odd       = filename.Get("n_odd").GetBinContent(1)
                    filename.Close()
                    xs = sample[ss]['xsec'] * sample[ss]['kfactor']
                    if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                        print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                        xs = 1.
                    t_w = LUMI * xs / nevents
                    if(b_skipTrain>0):
                        if(n_odd>0):
                            t_w *= float(n_pass/n_odd)
                else:
                    t_w = 0

            tree_weight_dict[ss] = t_w

    print tree_weight_dict
    '''
    chunk_size = 1000000
    #chunk_size = 100
    for b in sign:
        maps[b]['merged'] = {}
        maps[b]['resolved'] = {}
        for i, ss in enumerate(samples[b]['files']):
            print "Uproot iterating over ", NTUPLEDIR +ss+ '.root', " . . . "
            filename = NTUPLEDIR + ss + '.root'
            gen = uproot.iterate(filename,"tree",list_of_variables,entrysteps=chunk_size)
            for arrays in gen:
                
                n_events = -1
            
                #Take one event every two: to use one set as validation
                step = 2 if skip_even else 1

                #Easiest way: count how many tagged jets are associated to each LLP
                #include also an event-level flag for pre-selections
                preselected = arrays["isPreselected"][0:n_events:step]

                #print "N. events considered for creating maps..."
                #print preselected.shape
                #exit()

                ###debug, something wrong in the preselected flag...
                ##print preselected
                ##print arrays["pt"][0:n_events:step][preselected]
                ##print arrays["EventNumber"][0:n_events:step][preselected]
                ##exit()

                #number of tagged jets per LLP
                nTagJets = arrays["nTagJets_0p996"][0:n_events:step]
                nTagJets_to_LLP0 = arrays["nTagJets_to_LLP0"][0:n_events:step]
                nTagJets_to_LLP1 = arrays["nTagJets_to_LLP1"][0:n_events:step]
                nJets_to_LLP0 = arrays["nJets_to_LLP0"][0:n_events:step]
                nJets_to_LLP1 = arrays["nJets_to_LLP1"][0:n_events:step]

                #merged vs resolved
                genLLP0_merged = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][0:n_events:step] )))[0:1] < 0.4)
                genLLP1_merged = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][0:n_events:step] )))[1:2] < 0.4)

                genLLP0_resolved = np.logical_not(genLLP0_merged)
                genLLP1_resolved = np.logical_not(genLLP1_merged)

                genLLP0_r = np.transpose(np.stack(( arrays["GenLLPs.travelRadius"][0:n_events:step]/100. )))[0:1]
                genLLP1_r = np.transpose(np.stack(( arrays["GenLLPs.travelRadius"][0:n_events:step]/100. )))[1:2]

                genLLP0_z = np.absolute( np.transpose(np.stack(( arrays["GenLLPs.travelZ"][0:n_events:step]/100. )))[0:1] )
                genLLP1_z = np.absolute( np.transpose(np.stack(( arrays["GenLLPs.travelZ"][0:n_events:step]/100. )))[1:2] )

                genh0_pt = np.transpose(np.stack(( arrays["GenHiggs.pt"][0:n_events:step] )))[0:1]
                genh1_pt = np.transpose(np.stack(( arrays["GenHiggs.pt"][0:n_events:step] )))[1:2]

                genh0_eta = np.absolute( np.transpose(np.stack(( arrays["GenHiggs.eta"][0:n_events:step] )))[0:1] )
                genh1_eta = np.absolute( np.transpose(np.stack(( arrays["GenHiggs.eta"][0:n_events:step] )))[1:2] )

                genb0_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][0:n_events:step] )))[0:1]
                genb1_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][0:n_events:step] )))[1:2]
                genb2_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][0:n_events:step] )))[2:3]
                genb3_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][0:n_events:step] )))[3:]

                genb0_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][0:n_events:step] )))[0:1] )
                genb1_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][0:n_events:step] )))[1:2] )
                genb2_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][0:n_events:step] )))[2:3] )
                genb3_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][0:n_events:step] )))[3:] )


                #acceptance masks
                #merged
                acc_LLP0_merged_per_part = np.logical_and( np.logical_and(genh0_pt>30.,abs(genh0_eta)<1.) , genLLP0_merged)
                acc_LLP1_merged_per_part = np.logical_and( np.logical_and(genh1_pt>30.,abs(genh1_eta)<1.) , genLLP1_merged)

                acc_LLP0_merged = acc_LLP0_merged_per_part.reshape(acc_LLP0_merged_per_part.shape[1])
                acc_LLP1_merged = acc_LLP1_merged_per_part.reshape(acc_LLP1_merged_per_part.shape[1])


                acc_LLP0_resolved_per_part_or = np.logical_and( np.logical_or( np.logical_and(genb0_pt>30.,abs(genb0_eta)<1.), np.logical_and(genb1_pt>30.,abs(genb1_eta)<1.)), genLLP0_resolved)
                acc_LLP1_resolved_per_part_or = np.logical_and( np.logical_or( np.logical_and(genb2_pt>30.,abs(genb2_eta)<1.), np.logical_and(genb3_pt>30.,abs(genb3_eta)<1.)), genLLP1_resolved)

                #Only q1 in acceptance
                acc_LLP0_resolved_per_part_q1 = np.logical_and( 
                    np.logical_and( 
                        np.logical_and(genb0_pt>30.,abs(genb0_eta)<1.), #q1 in acceptance
                        np.logical_not(  np.logical_and(genb1_pt>30.,abs(genb1_eta)<1.) )  #q2 not in acceptance!
                    ), 
                    genLLP0_resolved)
                #Only q2 in acceptance
                acc_LLP0_resolved_per_part_q2 = np.logical_and( 
                    np.logical_and( 
                        np.logical_not( np.logical_and(genb0_pt>30.,abs(genb0_eta)<1.) ), #q1 not in acceptance
                        np.logical_and(genb1_pt>30.,abs(genb1_eta)<1.)  #q2 in acceptance
                    ), 
                    genLLP0_resolved)
                #Both q1 and q2 in acceptance
                acc_LLP0_resolved_per_part_q1q2 = np.logical_and( 
                    np.logical_and( 
                        np.logical_and(genb0_pt>30.,abs(genb0_eta)<1.), #q1 in acceptance
                        np.logical_and(genb1_pt>30.,abs(genb1_eta)<1.)  #q2 in acceptance
                    ), 
                    genLLP0_resolved)

                #Only q1 in acceptance
                acc_LLP1_resolved_per_part_q1 = np.logical_and( 
                    np.logical_and( 
                        np.logical_and(genb2_pt>30.,abs(genb2_eta)<1.), #q1 in acceptance
                        np.logical_not( np.logical_and(genb3_pt>30.,abs(genb3_eta)<1.) ) #q2 not in acceptance
                    ), 
                    genLLP1_resolved)

                #Only q2 in acceptance
                acc_LLP1_resolved_per_part_q2 = np.logical_and( 
                    np.logical_and( 
                        np.logical_not( np.logical_and(genb2_pt>30.,abs(genb2_eta)<1.) ), #q1 not in acceptance
                        np.logical_and(genb3_pt>30.,abs(genb3_eta)<1.) #q2 in acceptance
                    ), 
                    genLLP1_resolved)
                #Both q1 and q2 in acceptance
                acc_LLP1_resolved_per_part_q1q2 = np.logical_and( 
                    np.logical_and( 
                        np.logical_and(genb2_pt>30.,abs(genb2_eta)<1.), #q1 in acceptance
                        np.logical_and(genb3_pt>30.,abs(genb3_eta)<1.)  #q2 in acceptance
                    ), 
                    genLLP1_resolved)

                acc_LLP0_resolved_or = acc_LLP0_resolved_per_part_or.reshape(acc_LLP0_resolved_per_part_or.shape[1])
                acc_LLP1_resolved_or = acc_LLP1_resolved_per_part_or.reshape(acc_LLP1_resolved_per_part_or.shape[1])

                acc_LLP0_resolved_q1 = acc_LLP0_resolved_per_part_q1.reshape(acc_LLP0_resolved_per_part_q1.shape[1])
                acc_LLP0_resolved_q2 = acc_LLP0_resolved_per_part_q2.reshape(acc_LLP0_resolved_per_part_q2.shape[1])
                acc_LLP0_resolved_q1q2 = acc_LLP0_resolved_per_part_q1q2.reshape(acc_LLP0_resolved_per_part_q1q2.shape[1])

                acc_LLP1_resolved_q1 = acc_LLP1_resolved_per_part_q1.reshape(acc_LLP1_resolved_per_part_q1.shape[1])
                acc_LLP1_resolved_q2 = acc_LLP1_resolved_per_part_q2.reshape(acc_LLP1_resolved_per_part_q2.shape[1])
                acc_LLP1_resolved_q1q2 = acc_LLP1_resolved_per_part_q1q2.reshape(acc_LLP1_resolved_per_part_q1q2.shape[1])

                #zero_acceptance_merged = np.logical_and(acc_LLP0_merged==False, acc_LLP1_merged==False)
                #one_acceptance_merged  = np.logical_xor(acc_LLP0_merged==True, acc_LLP1_merged==True)
                #two_acceptance_merged  = np.logical_and(acc_LLP0_merged==True, acc_LLP1_merged==True)

                #zero_acceptance_resolved = np.logical_and(acc_LLP0_resolved_or==False, acc_LLP1_resolved_or==False)
                #one_acceptance_resolved  = np.logical_xor(acc_LLP0_resolved_or==True, acc_LLP1_resolved_or==True)
                #two_acceptance_resolved  = np.logical_and(acc_LLP0_resolved_or==True, acc_LLP1_resolved_or==True)

                #one_resolved_one_merged  = np.logical_and(one_acceptance_resolved==True, one_acceptance_merged==True)

                #Sort and split are used to understand a decent binning
                #merged case: parametrize vs gen higgs, we expect 1 jet per higgs
                sorted_genLLP0_r_merged = np.sort(genLLP0_r[acc_LLP0_merged_per_part])
                sorted_genLLP1_r_merged = np.sort(genLLP1_r[acc_LLP1_merged_per_part])

                sorted_genLLP0_z_merged = np.sort(genLLP0_z[acc_LLP0_merged_per_part])
                sorted_genLLP1_z_merged = np.sort(genLLP1_z[acc_LLP1_merged_per_part])

                sorted_genLLP0_pt_merged = np.sort(genh0_pt[acc_LLP0_merged_per_part])
                sorted_genLLP1_pt_merged = np.sort(genh1_pt[acc_LLP1_merged_per_part])

                sorted_genLLP0_eta_merged = np.sort(genh0_eta[acc_LLP0_merged_per_part])
                sorted_genLLP1_eta_merged = np.sort(genh1_eta[acc_LLP1_merged_per_part])

                #resolved case: parametrize vs gen b quark, we expect 1 jet per b-quark
                #binning based on or seems correct, more statistics
                sorted_genLLP0_r_resolved = np.sort(genLLP0_r[acc_LLP0_resolved_per_part_or])
                sorted_genLLP1_r_resolved = np.sort(genLLP1_r[acc_LLP1_resolved_per_part_or])

                sorted_genLLP0_z_resolved = np.sort(genLLP0_z[acc_LLP0_resolved_per_part_or])
                sorted_genLLP1_z_resolved = np.sort(genLLP1_z[acc_LLP1_resolved_per_part_or])

                sorted_genLLP0_pt1_resolved = np.sort(genb0_pt[acc_LLP0_resolved_per_part_or])
                sorted_genLLP1_pt1_resolved = np.sort(genb2_pt[acc_LLP1_resolved_per_part_or])

                sorted_genLLP0_pt2_resolved = np.sort(genb1_pt[acc_LLP0_resolved_per_part_or])
                sorted_genLLP1_pt2_resolved = np.sort(genb3_pt[acc_LLP1_resolved_per_part_or])

                sorted_genLLP0_eta1_resolved = np.sort(genb0_eta[acc_LLP0_resolved_per_part_or])
                sorted_genLLP1_eta1_resolved = np.sort(genb2_eta[acc_LLP1_resolved_per_part_or])

                sorted_genLLP0_eta2_resolved = np.sort(genb1_eta[acc_LLP0_resolved_per_part_or])
                sorted_genLLP1_eta2_resolved = np.sort(genb3_eta[acc_LLP1_resolved_per_part_or])

                #the binning highly affects the maps prediction accuracy
                #must do a dynamic range splitting
                
                '''
                n_splits = 4#2
                #print "split sorted_genLLP0_r_merged"
                
                #print np.array_split(sorted_genLLP0_r_merged,n_splits)[0][0]
                #print np.array_split(sorted_genLLP0_r_merged,n_splits)[0].shape
                #print np.array_split(sorted_genLLP0_r_merged,n_splits)[1][0]
                #print np.array_split(sorted_genLLP0_r_merged,n_splits)[1].shape
                #print np.array_split(sorted_genLLP0_r_merged,n_splits)[2][0]
                #print np.array_split(sorted_genLLP0_r_merged,n_splits)[2].shape
                #print np.array_split(sorted_genLLP0_r_merged,n_splits)[3][0]
                #print np.array_split(sorted_genLLP0_r_merged,n_splits)[3].shape

                print "split sorted_genLLP0_z_merged"
                print np.array_split(sorted_genLLP0_z_merged,n_splits)[0]
                print np.array_split(sorted_genLLP0_z_merged,n_splits)[1]
                print np.array_split(sorted_genLLP0_z_merged,n_splits)[2]
                print np.array_split(sorted_genLLP0_z_merged,n_splits)[3]

                print "split sorted_genLLP0_pt_merged"
                print np.array_split(sorted_genLLP0_pt_merged,n_splits)[0]
                print np.array_split(sorted_genLLP0_pt_merged,n_splits)[0].shape
                print np.array_split(sorted_genLLP0_pt_merged,n_splits)[1]
                print np.array_split(sorted_genLLP0_pt_merged,n_splits)[1].shape
                #print np.array_split(sorted_genLLP0_pt_merged,n_splits)[2]
                #print np.array_split(sorted_genLLP0_pt_merged,n_splits)[2].shape
                #print np.array_split(sorted_genLLP0_pt_merged,n_splits)[3]
                #print np.array_split(sorted_genLLP0_pt_merged,n_splits)[3].shape

                
                print "split sorted_genLLP0_eta_merged"
                print np.array_split(sorted_genLLP0_eta_merged,n_splits)[0]
                print np.array_split(sorted_genLLP0_eta_merged,n_splits)[0].shape
                print np.array_split(sorted_genLLP0_eta_merged,n_splits)[1]
                print np.array_split(sorted_genLLP0_eta_merged,n_splits)[1].shape
                #print np.array_split(sorted_genLLP0_eta_merged,n_splits)[2]
                #print np.array_split(sorted_genLLP0_eta_merged,n_splits)[2].shape
                #print np.array_split(sorted_genLLP0_eta_merged,n_splits)[3]
                #print np.array_split(sorted_genLLP0_eta_merged,n_splits)[3].shape

                print "\n"
                print "split sorted_genLLP0_r_resolved"
                print np.array_split(sorted_genLLP0_r_resolved,n_splits)[0]
                print np.array_split(sorted_genLLP0_r_resolved,n_splits)[0].shape
                print np.array_split(sorted_genLLP0_r_resolved,n_splits)[1]
                print np.array_split(sorted_genLLP0_r_resolved,n_splits)[1].shape
                #print np.array_split(sorted_genLLP0_r_resolved,n_splits)[2]
                #print np.array_split(sorted_genLLP0_r_resolved,n_splits)[2].shape
                #print np.array_split(sorted_genLLP0_r_resolved,n_splits)[3]
                #print np.array_split(sorted_genLLP0_r_resolved,n_splits)[3].shape

                print "\n"
                print "split sorted_genLLP0_z_resolved"
                print np.array_split(sorted_genLLP0_z_resolved,n_splits)[0]
                print np.array_split(sorted_genLLP0_z_resolved,n_splits)[1]
                #print np.array_split(sorted_genLLP0_z_resolved,n_splits)[2]
                #print np.array_split(sorted_genLLP0_z_resolved,n_splits)[3]


                print "\n"
                print "split sorted_genLLP0_pt1_resolved"
                print np.array_split(sorted_genLLP0_pt1_resolved,4)[0]
                print np.array_split(sorted_genLLP0_pt1_resolved,4)[0].shape
                print np.array_split(sorted_genLLP0_pt1_resolved,4)[1]
                print np.array_split(sorted_genLLP0_pt1_resolved,4)[1].shape
                print np.array_split(sorted_genLLP0_pt1_resolved,4)[2]
                print np.array_split(sorted_genLLP0_pt1_resolved,4)[2].shape
                print np.array_split(sorted_genLLP0_pt1_resolved,4)[3]
                print np.array_split(sorted_genLLP0_pt1_resolved,4)[3].shape

                print "\n"
                print "split sorted_genLLP0_pt2_resolved"
                print np.array_split(sorted_genLLP0_pt2_resolved,4)[0]
                print np.array_split(sorted_genLLP0_pt2_resolved,4)[0].shape
                print np.array_split(sorted_genLLP0_pt2_resolved,4)[1]
                print np.array_split(sorted_genLLP0_pt2_resolved,4)[1].shape
                print np.array_split(sorted_genLLP0_pt2_resolved,4)[2]
                print np.array_split(sorted_genLLP0_pt2_resolved,4)[2].shape
                print np.array_split(sorted_genLLP0_pt2_resolved,4)[3]
                print np.array_split(sorted_genLLP0_pt2_resolved,4)[3].shape
                
                print "\n"
                print "split sorted_genLLP0_eta1_resolved"
                print np.array_split(sorted_genLLP0_eta1_resolved,n_splits)[0]
                print np.array_split(sorted_genLLP0_eta1_resolved,n_splits)[0].shape
                print np.array_split(sorted_genLLP0_eta1_resolved,n_splits)[1]
                print np.array_split(sorted_genLLP0_eta1_resolved,n_splits)[1].shape
                #print np.array_split(sorted_genLLP0_eta1_resolved,n_splits)[2]
                #print np.array_split(sorted_genLLP0_eta1_resolved,n_splits)[2].shape
                #print np.array_split(sorted_genLLP0_eta1_resolved,n_splits)[3]
                #print np.array_split(sorted_genLLP0_eta1_resolved,n_splits)[3].shape

                print "\n"
                print "split sorted_genLLP0_eta2_resolved"
                print np.array_split(sorted_genLLP0_eta2_resolved,n_splits)[0]
                print np.array_split(sorted_genLLP0_eta2_resolved,n_splits)[0].shape
                print np.array_split(sorted_genLLP0_eta2_resolved,n_splits)[1]
                print np.array_split(sorted_genLLP0_eta2_resolved,n_splits)[1].shape
                #print np.array_split(sorted_genLLP0_eta2_resolved,n_splits)[2]
                #print np.array_split(sorted_genLLP0_eta2_resolved,n_splits)[2].shape
                #print np.array_split(sorted_genLLP0_eta2_resolved,n_splits)[3]
                #print np.array_split(sorted_genLLP0_eta2_resolved,n_splits)[3].shape
                '''


                print "can probably do a better splitting if merging together LLP0 and LLP1 distributions, more stat"

                n_splits = 4#2
                bins_r_resolved = np.array(  [0., np.array_split(sorted_genLLP0_r_resolved,n_splits)[1][0], np.array_split(sorted_genLLP0_r_resolved,n_splits)[2][0], np.array_split(sorted_genLLP0_r_resolved,n_splits)[3][0], 500.])
                bins_z_resolved = np.array(  [0., np.array_split(sorted_genLLP0_z_resolved,n_splits)[1][0], np.array_split(sorted_genLLP0_z_resolved,n_splits)[2][0], np.array_split(sorted_genLLP0_z_resolved,n_splits)[3][0], 1000.])
                bins_pt_resolved = np.array(  [0., np.array_split(sorted_genLLP0_pt1_resolved,n_splits)[1][0], np.array_split(sorted_genLLP0_pt1_resolved,n_splits)[2][0], np.array_split(sorted_genLLP0_pt1_resolved,n_splits)[3][0], 5000.])
                bins_eta_resolved = np.array([0., np.array_split(sorted_genLLP0_eta1_resolved,n_splits)[1][0], np.array_split(sorted_genLLP0_eta1_resolved,n_splits)[2][0], np.array_split(sorted_genLLP0_eta1_resolved,n_splits)[3][0], 10.])
                #bins_eta_resolved = np.array([0., 0.4, 10.])
                #bins_eta_resolved = np.array([0., 0.4, 5.])
                #bins_pt_resolved = np.array( [0.,  300.,  500., 800.,  5000.])
                #bins_z_resolved = np.array(  [0., 0.8, 100.])
                #bins_r_resolved = np.array(  [0.,  0.2,   100.])

                #First bin guess
                #4 splits
                #bins_eta_merged = np.array([-1., -0.4,  0.,   0.4,    1.])
                #bins_pt_merged = np.array( [0.,  765.,  900., 1100.,  5000.])
                #bins_z_merged = np.array(  [-5., -0.15, 0.,   0.15,   5.])
                #bins_r_merged = np.array(  [0.,  0.1,   0.3,  0.6,    10.])

                n_splits = 4#2
                bins_r_merged = np.array(  [0., np.array_split(sorted_genLLP0_r_merged,n_splits)[1][0], np.array_split(sorted_genLLP0_r_merged,n_splits)[2][0], np.array_split(sorted_genLLP0_r_merged,n_splits)[3][0], 500.])
                bins_z_merged = np.array(  [0., np.array_split(sorted_genLLP0_z_merged,n_splits)[1][0], np.array_split(sorted_genLLP0_z_merged,n_splits)[2][0], np.array_split(sorted_genLLP0_z_merged,n_splits)[3][0], 1000.])
                bins_pt_merged = np.array(  [0., np.array_split(sorted_genLLP0_pt_merged,n_splits)[1][0], np.array_split(sorted_genLLP0_pt_merged,n_splits)[2][0], np.array_split(sorted_genLLP0_pt_merged,n_splits)[3][0], 5000.])
                bins_eta_merged = np.array(  [0., np.array_split(sorted_genLLP0_eta_merged,n_splits)[1][0], np.array_split(sorted_genLLP0_eta_merged,n_splits)[2][0], np.array_split(sorted_genLLP0_eta_merged,n_splits)[3][0], 10.])
                #bins_eta_merged = np.array([0., 0.4, 10.])

                #needed reshaping
                nTagJets_to_LLP0  = nTagJets_to_LLP0.reshape(acc_LLP0_merged_per_part.shape)
                nTagJets_to_LLP1  = nTagJets_to_LLP1.reshape(acc_LLP1_merged_per_part.shape)
                preselected       = preselected.reshape(acc_LLP0_merged_per_part.shape)
                
                '''
                print "all:"
                print "merged"
                print nTagJets[zero_acceptance_merged]
                print nTagJets[one_acceptance_merged]
                print nTagJets[two_acceptance_merged]
                print "resolved"
                print nTagJets[zero_acceptance_resolved]
                print nTagJets[one_acceptance_resolved]
                print nTagJets[two_acceptance_resolved]
                print "one and one"
                print nTagJets[one_resolved_one_merged]

                print "matched to LLP0"
                print "merged"
                print acc_LLP0_merged
                print nTagJets_to_LLP0[acc_LLP0_merged]
                print "resolved"
                print acc_LLP0_resolved_or
                print nTagJets_to_LLP0[acc_LLP0_resolved_or]

                #need to create num/den based on topology and n of tags (0,1,2 or more)
                #Example of a correct 4-d histogram
                #H, edges = np.histogramdd(np.transpose([pt,eta,phi,n]), bins = (pt_bins,eta_bins,phi_bins,n_bins))
                '''

                #merged:
                m_denLLP0_r = genLLP0_r[acc_LLP0_merged_per_part]
                m_denLLP1_r = genLLP1_r[acc_LLP1_merged_per_part]
                m_denLLP0_z = genLLP0_z[acc_LLP0_merged_per_part]
                m_denLLP1_z = genLLP1_z[acc_LLP1_merged_per_part]
                m_denLLP0_pt = genh0_pt[acc_LLP0_merged_per_part]
                m_denLLP1_pt = genh1_pt[acc_LLP1_merged_per_part]
                m_denLLP0_eta = genh0_eta[acc_LLP0_merged_per_part]
                m_denLLP1_eta = genh1_eta[acc_LLP1_merged_per_part]
                #merge LLP0 and LLP1
                m_den_r = np.concatenate((m_denLLP0_r,m_denLLP1_r))
                m_den_z = np.concatenate((m_denLLP0_z,m_denLLP1_z))
                m_den_pt = np.concatenate((m_denLLP0_pt,m_denLLP1_pt))
                m_den_eta = np.concatenate((m_denLLP0_eta,m_denLLP1_eta))


                #w/o preselections
                #At least one tag
                m_numLLP0_r_T = genLLP0_r[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_merged)]
                m_numLLP1_r_T = genLLP1_r[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_merged)]
                m_numLLP0_z_T = genLLP0_z[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_merged)]
                m_numLLP1_z_T = genLLP1_z[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_merged)]
                m_numLLP0_pt_T = genh0_pt[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_merged)]
                m_numLLP1_pt_T = genh1_pt[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_merged)]
                m_numLLP0_eta_T = genh0_eta[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_merged)]
                m_numLLP1_eta_T = genh1_eta[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_merged)]
                #merge LLP0 and LLP1
                m_num_r_T = np.concatenate((m_numLLP0_r_T,m_numLLP1_r_T))
                m_num_z_T = np.concatenate((m_numLLP0_z_T,m_numLLP1_z_T))
                m_num_pt_T = np.concatenate((m_numLLP0_pt_T,m_numLLP1_pt_T))
                m_num_eta_T = np.concatenate((m_numLLP0_eta_T,m_numLLP1_eta_T))

                #Exactly 1T
                m_numLLP0_r_1T = genLLP0_r[np.logical_and(nTagJets_to_LLP0==1,acc_LLP0_merged)]
                m_numLLP1_r_1T = genLLP1_r[np.logical_and(nTagJets_to_LLP1==1,acc_LLP1_merged)]
                m_numLLP0_z_1T = genLLP0_z[np.logical_and(nTagJets_to_LLP0==1,acc_LLP0_merged)]
                m_numLLP1_z_1T = genLLP1_z[np.logical_and(nTagJets_to_LLP1==1,acc_LLP1_merged)]
                m_numLLP0_pt_1T = genh0_pt[np.logical_and(nTagJets_to_LLP0==1,acc_LLP0_merged)]
                m_numLLP1_pt_1T = genh1_pt[np.logical_and(nTagJets_to_LLP1==1,acc_LLP1_merged)]
                m_numLLP0_eta_1T = genh0_eta[np.logical_and(nTagJets_to_LLP0==1,acc_LLP0_merged)]
                m_numLLP1_eta_1T = genh1_eta[np.logical_and(nTagJets_to_LLP1==1,acc_LLP1_merged)]

                #2T or more
                m_numLLP0_r_2T = genLLP0_r[np.logical_and(nTagJets_to_LLP0>1,acc_LLP0_merged)]
                m_numLLP1_r_2T = genLLP1_r[np.logical_and(nTagJets_to_LLP1>1,acc_LLP1_merged)]
                m_numLLP0_z_2T = genLLP0_z[np.logical_and(nTagJets_to_LLP0>1,acc_LLP0_merged)]
                m_numLLP1_z_2T = genLLP1_z[np.logical_and(nTagJets_to_LLP1>1,acc_LLP1_merged)]
                m_numLLP0_pt_2T = genh0_pt[np.logical_and(nTagJets_to_LLP0>1,acc_LLP0_merged)]
                m_numLLP1_pt_2T = genh1_pt[np.logical_and(nTagJets_to_LLP1>1,acc_LLP1_merged)]
                m_numLLP0_eta_2T = genh0_eta[np.logical_and(nTagJets_to_LLP0>1,acc_LLP0_merged)]
                m_numLLP1_eta_2T = genh1_eta[np.logical_and(nTagJets_to_LLP1>1,acc_LLP1_merged)]

                #w presel
                #At least one tag
                m_numLLP0_r_T_presel = genLLP0_r[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_merged) )]
                m_numLLP1_r_T_presel = genLLP1_r[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_merged) )]
                m_numLLP0_z_T_presel = genLLP0_z[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_merged) )]
                m_numLLP1_z_T_presel = genLLP1_z[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_merged) )]
                m_numLLP0_pt_T_presel = genh0_pt[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_merged) )]
                m_numLLP1_pt_T_presel = genh1_pt[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_merged) )]
                m_numLLP0_eta_T_presel = genh0_eta[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_merged) )]
                m_numLLP1_eta_T_presel = genh1_eta[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_merged) )]
                #merge LLP0 and LLP1
                m_num_r_T_presel = np.concatenate((m_numLLP0_r_T_presel,m_numLLP1_r_T_presel))
                m_num_z_T_presel = np.concatenate((m_numLLP0_z_T_presel,m_numLLP1_z_T_presel))
                m_num_pt_T_presel = np.concatenate((m_numLLP0_pt_T_presel,m_numLLP1_pt_T_presel))
                m_num_eta_T_presel = np.concatenate((m_numLLP0_eta_T_presel,m_numLLP1_eta_T_presel))

                #Exactly 1T
                m_numLLP0_r_1T_presel = genLLP0_r[np.logical_and(nTagJets_to_LLP0==1, np.logical_and(preselected,acc_LLP0_merged) )]
                m_numLLP1_r_1T_presel = genLLP1_r[np.logical_and(nTagJets_to_LLP1==1, np.logical_and(preselected,acc_LLP1_merged) )]
                m_numLLP0_z_1T_presel = genLLP0_z[np.logical_and(nTagJets_to_LLP0==1, np.logical_and(preselected,acc_LLP0_merged) )]
                m_numLLP1_z_1T_presel = genLLP1_z[np.logical_and(nTagJets_to_LLP1==1, np.logical_and(preselected,acc_LLP1_merged) )]
                m_numLLP0_pt_1T_presel = genh0_pt[np.logical_and(nTagJets_to_LLP0==1, np.logical_and(preselected,acc_LLP0_merged) )]
                m_numLLP1_pt_1T_presel = genh1_pt[np.logical_and(nTagJets_to_LLP1==1, np.logical_and(preselected,acc_LLP1_merged) )]
                m_numLLP0_eta_1T_presel = genh0_eta[np.logical_and(nTagJets_to_LLP0==1, np.logical_and(preselected,acc_LLP0_merged) )]
                m_numLLP1_eta_1T_presel = genh1_eta[np.logical_and(nTagJets_to_LLP1==1, np.logical_and(preselected,acc_LLP1_merged) )]

                #2T or more
                m_numLLP0_r_2T_presel = genLLP0_r[np.logical_and(nTagJets_to_LLP0>1, np.logical_and(preselected,acc_LLP0_merged) )]
                m_numLLP1_r_2T_presel = genLLP1_r[np.logical_and(nTagJets_to_LLP1>1, np.logical_and(preselected,acc_LLP1_merged) )]
                m_numLLP0_z_2T_presel = genLLP0_z[np.logical_and(nTagJets_to_LLP0>1, np.logical_and(preselected,acc_LLP0_merged) )]
                m_numLLP1_z_2T_presel = genLLP1_z[np.logical_and(nTagJets_to_LLP1>1, np.logical_and(preselected,acc_LLP1_merged) )]
                m_numLLP0_pt_2T_presel = genh0_pt[np.logical_and(nTagJets_to_LLP0>1, np.logical_and(preselected,acc_LLP0_merged) )]
                m_numLLP1_pt_2T_presel = genh1_pt[np.logical_and(nTagJets_to_LLP1>1, np.logical_and(preselected,acc_LLP1_merged) )]
                m_numLLP0_eta_2T_presel = genh0_eta[np.logical_and(nTagJets_to_LLP0>1, np.logical_and(preselected,acc_LLP0_merged) )]
                m_numLLP1_eta_2T_presel = genh1_eta[np.logical_and(nTagJets_to_LLP1>1, np.logical_and(preselected,acc_LLP1_merged) )]


                #num: like den but with having tagged jets associated
                m_den_LLP0, m_den_LLP0_edges = np.histogramdd(np.transpose([m_denLLP0_r,m_denLLP0_z,m_denLLP0_pt,m_denLLP0_eta]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))
                m_den_LLP1, m_den_LLP1_edges = np.histogramdd(np.transpose([m_denLLP1_r,m_denLLP1_z,m_denLLP1_pt,m_denLLP1_eta]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))
                m_den, m_den_edges = np.histogramdd(np.transpose([m_den_r,m_den_z,m_den_pt,m_den_eta]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))


                #At least one tag
                m_num_LLP0_T, m_num_LLP0_edges_T = np.histogramdd(np.transpose([m_numLLP0_r_T,m_numLLP0_z_T,m_numLLP0_pt_T,m_numLLP0_eta_T]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))
                m_num_LLP0_T_presel, m_num_LLP0_edges_T_presel = np.histogramdd(np.transpose([m_numLLP0_r_T_presel,m_numLLP0_z_T_presel,m_numLLP0_pt_T_presel,m_numLLP0_eta_T_presel]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))
                m_eff_LLP0_T = np.divide(m_num_LLP0_T,m_den_LLP0,where=(m_den_LLP0 != 0))
                m_eff_LLP0_T_presel = np.divide(m_num_LLP0_T_presel,m_den_LLP0,where=(m_den_LLP0 != 0))
                #print "merged eff LLP0 T:"
                #print m_eff_LLP0_T
                print "merged eff LLP0 T presel:"
                print m_eff_LLP0_T_presel


                m_num_LLP1_T, m_num_LLP1_edges_T = np.histogramdd(np.transpose([m_numLLP1_r_T,m_numLLP1_z_T,m_numLLP1_pt_T,m_numLLP1_eta_T]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))
                m_num_LLP1_T_presel, m_num_LLP1_edges_T_presel = np.histogramdd(np.transpose([m_numLLP1_r_T_presel,m_numLLP1_z_T_presel,m_numLLP1_pt_T_presel,m_numLLP1_eta_T_presel]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))
                m_eff_LLP1_T = np.divide(m_num_LLP1_T,m_den_LLP1,where=(m_den_LLP1 != 0))
                m_eff_LLP1_T_presel = np.divide(m_num_LLP1_T_presel,m_den_LLP1,where=(m_den_LLP1 != 0))
                #print "merged eff LLP1 T:"
                #print m_eff_LLP1_T
                print "merged eff LLP1 T presel:"
                print m_eff_LLP1_T_presel


                m_num_T, m_num_edges_T = np.histogramdd(np.transpose([m_num_r_T,m_num_z_T,m_num_pt_T,m_num_eta_T]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))
                m_num_T_presel, m_num_edges_T_presel = np.histogramdd(np.transpose([m_num_r_T_presel,m_num_z_T_presel,m_num_pt_T_presel,m_num_eta_T_presel]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))
                m_eff_T = np.divide(m_num_T,m_den,where=(m_den != 0))
                m_eff_T_presel = np.divide(m_num_T_presel,m_den,where=(m_den != 0))
                #print "all merged eff T:"
                #print m_eff_T
                print "all merged eff T presel:"
                print m_eff_T_presel


                '''
                #Exactly 1T
                m_num_LLP0_1T, m_num_LLP0_edges_1T = np.histogramdd(np.transpose([m_numLLP0_r_1T,m_numLLP0_z_1T,m_numLLP0_pt_1T,m_numLLP0_eta_1T]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))
                m_num_LLP0_1T_presel, m_num_LLP0_edges_1T_presel = np.histogramdd(np.transpose([m_numLLP0_r_1T_presel,m_numLLP0_z_1T_presel,m_numLLP0_pt_1T_presel,m_numLLP0_eta_1T_presel]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))
                m_eff_LLP0_1T = np.divide(m_num_LLP0_1T,m_den_LLP0,where=(m_den_LLP0 != 0))
                m_eff_LLP0_1T_presel = np.divide(m_num_LLP0_1T_presel,m_den_LLP0,where=(m_den_LLP0 != 0))
                #print "merged eff LLP0 1T:"
                #print m_eff_LLP0_1T
                #print "merged eff LLP0 1T presel:"
                #print m_eff_LLP0_1T_presel

                #2T or more
                m_num_LLP0_2T, m_num_LLP0_edges_2T = np.histogramdd(np.transpose([m_numLLP0_r_2T,m_numLLP0_z_2T,m_numLLP0_pt_2T,m_numLLP0_eta_2T]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))
                m_num_LLP0_2T_presel, m_num_LLP0_edges_2T_presel = np.histogramdd(np.transpose([m_numLLP0_r_2T_presel,m_numLLP0_z_2T_presel,m_numLLP0_pt_2T_presel,m_numLLP0_eta_2T_presel]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged,bins_eta_merged))
                m_eff_LLP0_2T = np.divide(m_num_LLP0_2T,m_den_LLP0,where=(m_den_LLP0 != 0))
                m_eff_LLP0_2T_presel = np.divide(m_num_LLP0_2T_presel,m_den_LLP0,where=(m_den_LLP0 != 0))
                #print "merged eff LLP0 2T:"
                #print m_eff_LLP0_2T
                #print "merged eff LLP0 2T presel:"
                #print m_eff_LLP0_2T_presel
                '''

                # -----

                #resolved:

                #Systematic overprediction.
                r_denLLP0_r = genLLP0_r[acc_LLP0_resolved_per_part_or]
                r_denLLP1_r = genLLP1_r[acc_LLP1_resolved_per_part_or]
                r_denLLP0_z = genLLP0_z[acc_LLP0_resolved_per_part_or]
                r_denLLP1_z = genLLP1_z[acc_LLP1_resolved_per_part_or]
                r_denLLP0_pt1 = genb0_pt[acc_LLP0_resolved_per_part_or]
                r_denLLP1_pt1 = genb2_pt[acc_LLP1_resolved_per_part_or]
                r_denLLP0_pt2 = genb1_pt[acc_LLP0_resolved_per_part_or]
                r_denLLP1_pt2 = genb3_pt[acc_LLP1_resolved_per_part_or]
                r_denLLP0_eta1 = genb0_eta[acc_LLP0_resolved_per_part_or]
                r_denLLP1_eta1 = genb2_eta[acc_LLP1_resolved_per_part_or]
                r_denLLP0_eta2 = genb1_eta[acc_LLP0_resolved_per_part_or]
                r_denLLP1_eta2 = genb3_eta[acc_LLP1_resolved_per_part_or]


                #Try to split cases where there is only 1 q in acceptance or when there are 2
                r_denLLP0_r_q1   = genLLP0_r[acc_LLP0_resolved_per_part_q1]
                r_denLLP1_r_q1   = genLLP1_r[acc_LLP1_resolved_per_part_q1]
                r_denLLP0_z_q1   = genLLP0_z[acc_LLP0_resolved_per_part_q1]
                r_denLLP1_z_q1   = genLLP1_z[acc_LLP1_resolved_per_part_q1]
                r_denLLP0_pt_q1  = genb0_pt[acc_LLP0_resolved_per_part_q1]
                r_denLLP1_pt_q1  = genb2_pt[acc_LLP1_resolved_per_part_q1]
                r_denLLP0_eta_q1 = genb0_eta[acc_LLP0_resolved_per_part_q1]
                r_denLLP1_eta_q1 = genb2_eta[acc_LLP1_resolved_per_part_q1]

                r_denLLP0_r_q2   = genLLP0_r[acc_LLP0_resolved_per_part_q2]
                r_denLLP1_r_q2   = genLLP1_r[acc_LLP1_resolved_per_part_q2]
                r_denLLP0_z_q2   = genLLP0_z[acc_LLP0_resolved_per_part_q2]
                r_denLLP1_z_q2   = genLLP1_z[acc_LLP1_resolved_per_part_q2]
                r_denLLP0_pt_q2  = genb1_pt[acc_LLP0_resolved_per_part_q2]
                r_denLLP1_pt_q2  = genb3_pt[acc_LLP1_resolved_per_part_q2]
                r_denLLP0_eta_q2 = genb1_eta[acc_LLP0_resolved_per_part_q2]
                r_denLLP1_eta_q2 = genb3_eta[acc_LLP1_resolved_per_part_q2]

                r_denLLP0_r_q1q2    = genLLP0_r[acc_LLP0_resolved_per_part_q1q2]
                r_denLLP1_r_q1q2    = genLLP1_r[acc_LLP1_resolved_per_part_q1q2]
                r_denLLP0_z_q1q2    = genLLP0_z[acc_LLP0_resolved_per_part_q1q2]
                r_denLLP1_z_q1q2    = genLLP1_z[acc_LLP1_resolved_per_part_q1q2]
                r_denLLP0_pt1_q1q2  = genb0_pt[acc_LLP0_resolved_per_part_q1q2]
                r_denLLP1_pt1_q1q2  = genb2_pt[acc_LLP1_resolved_per_part_q1q2]
                r_denLLP0_pt2_q1q2  = genb1_pt[acc_LLP0_resolved_per_part_q1q2]
                r_denLLP1_pt2_q1q2  = genb3_pt[acc_LLP1_resolved_per_part_q1q2]
                r_denLLP0_eta1_q1q2 = genb0_eta[acc_LLP0_resolved_per_part_q1q2]
                r_denLLP1_eta1_q1q2 = genb2_eta[acc_LLP1_resolved_per_part_q1q2]
                r_denLLP0_eta2_q1q2 = genb1_eta[acc_LLP0_resolved_per_part_q1q2]
                r_denLLP1_eta2_q1q2 = genb3_eta[acc_LLP1_resolved_per_part_q1q2]



                #merge LLP0 and LLP1
                #later if needed
                #r_den_r = np.concatenate((r_denLLP0_r,r_denLLP1_r))
                #r_den_z = np.concatenate((r_denLLP0_z,r_denLLP1_z))
                #r_den_pt1 = np.concatenate((r_denLLP0_pt1,r_denLLP1_pt1))
                #r_den_pt2 = np.concatenate((r_denLLP0_pt2,r_denLLP1_pt2))
                #r_den_eta1 = np.concatenate((r_denLLP0_eta1,r_denLLP1_eta1))
                #r_den_eta2 = np.concatenate((r_denLLP0_eta2,r_denLLP1_eta2))

                #OR: gives systematic overprediction
                '''                
                #w/o preselections
                #At least one tag
                r_numLLP0_r_or_T = genLLP0_r[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_or)]
                r_numLLP1_r_or_T = genLLP1_r[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_or)]
                r_numLLP0_z_or_T = genLLP0_z[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_or)]
                r_numLLP1_z_or_T = genLLP1_z[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_or)]
                r_numLLP0_pt1_or_T = genb0_pt[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_or)]
                r_numLLP1_pt1_or_T = genb2_pt[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_or)]
                r_numLLP0_pt2_or_T = genb1_pt[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_or)]
                r_numLLP1_pt2_or_T = genb3_pt[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_or)]
                r_numLLP0_eta1_or_T = genb0_eta[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_or)]
                r_numLLP1_eta1_or_T = genb2_eta[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_or)]
                r_numLLP0_eta2_or_T = genb1_eta[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_or)]
                r_numLLP1_eta2_or_T = genb3_eta[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_or)]
                #merge LLP0 and LLP1
                #later if needed
                #r_num_r_T = np.concatenate((r_numLLP0_r_T,r_numLLP1_r_T))
                #r_num_z_T = np.concatenate((r_numLLP0_z_T,r_numLLP1_z_T))
                #r_num_pt1_T = np.concatenate((r_numLLP0_pt1_T,r_numLLP1_pt1_T))
                #r_num_pt2_T = np.concatenate((r_numLLP0_pt2_T,r_numLLP1_pt2_T))
                #r_num_eta1_T = np.concatenate((r_numLLP0_eta1_T,r_numLLP1_eta1_T))
                #r_num_eta2_T = np.concatenate((r_numLLP0_eta2_T,r_numLLP1_eta2_T))

                #w presel
                #At least one tag
                r_numLLP0_r_or_T_presel = genLLP0_r[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_resolved_or) )]
                r_numLLP1_r_or_T_presel = genLLP1_r[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_resolved_or) )]
                r_numLLP0_z_or_T_presel = genLLP0_z[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_resolved_or) )]
                r_numLLP1_z_or_T_presel = genLLP1_z[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_resolved_or) )]
                r_numLLP0_pt1_or_T_presel = genb0_pt[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_resolved_or) )]
                r_numLLP1_pt1_or_T_presel = genb2_pt[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_resolved_or) )]
                r_numLLP0_pt2_or_T_presel = genb1_pt[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_resolved_or) )]
                r_numLLP1_pt2_or_T_presel = genb3_pt[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_resolved_or) )]
                r_numLLP0_eta1_or_T_presel = genb0_eta[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_resolved_or) )]
                r_numLLP1_eta1_or_T_presel = genb2_eta[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_resolved_or) )]
                r_numLLP0_eta2_or_T_presel = genb1_eta[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_resolved_or) )]
                r_numLLP1_eta2_or_T_presel = genb3_eta[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_resolved_or) )]
                #merge LLP0 and LLP1
                #later if needed
                #r_num_r_T_presel = np.concatenate((r_numLLP0_r_T_presel,r_numLLP1_r_T_presel))
                #r_num_z_T_presel = np.concatenate((r_numLLP0_z_T_presel,r_numLLP1_z_T_presel))
                #r_num_pt1_T_presel = np.concatenate((r_numLLP0_pt1_T_presel,r_numLLP1_pt1_T_presel))
                #r_num_pt2_T_presel = np.concatenate((r_numLLP0_pt2_T_presel,r_numLLP1_pt2_T_presel))
                #r_num_eta1_T_presel = np.concatenate((r_numLLP0_eta1_T_presel,r_numLLP1_eta1_T_presel))
                #r_num_eta2_T_presel = np.concatenate((r_numLLP0_eta2_T_presel,r_numLLP1_eta2_T_presel))
                

                #now one map per b-quark
                r_den1_LLP0, r_den1_LLP0_edges = np.histogramdd(np.transpose([r_denLLP0_r,r_denLLP0_z,r_denLLP0_pt1,r_denLLP0_eta1]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_den1_LLP1, r_den1_LLP1_edges = np.histogramdd(np.transpose([r_denLLP1_r,r_denLLP1_z,r_denLLP1_pt1,r_denLLP1_eta1]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_den1, r_den1_edges = np.histogramdd(np.transpose([r_den_r,r_den_z,r_den_pt1,r_den_eta1]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))

                r_den2_LLP0, r_den2_LLP0_edges = np.histogramdd(np.transpose([r_denLLP0_r,r_denLLP0_z,r_denLLP0_pt2,r_denLLP0_eta2]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_den2_LLP1, r_den2_LLP1_edges = np.histogramdd(np.transpose([r_denLLP1_r,r_denLLP1_z,r_denLLP1_pt2,r_denLLP1_eta2]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_den2, r_den2_edges = np.histogramdd(np.transpose([r_den_r,r_den_z,r_den_pt2,r_den_eta2]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))

                #At least one tag
                #LLP0
                r_num1_LLP0_T, r_num1_LLP0_edges_T = np.histogramdd(np.transpose([r_numLLP0_r_T,r_numLLP0_z_T,r_numLLP0_pt1_T,r_numLLP0_eta1_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_num1_LLP0_T_presel, r_num1_LLP0_edges_T_presel = np.histogramdd(np.transpose([r_numLLP0_r_T_presel,r_numLLP0_z_T_presel,r_numLLP0_pt1_T_presel,r_numLLP0_eta1_T_presel]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_eff1_LLP0_T = np.divide(r_num1_LLP0_T,r_den1_LLP0,where=(r_den1_LLP0 != 0))
                r_eff1_LLP0_T_presel = np.divide(r_num1_LLP0_T_presel,r_den1_LLP0,where=(r_den1_LLP0 != 0))

                r_num2_LLP0_T, r_num2_LLP0_edges_T = np.histogramdd(np.transpose([r_numLLP0_r_T,r_numLLP0_z_T,r_numLLP0_pt2_T,r_numLLP0_eta2_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_num2_LLP0_T_presel, r_num2_LLP0_edges_T_presel = np.histogramdd(np.transpose([r_numLLP0_r_T_presel,r_numLLP0_z_T_presel,r_numLLP0_pt2_T_presel,r_numLLP0_eta2_T_presel]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_eff2_LLP0_T = np.divide(r_num2_LLP0_T,r_den2_LLP0,where=(r_den2_LLP0 != 0))
                r_eff2_LLP0_T_presel = np.divide(r_num2_LLP0_T_presel,r_den2_LLP0,where=(r_den2_LLP0 != 0))

                #LLP1
                r_num1_LLP1_T, r_num1_LLP1_edges_T = np.histogramdd(np.transpose([r_numLLP1_r_T,r_numLLP1_z_T,r_numLLP1_pt1_T,r_numLLP1_eta1_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_num1_LLP1_T_presel, r_num1_LLP1_edges_T_presel = np.histogramdd(np.transpose([r_numLLP1_r_T_presel,r_numLLP1_z_T_presel,r_numLLP1_pt1_T_presel,r_numLLP1_eta1_T_presel]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_eff1_LLP1_T = np.divide(r_num1_LLP1_T,r_den1_LLP1,where=(r_den1_LLP1 != 0))
                r_eff1_LLP1_T_presel = np.divide(r_num1_LLP1_T_presel,r_den1_LLP1,where=(r_den1_LLP1 != 0))

                r_num2_LLP1_T, r_num2_LLP1_edges_T = np.histogramdd(np.transpose([r_numLLP1_r_T,r_numLLP1_z_T,r_numLLP1_pt2_T,r_numLLP1_eta2_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_num2_LLP1_T_presel, r_num2_LLP1_edges_T_presel = np.histogramdd(np.transpose([r_numLLP1_r_T_presel,r_numLLP1_z_T_presel,r_numLLP1_pt2_T_presel,r_numLLP1_eta2_T_presel]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_eff2_LLP1_T = np.divide(r_num2_LLP1_T,r_den2_LLP1,where=(r_den2_LLP1 != 0))
                r_eff2_LLP1_T_presel = np.divide(r_num2_LLP1_T_presel,r_den2_LLP1,where=(r_den2_LLP1 != 0))

                #LLP0+LLP1
                #later if needed
                #r_num1_T, r_num1_edges_T = np.histogramdd(np.transpose([r_num_r_T,r_num_z_T,r_num_pt1_T,r_num_eta1_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                #r_num1_T_presel, r_num1_edges_T_presel = np.histogramdd(np.transpose([r_num_r_T_presel,r_num_z_T_presel,r_num_pt1_T_presel,r_num_eta1_T_presel]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                #r_eff1_T = np.divide(r_num1_T,r_den1,where=(r_den1 != 0))
                #r_eff1_T_presel = np.divide(r_num1_T_presel,r_den1,where=(r_den1 != 0))

                #r_num2_T, r_num2_edges_T = np.histogramdd(np.transpose([r_num_r_T,r_num_z_T,r_num_pt2_T,r_num_eta2_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                #r_num2_T_presel, r_num2_edges_T_presel = np.histogramdd(np.transpose([r_num_r_T_presel,r_num_z_T_presel,r_num_pt2_T_presel,r_num_eta2_T_presel]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                #r_eff2_T = np.divide(r_num2_T,r_den2,where=(r_den2 != 0))
                #r_eff2_T_presel = np.divide(r_num2_T_presel,r_den2,where=(r_den2 != 0))


                print "resolved eff1 LLP0 T:"
                print r_eff1_LLP0_T
                #print "resolved eff1 LLP0 T presel:"
                #print r_eff1_LLP0_T_presel

                print "resolved eff2 LLP0 T:"
                print r_eff2_LLP0_T
                #print "resolved eff2 LLP0 T presel:"
                #print r_eff2_LLP0_T_presel

                print "resolved eff1 LLP1 T:"
                print r_eff1_LLP1_T
                #print "resolved eff1 LLP1 T presel:"
                #print r_eff1_LLP1_T_presel

                print "resolved eff2 LLP1 T:"
                print r_eff2_LLP1_T
                #print "resolved eff2 LLP1 T presel:"
                #print r_eff2_LLP1_T_presel

                print "all resolved eff1 T:"
                print r_eff1_T
                #print "all resolved eff1 T presel:"
                #print r_eff1_T_presel

                print "all resolved eff2 T:"
                print r_eff2_T
                #print "all resolved eff2 T presel:"
                #print r_eff2_T_presel

                #Try to store these for the time being
                #And assess the variables needed
                '''
                
                #HERE
                #Split cases when there is 1q or 2q in acceptance

                #w/o preselections
                #At least one tag
                r_numLLP0_r_q1_T   = genLLP0_r[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_q1)]
                r_numLLP1_r_q1_T   = genLLP1_r[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_q1)]
                r_numLLP0_z_q1_T   = genLLP0_z[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_q1)]
                r_numLLP1_z_q1_T   = genLLP1_z[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_q1)]
                r_numLLP0_pt_q1_T  = genb0_pt[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_q1)]
                r_numLLP1_pt_q1_T  = genb2_pt[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_q1)]
                r_numLLP0_eta_q1_T = genb0_eta[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_q1)]
                r_numLLP1_eta_q1_T = genb2_eta[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_q1)]

                r_numLLP0_r_q2_T   = genLLP0_r[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_q2)]
                r_numLLP1_r_q2_T   = genLLP1_r[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_q2)]
                r_numLLP0_z_q2_T   = genLLP0_z[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_q2)]
                r_numLLP1_z_q2_T   = genLLP1_z[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_q2)]
                r_numLLP0_pt_q2_T  = genb1_pt[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_q2)]
                r_numLLP1_pt_q2_T  = genb3_pt[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_q2)]
                r_numLLP0_eta_q2_T = genb1_eta[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_resolved_q2)]
                r_numLLP1_eta_q2_T = genb3_eta[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_resolved_q2)]

                print "I am not convinced asking for at least 1 tag is correct for q1q2..."
                r_numLLP0_r_q1q2_1T   = genLLP0_r[np.logical_and(nTagJets_to_LLP0==1,acc_LLP0_resolved_q1q2)]
                r_numLLP1_r_q1q2_1T   = genLLP1_r[np.logical_and(nTagJets_to_LLP1==1,acc_LLP1_resolved_q1q2)]
                r_numLLP0_z_q1q2_1T   = genLLP0_z[np.logical_and(nTagJets_to_LLP0==1,acc_LLP0_resolved_q1q2)]
                r_numLLP1_z_q1q2_1T   = genLLP1_z[np.logical_and(nTagJets_to_LLP1==1,acc_LLP1_resolved_q1q2)]
                r_numLLP0_pt1_q1q2_1T  = genb0_pt[np.logical_and(nTagJets_to_LLP0==1,acc_LLP0_resolved_q1q2)]
                r_numLLP1_pt1_q1q2_1T  = genb2_pt[np.logical_and(nTagJets_to_LLP1==1,acc_LLP1_resolved_q1q2)]
                r_numLLP0_pt2_q1q2_1T  = genb1_pt[np.logical_and(nTagJets_to_LLP0==1,acc_LLP0_resolved_q1q2)]
                r_numLLP1_pt2_q1q2_1T  = genb3_pt[np.logical_and(nTagJets_to_LLP1==1,acc_LLP1_resolved_q1q2)]
                r_numLLP0_eta1_q1q2_1T = genb0_eta[np.logical_and(nTagJets_to_LLP0==1,acc_LLP0_resolved_q1q2)]
                r_numLLP1_eta1_q1q2_1T = genb2_eta[np.logical_and(nTagJets_to_LLP1==1,acc_LLP1_resolved_q1q2)]
                r_numLLP0_eta2_q1q2_1T = genb1_eta[np.logical_and(nTagJets_to_LLP0==1,acc_LLP0_resolved_q1q2)]
                r_numLLP1_eta2_q1q2_1T = genb3_eta[np.logical_and(nTagJets_to_LLP1==1,acc_LLP1_resolved_q1q2)]

                r_numLLP0_r_q1q2_2T   = genLLP0_r[np.logical_and(nTagJets_to_LLP0>1,acc_LLP0_resolved_q1q2)]
                r_numLLP1_r_q1q2_2T   = genLLP1_r[np.logical_and(nTagJets_to_LLP1>1,acc_LLP1_resolved_q1q2)]
                r_numLLP0_z_q1q2_2T   = genLLP0_z[np.logical_and(nTagJets_to_LLP0>1,acc_LLP0_resolved_q1q2)]
                r_numLLP1_z_q1q2_2T   = genLLP1_z[np.logical_and(nTagJets_to_LLP1>1,acc_LLP1_resolved_q1q2)]
                r_numLLP0_pt1_q1q2_2T  = genb0_pt[np.logical_and(nTagJets_to_LLP0>1,acc_LLP0_resolved_q1q2)]
                r_numLLP1_pt1_q1q2_2T  = genb2_pt[np.logical_and(nTagJets_to_LLP1>1,acc_LLP1_resolved_q1q2)]
                r_numLLP0_pt2_q1q2_2T  = genb1_pt[np.logical_and(nTagJets_to_LLP0>1,acc_LLP0_resolved_q1q2)]
                r_numLLP1_pt2_q1q2_2T  = genb3_pt[np.logical_and(nTagJets_to_LLP1>1,acc_LLP1_resolved_q1q2)]
                r_numLLP0_eta1_q1q2_2T = genb0_eta[np.logical_and(nTagJets_to_LLP0>1,acc_LLP0_resolved_q1q2)]
                r_numLLP1_eta1_q1q2_2T = genb2_eta[np.logical_and(nTagJets_to_LLP1>1,acc_LLP1_resolved_q1q2)]
                r_numLLP0_eta2_q1q2_2T = genb1_eta[np.logical_and(nTagJets_to_LLP0>1,acc_LLP0_resolved_q1q2)]
                r_numLLP1_eta2_q1q2_2T = genb3_eta[np.logical_and(nTagJets_to_LLP1>1,acc_LLP1_resolved_q1q2)]


                #One map per b-quark when q1, q2
                #q1
                r_den_LLP0_q1, r_den_LLP0_edges_q1 = np.histogramdd(np.transpose([r_denLLP0_r_q1,r_denLLP0_z_q1,r_denLLP0_pt_q1,r_denLLP0_eta_q1]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_den_LLP1_q1, r_den_LLP1_edges_q1 = np.histogramdd(np.transpose([r_denLLP1_r_q1,r_denLLP1_z_q1,r_denLLP1_pt_q1,r_denLLP1_eta_q1]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                #q2
                r_den_LLP0_q2, r_den_LLP0_edges_q2 = np.histogramdd(np.transpose([r_denLLP0_r_q2,r_denLLP0_z_q2,r_denLLP0_pt_q2,r_denLLP0_eta_q2]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_den_LLP1_q2, r_den_LLP1_edges_q2 = np.histogramdd(np.transpose([r_denLLP1_r_q2,r_denLLP1_z_q2,r_denLLP1_pt_q2,r_denLLP1_eta_q2]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                #6D maps for q1q2?
                r_den_LLP0_q1q2, r_den_LLP0_edges_q1q2 = np.histogramdd(np.transpose([r_denLLP0_r_q1q2,r_denLLP0_z_q1q2,r_denLLP0_pt1_q1q2,r_denLLP0_pt2_q1q2,r_denLLP0_eta1_q1q2,r_denLLP0_eta2_q1q2]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_pt_resolved,bins_eta_resolved,bins_eta_resolved))
                r_den_LLP1_q1q2, r_den_LLP1_edges_q1q2 = np.histogramdd(np.transpose([r_denLLP1_r_q1q2,r_denLLP1_z_q1q2,r_denLLP1_pt1_q1q2,r_denLLP1_pt2_q1q2,r_denLLP1_eta1_q1q2,r_denLLP1_eta2_q1q2]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_pt_resolved,bins_eta_resolved,bins_eta_resolved))

                #At least one tag
                #q1
                r_num_LLP0_q1_T, r_num_LLP0_edges_q1_T = np.histogramdd(np.transpose([r_numLLP0_r_q1_T,r_numLLP0_z_q1_T,r_numLLP0_pt_q1_T,r_numLLP0_eta_q1_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_num_LLP1_q1_T, r_num_LLP1_edges_q1_T = np.histogramdd(np.transpose([r_numLLP1_r_q1_T,r_numLLP1_z_q1_T,r_numLLP1_pt_q1_T,r_numLLP1_eta_q1_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_eff_LLP0_q1_T = np.divide(r_num_LLP0_q1_T,r_den_LLP0_q1,where=(r_den_LLP0_q1 != 0))
                r_eff_LLP1_q1_T = np.divide(r_num_LLP1_q1_T,r_den_LLP1_q1,where=(r_den_LLP1_q1 != 0))
                #q2
                r_num_LLP0_q2_T, r_num_LLP0_edges_q2_T = np.histogramdd(np.transpose([r_numLLP0_r_q2_T,r_numLLP0_z_q2_T,r_numLLP0_pt_q2_T,r_numLLP0_eta_q2_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_num_LLP1_q2_T, r_num_LLP1_edges_q2_T = np.histogramdd(np.transpose([r_numLLP1_r_q2_T,r_numLLP1_z_q2_T,r_numLLP1_pt_q2_T,r_numLLP1_eta_q2_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_eta_resolved))
                r_eff_LLP0_q2_T = np.divide(r_num_LLP0_q2_T,r_den_LLP0_q2,where=(r_den_LLP0_q2 != 0))
                r_eff_LLP1_q2_T = np.divide(r_num_LLP1_q2_T,r_den_LLP1_q2,where=(r_den_LLP1_q2 != 0))
                #q1q2
                #One tag
                r_num_LLP0_q1q2_1T, r_num_LLP0_edges_q1q2_1T = np.histogramdd(np.transpose([r_numLLP0_r_q1q2_1T,r_numLLP0_z_q1q2_1T,r_numLLP0_pt1_q1q2_1T,r_numLLP0_pt2_q1q2_1T,r_numLLP0_eta1_q1q2_1T,r_numLLP0_eta2_q1q2_1T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_pt_resolved,bins_eta_resolved,bins_eta_resolved))
                r_num_LLP1_q1q2_1T, r_num_LLP1_edges_q1q2_1T = np.histogramdd(np.transpose([r_numLLP1_r_q1q2_1T,r_numLLP1_z_q1q2_1T,r_numLLP1_pt1_q1q2_1T,r_numLLP1_pt2_q1q2_1T,r_numLLP1_eta1_q1q2_1T,r_numLLP1_eta2_q1q2_1T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_pt_resolved,bins_eta_resolved,bins_eta_resolved))
                r_eff_LLP0_q1q2_1T = np.divide(r_num_LLP0_q1q2_1T,r_den_LLP0_q1q2,where=(r_den_LLP0_q1q2 != 0))
                r_eff_LLP1_q1q2_1T = np.divide(r_num_LLP1_q1q2_1T,r_den_LLP1_q1q2,where=(r_den_LLP1_q1q2 != 0))
                #Two tags
                r_num_LLP0_q1q2_2T, r_num_LLP0_edges_q1q2_2T = np.histogramdd(np.transpose([r_numLLP0_r_q1q2_2T,r_numLLP0_z_q1q2_2T,r_numLLP0_pt1_q1q2_2T,r_numLLP0_pt2_q1q2_2T,r_numLLP0_eta1_q1q2_2T,r_numLLP0_eta2_q1q2_2T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_pt_resolved,bins_eta_resolved,bins_eta_resolved))
                r_num_LLP1_q1q2_2T, r_num_LLP1_edges_q1q2_2T = np.histogramdd(np.transpose([r_numLLP1_r_q1q2_2T,r_numLLP1_z_q1q2_2T,r_numLLP1_pt1_q1q2_2T,r_numLLP1_pt2_q1q2_2T,r_numLLP1_eta1_q1q2_2T,r_numLLP1_eta2_q1q2_2T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved,bins_pt_resolved,bins_eta_resolved,bins_eta_resolved))
                r_eff_LLP0_q1q2_2T = np.divide(r_num_LLP0_q1q2_2T,r_den_LLP0_q1q2,where=(r_den_LLP0_q1q2 != 0))
                r_eff_LLP1_q1q2_2T = np.divide(r_num_LLP1_q1q2_2T,r_den_LLP1_q1q2,where=(r_den_LLP1_q1q2 != 0))


        #Maps per sample:
        #m_eff_LLP0_T,m_eff_LLP1_T,m_eff_T
        #m_eff_LLP0_T_presel,m_eff_LLP1_T_presel,m_eff_T_presel
        #r_eff1_LLP0_T, r_eff1_LLP1_T, r_eff1_T
        #r_eff1_LLP0_T_presel, r_eff1_LLP1_T_presel, r_eff1_T_presel
        #r_eff2_LLP0_T, r_eff2_LLP1_T, r_eff2_T
        #r_eff2_LLP0_T_presel, r_eff2_LLP1_T_presel, r_eff2_T_presel

        maps[b]['merged']['eff_LLP0'] = m_eff_LLP0_T
        maps[b]['merged']['eff_LLP1'] = m_eff_LLP1_T
        maps[b]['merged']['eff'] = m_eff_T
        maps[b]['merged']['eff_LLP0_presel'] = m_eff_LLP0_T_presel
        maps[b]['merged']['eff_LLP1_presel'] = m_eff_LLP1_T_presel
        maps[b]['merged']['eff_presel'] = m_eff_T_presel
        maps[b]['merged']['bins_r'] = bins_r_merged
        maps[b]['merged']['bins_z'] = bins_z_merged
        maps[b]['merged']['bins_pt'] = bins_pt_merged
        maps[b]['merged']['bins_eta'] = bins_eta_merged

        '''
        Old way with the or
        maps[b]['resolved']['eff_LLP0_q1'] = r_eff1_LLP0_T
        maps[b]['resolved']['eff_LLP1_q1'] = r_eff1_LLP1_T
        maps[b]['resolved']['eff_q1'] = r_eff1_T
        maps[b]['resolved']['eff_LLP0_q2'] = r_eff2_LLP0_T
        maps[b]['resolved']['eff_LLP1_q2'] = r_eff2_LLP1_T
        maps[b]['resolved']['eff_q2'] = r_eff2_T
        maps[b]['resolved']['eff_LLP0_q1_presel'] = r_eff1_LLP0_T_presel
        maps[b]['resolved']['eff_LLP1_q1_presel'] = r_eff1_LLP1_T_presel
        maps[b]['resolved']['eff_q1_presel'] = r_eff1_T_presel
        maps[b]['resolved']['eff_LLP0_q2_presel'] = r_eff2_LLP0_T_presel
        maps[b]['resolved']['eff_LLP1_q2_presel'] = r_eff2_LLP1_T_presel
        maps[b]['resolved']['eff_q2_presel'] = r_eff2_T_presel
        maps[b]['resolved']['bins_r'] = bins_r_resolved
        maps[b]['resolved']['bins_z'] = bins_z_resolved
        maps[b]['resolved']['bins_pt'] = bins_pt_resolved
        maps[b]['resolved']['bins_eta'] = bins_eta_resolved
        '''

        maps[b]['resolved']['eff_LLP0_q1'] = r_eff_LLP0_q1_T
        maps[b]['resolved']['eff_LLP1_q1'] = r_eff_LLP1_q1_T
        maps[b]['resolved']['eff_LLP0_q2'] = r_eff_LLP0_q2_T
        maps[b]['resolved']['eff_LLP1_q2'] = r_eff_LLP1_q2_T
        maps[b]['resolved']['eff_LLP0_q1q2_1T'] = r_eff_LLP0_q1q2_1T
        maps[b]['resolved']['eff_LLP1_q1q2_1T'] = r_eff_LLP1_q1q2_1T
        maps[b]['resolved']['eff_LLP0_q1q2_2T'] = r_eff_LLP0_q1q2_2T
        maps[b]['resolved']['eff_LLP1_q1q2_2T'] = r_eff_LLP1_q1q2_2T
        maps[b]['resolved']['bins_r'] = bins_r_resolved
        maps[b]['resolved']['bins_z'] = bins_z_resolved
        maps[b]['resolved']['bins_pt'] = bins_pt_resolved
        maps[b]['resolved']['bins_eta'] = bins_eta_resolved

        with open(MAIN_OUT+"Tag_eff_map_"+b+".yaml","w") as f:
            yaml.dump(maps[b], f)
            f.close()
            print "Info: dictionary written in file "+MAIN_OUT+"Tag_eff_map_"+b+".yaml"

def use_maps(sign, skip_odd=False):

    list_of_variables = [
        "GenLLPs*","GenBquarks*","GenHiggs*","nTagJets_0p996","nTagJets_to_LLP0","nTagJets_to_LLP1","nJets_to_LLP0","nJets_to_LLP1","isPreselected","pt","EventNumber",
    ]


    #Define tables
    table_m = PrettyTable([
        'signal',
        'true LLP0',
        'pred LLP0',
        'diff LLP0',
        'true LLP1',
        'pred LLP1',
        'diff LLP1',
    ])

    table_r1_LLP0 = PrettyTable([
        'signal',
        'true q1 LLP0',
        'pred q1 LLP0',
        'diff q1 LLP0',
        'true q2 LLP0',
        'pred q2 LLP0',
        'diff q2 LLP0',
    ])


    table_r2_LLP0 = PrettyTable([
        'signal',
        'true q1q2 1T LLP0',
        'pred q1q2 1T LLP0',
        'diff q1q2 1T LLP0',
        'true q1q2 2T LLP0',
        'pred q1q2 2T LLP0',
        'diff q1q2 2T LLP0',
    ])

    table_r1_LLP1 = PrettyTable([
        'signal',
        'true q1 LLP1',
        'pred q1 LLP1',
        'diff q1 LLP1',
        'true q2 LLP1',
        'pred q2 LLP1',
        'diff q2 LLP1',
    ])

    table_r2_LLP1 = PrettyTable([
        'signal',
        'true q1q2 1T LLP1',
        'pred q1q2 1T LLP1',
        'diff q1q2 1T LLP1',
        'true q1q2 2T LLP1',
        'pred q1q2 2T LLP1',
        'diff q1q2 2T LLP1',
    ])

    chunk_size = 1000000
    for b in sign:

        with open(MAIN_OUT+"Tag_eff_map_"+b+".yaml") as f:
            maps = yaml.load(f, Loader=yaml.Loader)
            f.close()

        #print maps

        for i, ss in enumerate(samples[b]['files']):
            print "Uproot iterating over ", NTUPLEDIR +ss+ '.root', " . . . "
            filename = NTUPLEDIR + ss + '.root'
            gen = uproot.iterate(filename,"tree",list_of_variables,entrysteps=chunk_size)
            for arrays in gen:
                
                n_events = -1            
                #Take one event every two: to use one set as validation
                step = 2 if skip_odd else 1
                start = 1 if skip_odd else 0

                genLLP0_dR = np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][start:n_events:step] )))[0:1]
                genLLP1_dR = np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][start:n_events:step] )))[1:2]

                #print "N. events considered for using maps..."
                #print genLLP0_dR.shape
                #exit()

                genLLP0_merged = (genLLP0_dR < 0.4)
                genLLP1_merged = (genLLP1_dR < 0.4)

                genLLP0_resolved = np.logical_not(genLLP0_merged)
                genLLP1_resolved = np.logical_not(genLLP1_merged)

                genLLP0_r = np.transpose(np.stack(( arrays["GenLLPs.travelRadius"][start:n_events:step]/100. )))[0:1]
                genLLP1_r = np.transpose(np.stack(( arrays["GenLLPs.travelRadius"][start:n_events:step]/100. )))[1:2]

                genLLP0_z = np.absolute( np.transpose(np.stack(( arrays["GenLLPs.travelZ"][start:n_events:step]/100. )))[0:1] )
                genLLP1_z = np.absolute( np.transpose(np.stack(( arrays["GenLLPs.travelZ"][start:n_events:step]/100. )))[1:2] )

                genh0_pt = np.transpose(np.stack(( arrays["GenHiggs.pt"][start:n_events:step] )))[0:1]
                genh1_pt = np.transpose(np.stack(( arrays["GenHiggs.pt"][start:n_events:step] )))[1:2]

                genh0_eta = np.absolute( np.transpose(np.stack(( arrays["GenHiggs.eta"][start:n_events:step] )))[0:1] )
                genh1_eta = np.absolute( np.transpose(np.stack(( arrays["GenHiggs.eta"][start:n_events:step] )))[1:2] )

                genb0_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][start:n_events:step] )))[0:1]
                genb1_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][start:n_events:step] )))[1:2]
                genb2_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][start:n_events:step] )))[2:3]
                genb3_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][start:n_events:step] )))[3:]

                genb0_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][start:n_events:step] )))[0:1] )
                genb1_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][start:n_events:step] )))[1:2] )
                genb2_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][start:n_events:step] )))[2:3] )
                genb3_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][start:n_events:step] )))[3:] )

                #Easiest way: count how many tagged jets are associated to each LLP
                #include also an event-level flag for pre-selections
                LLP0_m = np.transpose(np.stack(( arrays["GenLLPs.mass"][start:n_events:step] )))[0:1].flatten()
                LLP0_dR = genLLP0_dR.flatten()
                LLP0_r = genLLP0_r.flatten()
                LLP0_z = genLLP0_z.flatten()
                LLP0_pt = genh0_pt.flatten()
                LLP0_eta = genh0_eta.flatten()
                LLP0_pt1 = genb0_pt.flatten()
                LLP0_eta1 = genb0_eta.flatten()
                LLP0_pt2 = genb1_pt.flatten()
                LLP0_eta2 = genb1_eta.flatten()

                LLP1_m = np.transpose(np.stack(( arrays["GenLLPs.mass"][start:n_events:step] )))[1:2].flatten()
                LLP1_dR = genLLP1_dR.flatten()
                LLP1_r = genLLP1_r.flatten()
                LLP1_z = genLLP1_z.flatten()
                LLP1_pt = genh1_pt.flatten()
                LLP1_eta = genh1_eta.flatten()
                LLP1_pt1 = genb2_pt.flatten()
                LLP1_eta1 = genb2_eta.flatten()
                LLP1_pt2 = genb3_pt.flatten()
                LLP1_eta2 = genb3_eta.flatten()

                nTagJets = arrays["nTagJets_0p996"][start:n_events:step].flatten()
                nTagJets_to_LLP0 = arrays["nTagJets_to_LLP0"][start:n_events:step].flatten()
                nTagJets_to_LLP1 = arrays["nTagJets_to_LLP1"][start:n_events:step].flatten()

        #Here use the maps!
        #Inputs are simple arrays
        
        #print LLP1_m.shape
        #print LLP1_pt.shape

        LLP0_merged = (LLP0_dR < 0.4)
        LLP1_merged = (LLP1_dR < 0.4)

        LLP0_resolved = np.logical_not(LLP0_merged)
        LLP1_resolved = np.logical_not(LLP1_merged)

        #print "Acceptance cut missing!!!"
        acc_LLP0_merged = np.logical_and( np.logical_and(LLP0_pt>30.,abs(LLP0_eta)<1.) , LLP0_merged)
        acc_LLP1_merged = np.logical_and( np.logical_and(LLP1_pt>30.,abs(LLP1_eta)<1.) , LLP1_merged)

        LLP0_merged = np.logical_and(LLP0_merged,acc_LLP0_merged)
        LLP1_merged = np.logical_and(LLP1_merged,acc_LLP1_merged)

        nTagLLP0_m = nTagJets_to_LLP0[LLP0_merged]
        nTagLLP1_m = nTagJets_to_LLP1[LLP1_merged]

        #resolved acceptance here....
        #LLP0
        #Only q1 in acceptance
        acc_LLP0_resolved_q1 = np.logical_and( 
            np.logical_and( 
                np.logical_and(LLP0_pt1>30.,abs(LLP0_eta1)<1.), 
                np.logical_not( np.logical_and( LLP0_pt2>30.,abs(LLP0_eta2)<1.) )
            ), 
            LLP0_resolved)
        #Only q2 in acceptance
        acc_LLP0_resolved_q2 = np.logical_and( 
            np.logical_and( 
                np.logical_not( np.logical_and(LLP0_pt1>30.,abs(LLP0_eta1)<1.) ), 
                np.logical_and(LLP0_pt2>30.,abs(LLP0_eta2)<1.)
            ), 
            LLP0_resolved)
        #Both q1 and q2 in acceptance
        acc_LLP0_resolved_q1q2 = np.logical_and( 
            np.logical_and( 
                np.logical_and(LLP0_pt1>30.,abs(LLP0_eta1)<1.), 
                np.logical_and(LLP0_pt2>30.,abs(LLP0_eta2)<1.)
            ), 
            LLP0_resolved)

        #LLP1
        #Only q1 in acceptance
        acc_LLP1_resolved_q1 = np.logical_and( 
            np.logical_and( 
                np.logical_and(LLP1_pt1>30.,abs(LLP1_eta1)<1.), 
                np.logical_not( np.logical_and( LLP1_pt2>30.,abs(LLP1_eta2)<1.) )
            ), 
            LLP1_resolved)
        #Only q2 in acceptance
        acc_LLP1_resolved_q2 = np.logical_and( 
            np.logical_and( 
                np.logical_not( np.logical_and(LLP1_pt1>30.,abs(LLP1_eta1)<1.) ), 
                np.logical_and(LLP1_pt2>30.,abs(LLP1_eta2)<1.)
            ), 
            LLP1_resolved)
        #Both q1 and q2 in acceptance
        acc_LLP1_resolved_q1q2 = np.logical_and( 
            np.logical_and( 
                np.logical_and(LLP1_pt1>30.,abs(LLP1_eta1)<1.), 
                np.logical_and(LLP1_pt2>30.,abs(LLP1_eta2)<1.)
            ), 
            LLP1_resolved)

        LLP0_resolved_q1 = np.logical_and(acc_LLP0_resolved_q1,LLP0_resolved)
        LLP0_resolved_q2 = np.logical_and(acc_LLP0_resolved_q2,LLP0_resolved)
        LLP0_resolved_q1q2 = np.logical_and(acc_LLP0_resolved_q1q2,LLP0_resolved)

        LLP1_resolved_q1 = np.logical_and(acc_LLP1_resolved_q1,LLP1_resolved)
        LLP1_resolved_q2 = np.logical_and(acc_LLP1_resolved_q2,LLP1_resolved)
        LLP1_resolved_q1q2 = np.logical_and(acc_LLP1_resolved_q1q2,LLP1_resolved)

        nTagLLP0_r_q1 = nTagJets_to_LLP0[LLP0_resolved_q1]
        nTagLLP0_r_q2 = nTagJets_to_LLP0[LLP0_resolved_q2]
        nTagLLP0_r_q1q2 = nTagJets_to_LLP0[LLP0_resolved_q1q2]

        nTagLLP1_r_q1 = nTagJets_to_LLP1[LLP1_resolved_q1]
        nTagLLP1_r_q2 = nTagJets_to_LLP1[LLP1_resolved_q2]
        nTagLLP1_r_q1q2 = nTagJets_to_LLP1[LLP1_resolved_q1q2]


        #Merged pred.
        m_LLP0_r = LLP0_r[LLP0_merged]
        m_LLP0_z = LLP0_z[LLP0_merged]
        m_LLP0_pt = LLP0_pt[LLP0_merged]
        m_LLP0_eta = LLP0_eta[LLP0_merged]

        m_LLP1_r = LLP1_r[LLP1_merged]
        m_LLP1_z = LLP1_z[LLP1_merged]
        m_LLP1_pt = LLP1_pt[LLP1_merged]
        m_LLP1_eta = LLP1_eta[LLP1_merged]


        bins_r_merged =  maps['merged']['bins_r']
        bins_z_merged = maps['merged']['bins_z']
        bins_pt_merged = maps['merged']['bins_pt']
        bins_eta_merged = maps['merged']['bins_eta']

        #digitize is the right function to use
        m_LLP0_r_idx = np.digitize(m_LLP0_r,maps['merged']['bins_r'])-1
        m_LLP0_z_idx = np.digitize(m_LLP0_z,maps['merged']['bins_z'])-1
        m_LLP0_pt_idx = np.digitize(m_LLP0_pt,maps['merged']['bins_pt'])-1
        m_LLP0_eta_idx = np.digitize(m_LLP0_eta,maps['merged']['bins_eta'])-1

        m_LLP1_r_idx = np.digitize(m_LLP1_r,maps['merged']['bins_r'])-1
        m_LLP1_z_idx = np.digitize(m_LLP1_z,maps['merged']['bins_z'])-1
        m_LLP1_pt_idx = np.digitize(m_LLP1_pt,maps['merged']['bins_pt'])-1
        m_LLP1_eta_idx = np.digitize(m_LLP1_eta,maps['merged']['bins_eta'])-1


        #print "bin content"
        #print "r ",bins_r_merged
        #print "z ",bins_z_merged
        #print "pt ",bins_pt_merged
        #print "eta ",bins_eta_merged

        #print "indices position"
        #print np.transpose([m_LLP0_r_idx,m_LLP0_z_idx,m_LLP0_pt_idx,m_LLP0_eta_idx])

        nTagLLP0_pred_m = 0
        nTagLLP1_pred_m = 0

        for it in range(len(m_LLP0_r_idx)):
            #print "Ev n. ", it
            #print "actual values"
            #print "r, z, pt, eta"
            #print m_LLP0_r[it], m_LLP0_z[it], m_LLP0_pt[it], m_LLP0_eta[it] 
            #print "bin content"
            #print "r ",bins_r_merged
            #print "z ",bins_z_merged
            #print "pt ",bins_pt_merged
            #print "eta ",bins_eta_merged
            ##print "bin indices"
            ##print m_LLP0_r_idx[it],m_LLP0_z_idx[it],m_LLP0_pt_idx[it],m_LLP0_eta_idx[it]
            ##print "map value"
            pred = maps['merged']['eff_LLP0'][ m_LLP0_r_idx[it] ][  m_LLP0_z_idx[it] ][  m_LLP0_pt_idx[it] ][  m_LLP0_eta_idx[it] ]
            #print "predicted value: ", pred
            nTagLLP0_pred_m+= pred
            #print "nTagLLP0_pred_m temp: ", nTagLLP0_pred_m

        for it in range(len(m_LLP1_r_idx)):
            pred = maps['merged']['eff_LLP1'][ m_LLP1_r_idx[it] ][  m_LLP1_z_idx[it] ][  m_LLP1_pt_idx[it] ][  m_LLP1_eta_idx[it] ]
            nTagLLP1_pred_m+= pred

        print "\n"
        print "   M e r g e d"
        print "\n"

        '''
        print "True tags to LLP0 (stat unc)"
        #print nTagLLP0_m
        print nTagLLP0_m[nTagLLP0_m>0].astype(bool).sum(), " +- ", np.sqrt(nTagLLP0_m[nTagLLP0_m>0].astype(bool).sum()), " (", (100*np.sqrt(nTagLLP0_m[nTagLLP0_m>0].astype(bool).sum()) / nTagLLP0_m[nTagLLP0_m>0].astype(bool).sum() ), "%)"

        print "Predicted tags to LLP0"
        print nTagLLP0_pred_m
        print "Wrong by: ", 100*( (nTagLLP0_pred_m-nTagLLP0_m[nTagLLP0_m>0].astype(bool).sum())/ ( nTagLLP0_m[nTagLLP0_m>0].astype(bool).sum() ) ), "%"

        print "True tags to LLP1 (stat unc)"
        #print nTagLLP1_m
        print nTagLLP1_m[nTagLLP1_m>0].astype(bool).sum(), " +- ", np.sqrt(nTagLLP1_m[nTagLLP1_m>0].astype(bool).sum()), " (", (100*np.sqrt(nTagLLP1_m[nTagLLP1_m>0].astype(bool).sum()) / nTagLLP1_m[nTagLLP1_m>0].astype(bool).sum() ),"%)"

        print "Predicted tags to LLP1"
        print nTagLLP1_pred_m
        print "Wrong by: ", 100*( (nTagLLP1_pred_m-nTagLLP1_m[nTagLLP1_m>0].astype(bool).sum())/ ( nTagLLP1_m[nTagLLP1_m>0].astype(bool).sum() ) ), "%"
        '''


        row = [
            b,
            str(nTagLLP0_m[nTagLLP0_m>0].astype(bool).sum()) + " +- " + str( round( (100*np.sqrt(nTagLLP0_m[nTagLLP0_m>0].astype(bool).sum()) / nTagLLP0_m[nTagLLP0_m>0].astype(bool).sum() ),2) ) + "%",
            str( round(nTagLLP0_pred_m,2) ) + " +- " + str( round( (100*np.sqrt(nTagLLP0_pred_m) / nTagLLP0_pred_m),2) ) + "%",
            str( round( 100*( (nTagLLP0_pred_m-nTagLLP0_m[nTagLLP0_m>0].astype(bool).sum())/ ( nTagLLP0_m[nTagLLP0_m>0].astype(bool).sum() ) ), 2) ) + "%",

            str(nTagLLP1_m[nTagLLP1_m>0].astype(bool).sum()) + " +- " + str( round( (100*np.sqrt(nTagLLP1_m[nTagLLP1_m>0].astype(bool).sum()) / nTagLLP1_m[nTagLLP1_m>0].astype(bool).sum() ),2) ) + "%",
            str( round(nTagLLP1_pred_m,2) ) + " +- " + str( round( (100*np.sqrt(nTagLLP1_pred_m) / nTagLLP1_pred_m),2) ) + "%",
            str( round( 100*( (nTagLLP1_pred_m-nTagLLP1_m[nTagLLP1_m>0].astype(bool).sum())/ ( nTagLLP1_m[nTagLLP1_m>0].astype(bool).sum() ) ), 2) ) + "%",
        ]

        table_m.add_row(row)

        #resolved pred 
        r_LLP0_r_q1 = LLP0_r[LLP0_resolved_q1]
        r_LLP0_z_q1 = LLP0_z[LLP0_resolved_q1]
        r_LLP0_pt_q1 = LLP0_pt1[LLP0_resolved_q1]
        r_LLP0_eta_q1 = LLP0_eta1[LLP0_resolved_q1]

        r_LLP0_r_q2 = LLP0_r[LLP0_resolved_q2]
        r_LLP0_z_q2 = LLP0_z[LLP0_resolved_q2]
        r_LLP0_pt_q2 = LLP0_pt2[LLP0_resolved_q2]
        r_LLP0_eta_q2 = LLP0_eta2[LLP0_resolved_q2]

        r_LLP0_r_q1q2 = LLP0_r[LLP0_resolved_q1q2]
        r_LLP0_z_q1q2 = LLP0_z[LLP0_resolved_q1q2]
        r_LLP0_pt1_q1q2 = LLP0_pt1[LLP0_resolved_q1q2]
        r_LLP0_pt2_q1q2 = LLP0_pt2[LLP0_resolved_q1q2]
        r_LLP0_eta1_q1q2 = LLP0_eta1[LLP0_resolved_q1q2]
        r_LLP0_eta2_q1q2 = LLP0_eta2[LLP0_resolved_q1q2]


        r_LLP1_r_q1 = LLP1_r[LLP1_resolved_q1]
        r_LLP1_z_q1 = LLP1_z[LLP1_resolved_q1]
        r_LLP1_pt_q1 = LLP1_pt1[LLP1_resolved_q1]
        r_LLP1_eta_q1 = LLP1_eta1[LLP1_resolved_q1]

        r_LLP1_r_q2 = LLP1_r[LLP1_resolved_q2]
        r_LLP1_z_q2 = LLP1_z[LLP1_resolved_q2]
        r_LLP1_pt_q2 = LLP1_pt2[LLP1_resolved_q2]
        r_LLP1_eta_q2 = LLP1_eta2[LLP1_resolved_q2]

        r_LLP1_r_q1q2 = LLP1_r[LLP1_resolved_q1q2]
        r_LLP1_z_q1q2 = LLP1_z[LLP1_resolved_q1q2]
        r_LLP1_pt1_q1q2 = LLP1_pt1[LLP1_resolved_q1q2]
        r_LLP1_pt2_q1q2 = LLP1_pt2[LLP1_resolved_q1q2]
        r_LLP1_eta1_q1q2 = LLP1_eta1[LLP1_resolved_q1q2]
        r_LLP1_eta2_q1q2 = LLP1_eta2[LLP1_resolved_q1q2]

        bins_r_resolved =  maps['resolved']['bins_r']
        bins_z_resolved = maps['resolved']['bins_z']
        bins_pt_resolved = maps['resolved']['bins_pt']
        bins_eta_resolved = maps['resolved']['bins_eta']

        #digitize is the right function to use
        r_LLP0_r_idx_q1    = np.digitize(r_LLP0_r_q1,maps['resolved']['bins_r'])-1
        r_LLP0_z_idx_q1    = np.digitize(r_LLP0_z_q1,maps['resolved']['bins_z'])-1
        r_LLP0_pt_idx_q1   = np.digitize(r_LLP0_pt_q1,maps['resolved']['bins_pt'])-1
        r_LLP0_eta_idx_q1  = np.digitize(r_LLP0_eta_q1,maps['resolved']['bins_eta'])-1

        r_LLP0_r_idx_q2    = np.digitize(r_LLP0_r_q2,maps['resolved']['bins_r'])-1
        r_LLP0_z_idx_q2    = np.digitize(r_LLP0_z_q2,maps['resolved']['bins_z'])-1
        r_LLP0_pt_idx_q2   = np.digitize(r_LLP0_pt_q2,maps['resolved']['bins_pt'])-1
        r_LLP0_eta_idx_q2  = np.digitize(r_LLP0_eta_q2,maps['resolved']['bins_eta'])-1

        r_LLP0_r_idx_q1q2     = np.digitize(r_LLP0_r_q1q2,maps['resolved']['bins_r'])-1
        r_LLP0_z_idx_q1q2     = np.digitize(r_LLP0_z_q1q2,maps['resolved']['bins_z'])-1
        r_LLP0_pt1_idx_q1q2   = np.digitize(r_LLP0_pt1_q1q2,maps['resolved']['bins_pt'])-1
        r_LLP0_pt2_idx_q1q2   = np.digitize(r_LLP0_pt2_q1q2,maps['resolved']['bins_pt'])-1
        r_LLP0_eta1_idx_q1q2  = np.digitize(r_LLP0_eta1_q1q2,maps['resolved']['bins_eta'])-1
        r_LLP0_eta2_idx_q1q2  = np.digitize(r_LLP0_eta2_q1q2,maps['resolved']['bins_eta'])-1

        r_LLP1_r_idx_q1    = np.digitize(r_LLP1_r_q1,maps['resolved']['bins_r'])-1
        r_LLP1_z_idx_q1    = np.digitize(r_LLP1_z_q1,maps['resolved']['bins_z'])-1
        r_LLP1_pt_idx_q1   = np.digitize(r_LLP1_pt_q1,maps['resolved']['bins_pt'])-1
        r_LLP1_eta_idx_q1  = np.digitize(r_LLP1_eta_q1,maps['resolved']['bins_eta'])-1

        r_LLP1_r_idx_q2    = np.digitize(r_LLP1_r_q2,maps['resolved']['bins_r'])-1
        r_LLP1_z_idx_q2    = np.digitize(r_LLP1_z_q2,maps['resolved']['bins_z'])-1
        r_LLP1_pt_idx_q2   = np.digitize(r_LLP1_pt_q2,maps['resolved']['bins_pt'])-1
        r_LLP1_eta_idx_q2  = np.digitize(r_LLP1_eta_q2,maps['resolved']['bins_eta'])-1

        r_LLP1_r_idx_q1q2     = np.digitize(r_LLP1_r_q1q2,maps['resolved']['bins_r'])-1
        r_LLP1_z_idx_q1q2     = np.digitize(r_LLP1_z_q1q2,maps['resolved']['bins_z'])-1
        r_LLP1_pt1_idx_q1q2   = np.digitize(r_LLP1_pt1_q1q2,maps['resolved']['bins_pt'])-1
        r_LLP1_pt2_idx_q1q2   = np.digitize(r_LLP1_pt2_q1q2,maps['resolved']['bins_pt'])-1
        r_LLP1_eta1_idx_q1q2  = np.digitize(r_LLP1_eta1_q1q2,maps['resolved']['bins_eta'])-1
        r_LLP1_eta2_idx_q1q2  = np.digitize(r_LLP1_eta2_q1q2,maps['resolved']['bins_eta'])-1


        #LLP0 predictions
        nTagLLP0_pred_r_q1 = 0
        nTagLLP0_pred_r_q2 = 0
        nTagLLP0_pred_r_q1q2_1T = 0
        nTagLLP0_pred_r_q1q2_2T = 0

        for it in range(len(r_LLP0_r_idx_q1)):
            #print "Ev n. ", it
            #print "actual values"
            #print "r, z, pt, eta"
            #print r_LLP1_r[it], r_LLP1_z[it], r_LLP1_pt1[it], r_LLP1_eta1[it] 
            #print "bin content"
            #print "r ",bins_r_merged
            #print "z ",bins_z_merged
            #print "pt ",bins_pt_merged
            #print "eta ",bins_eta_merged
            pred = maps['resolved']['eff_LLP0_q1'][ r_LLP0_r_idx_q1[it] ][  r_LLP0_z_idx_q1[it] ][  r_LLP0_pt_idx_q1[it] ][  r_LLP0_eta_idx_q1[it] ]
            nTagLLP0_pred_r_q1+= pred


        for it in range(len(r_LLP0_r_idx_q2)):
            pred = maps['resolved']['eff_LLP0_q2'][ r_LLP0_r_idx_q2[it] ][  r_LLP0_z_idx_q2[it] ][  r_LLP0_pt_idx_q2[it] ][  r_LLP0_eta_idx_q2[it] ]
            nTagLLP0_pred_r_q2+= pred


        for it in range(len(r_LLP0_r_idx_q1q2)):
            pred_1T = maps['resolved']['eff_LLP0_q1q2_1T'][ r_LLP0_r_idx_q1q2[it] ][  r_LLP0_z_idx_q1q2[it] ][  r_LLP0_pt1_idx_q1q2[it] ][  r_LLP0_pt2_idx_q1q2[it] ][  r_LLP0_eta1_idx_q1q2[it] ][  r_LLP0_eta2_idx_q1q2[it] ]
            nTagLLP0_pred_r_q1q2_1T+= pred_1T

            pred_2T = maps['resolved']['eff_LLP0_q1q2_2T'][ r_LLP0_r_idx_q1q2[it] ][  r_LLP0_z_idx_q1q2[it] ][  r_LLP0_pt1_idx_q1q2[it] ][  r_LLP0_pt2_idx_q1q2[it] ][  r_LLP0_eta1_idx_q1q2[it] ][  r_LLP0_eta2_idx_q1q2[it] ]
            nTagLLP0_pred_r_q1q2_2T+= pred_2T


        #LLP1 predictions
        nTagLLP1_pred_r_q1 = 0
        nTagLLP1_pred_r_q2 = 0
        nTagLLP1_pred_r_q1q2_1T = 0
        nTagLLP1_pred_r_q1q2_2T = 0

        for it in range(len(r_LLP1_r_idx_q1)):
            #print "Ev n. ", it
            #print "actual values"
            #print "r, z, pt, eta"
            #print r_LLP1_r[it], r_LLP1_z[it], r_LLP1_pt1[it], r_LLP1_eta1[it] 
            #print "bin content"
            #print "r ",bins_r_merged
            #print "z ",bins_z_merged
            #print "pt ",bins_pt_merged
            #print "eta ",bins_eta_merged
            pred = maps['resolved']['eff_LLP1_q1'][ r_LLP1_r_idx_q1[it] ][  r_LLP1_z_idx_q1[it] ][  r_LLP1_pt_idx_q1[it] ][  r_LLP1_eta_idx_q1[it] ]
            nTagLLP1_pred_r_q1+= pred


        for it in range(len(r_LLP1_r_idx_q2)):
            pred = maps['resolved']['eff_LLP1_q2'][ r_LLP1_r_idx_q2[it] ][  r_LLP1_z_idx_q2[it] ][  r_LLP1_pt_idx_q2[it] ][  r_LLP1_eta_idx_q2[it] ]
            nTagLLP1_pred_r_q2+= pred


        for it in range(len(r_LLP1_r_idx_q1q2)):
            pred_1T = maps['resolved']['eff_LLP1_q1q2_1T'][ r_LLP1_r_idx_q1q2[it] ][  r_LLP1_z_idx_q1q2[it] ][  r_LLP1_pt1_idx_q1q2[it] ][  r_LLP1_pt2_idx_q1q2[it] ][  r_LLP1_eta1_idx_q1q2[it] ][  r_LLP1_eta2_idx_q1q2[it] ]
            nTagLLP1_pred_r_q1q2_1T+= pred_1T

            pred_2T = maps['resolved']['eff_LLP1_q1q2_2T'][ r_LLP1_r_idx_q1q2[it] ][  r_LLP1_z_idx_q1q2[it] ][  r_LLP1_pt1_idx_q1q2[it] ][  r_LLP1_pt2_idx_q1q2[it] ][  r_LLP1_eta1_idx_q1q2[it] ][  r_LLP1_eta2_idx_q1q2[it] ]
            nTagLLP1_pred_r_q1q2_2T+= pred_2T


        print "\n"
        print "   R e s o l v e d"
        print "\n"

        t_nTagLLP0_r_q1 = nTagLLP0_r_q1[nTagLLP0_r_q1>0].astype(bool).sum()
        t_nTagLLP0_r_q1_unc = 100./np.sqrt(nTagLLP0_r_q1[nTagLLP0_r_q1>0].astype(bool).sum())
        t_diff_nTagLLP0_r_q1 = 100*( (nTagLLP0_pred_r_q1-nTagLLP0_r_q1[nTagLLP0_r_q1>0].astype(bool).sum())/ ( nTagLLP0_r_q1[nTagLLP0_r_q1>0].astype(bool).sum() ) )

        t_nTagLLP0_r_q2 = nTagLLP0_r_q2[nTagLLP0_r_q2>0].astype(bool).sum()
        t_nTagLLP0_r_q2_unc = 100./np.sqrt(nTagLLP0_r_q2[nTagLLP0_r_q2>0].astype(bool).sum())
        t_diff_nTagLLP0_r_q2 = 100*( (nTagLLP0_pred_r_q2-nTagLLP0_r_q2[nTagLLP0_r_q2>0].astype(bool).sum())/ ( nTagLLP0_r_q2[nTagLLP0_r_q2>0].astype(bool).sum() ) )

        t_nTagLLP0_r_q1q2_1T = nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2==1 ].astype(bool).sum()
        t_nTagLLP0_r_q1q2_1T_unc = 100./np.sqrt(nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2==1 ].astype(bool).sum())
        t_diff_nTagLLP0_r_q1q2_1T = 100*( ( (nTagLLP0_pred_r_q1q2_1T)- nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2==1 ].astype(bool).sum() )/ ( nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2==1 ].astype(bool).sum() ) )

        t_nTagLLP0_r_q1q2_2T = nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2>1 ].astype(bool).sum()
        t_nTagLLP0_r_q1q2_2T_unc = 100./np.sqrt(nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2>1 ].astype(bool).sum())
        t_diff_nTagLLP0_r_q1q2_2T = 100*( ( (nTagLLP0_pred_r_q1q2_2T)- nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2>1 ].astype(bool).sum() )/ ( nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2>1 ].astype(bool).sum() ) )




        row = [
            b,
            str(t_nTagLLP0_r_q1) + " +- " + str( round( t_nTagLLP0_r_q1_unc,2) ) + "%",
            str(round(nTagLLP0_pred_r_q1,2) ) + " +- " + str( round( 100./np.sqrt(nTagLLP0_pred_r_q1),2) ) + "%",
            str( round(t_diff_nTagLLP0_r_q1, 2) ) + "%",

            str(t_nTagLLP0_r_q2) + " +- " + str( round( t_nTagLLP0_r_q2_unc,2) ) + "%",
            str(round(nTagLLP0_pred_r_q2,2) ) + " +- " + str( round( 100./np.sqrt(nTagLLP0_pred_r_q2),2) ) + "%",
            str( round(t_diff_nTagLLP0_r_q2, 2) ) + "%",

        ]

        table_r1_LLP0.add_row(row)



        row = [
            b,
            str(t_nTagLLP0_r_q1q2_1T) + " +- " + str( round( t_nTagLLP0_r_q1q2_1T_unc,2) ) + "%",
            str(round(nTagLLP0_pred_r_q1q2_1T,2) ) + " +- " + str( round( 100./np.sqrt(nTagLLP0_pred_r_q1q2_1T),2) ) + "%",
            str( round(t_diff_nTagLLP0_r_q1q2_1T, 2) ) + "%",

            str(t_nTagLLP0_r_q1q2_2T) + " +- " + str( round( t_nTagLLP0_r_q1q2_2T_unc,2) ) + "%",
            str(round(nTagLLP0_pred_r_q1q2_2T,2) ) + " +- " + str( round( 100./np.sqrt(nTagLLP0_pred_r_q1q2_2T),2) ) + "%",
            str( round(t_diff_nTagLLP0_r_q1q2_2T, 2) ) + "%",


        ]

        table_r2_LLP0.add_row(row)

        '''
        print "True tags to LLP0, q1"
        print nTagLLP0_r_q1[nTagLLP0_r_q1>0].astype(bool).sum()
        print "Predicted q1 tags to LLP0"
        print nTagLLP0_pred_r_q1
        print "Wrong by: ", 100*( (nTagLLP0_pred_r_q1-nTagLLP0_r_q1[nTagLLP0_r_q1>0].astype(bool).sum())/ ( nTagLLP0_r_q1[nTagLLP0_r_q1>0].astype(bool).sum() ) ), "%"

        print "\n"
        print "True tags to LLP0, q2"
        print nTagLLP0_r_q2[nTagLLP0_r_q2>0].astype(bool).sum()
        print "Predicted q2 tags to LLP0"
        print nTagLLP0_pred_r_q2
        print "Wrong by: ", 100*( (nTagLLP0_pred_r_q2-nTagLLP0_r_q2[nTagLLP0_r_q2>0].astype(bool).sum())/ ( nTagLLP0_r_q2[nTagLLP0_r_q2>0].astype(bool).sum() ) ), "%"

        print "\n"
        print "True tags to LLP0, q1 q2 when 1 tag"
        print nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2==1 ].astype(bool).sum()
        print "Predicted q1q2 1 tag to LLP0"
        print nTagLLP0_pred_r_q1q2_1T
        print "Wrong by: ", 100*( ( (nTagLLP0_pred_r_q1q2_1T)- nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2==1 ].astype(bool).sum() )/ ( nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2==1 ].astype(bool).sum() ) ), "%"


        print "\n"
        print "True tags to LLP0, q1 q2 when >1 tag"
        print nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2>1 ].astype(bool).sum()
        print "Predicted q1q2 2 tags to LLP0"
        print nTagLLP0_pred_r_q1q2_2T
        print "Wrong by: ", 100*( ( (nTagLLP0_pred_r_q1q2_2T)- nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2>1 ].astype(bool).sum() )/ ( nTagLLP0_r_q1q2[ nTagLLP0_r_q1q2>1 ].astype(bool).sum() ) ), "%"
        '''

        t_nTagLLP1_r_q1 = nTagLLP1_r_q1[nTagLLP1_r_q1>0].astype(bool).sum()
        t_nTagLLP1_r_q1_unc = 100./np.sqrt(nTagLLP1_r_q1[nTagLLP1_r_q1>0].astype(bool).sum())
        t_diff_nTagLLP1_r_q1 = 100*( (nTagLLP1_pred_r_q1-nTagLLP1_r_q1[nTagLLP1_r_q1>0].astype(bool).sum())/ ( nTagLLP1_r_q1[nTagLLP1_r_q1>0].astype(bool).sum() ) )

        t_nTagLLP1_r_q2 = nTagLLP1_r_q2[nTagLLP1_r_q2>0].astype(bool).sum()
        t_nTagLLP1_r_q2_unc = 100./np.sqrt(nTagLLP1_r_q2[nTagLLP1_r_q2>0].astype(bool).sum())
        t_diff_nTagLLP1_r_q2 = 100*( (nTagLLP1_pred_r_q2-nTagLLP1_r_q2[nTagLLP1_r_q2>0].astype(bool).sum())/ ( nTagLLP1_r_q2[nTagLLP1_r_q2>0].astype(bool).sum() ) )

        t_nTagLLP1_r_q1q2_1T = nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2==1 ].astype(bool).sum()
        t_nTagLLP1_r_q1q2_1T_unc = 100./np.sqrt(nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2==1 ].astype(bool).sum())
        t_diff_nTagLLP1_r_q1q2_1T = 100*( ( (nTagLLP1_pred_r_q1q2_1T)- nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2==1 ].astype(bool).sum() )/ ( nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2==1 ].astype(bool).sum() ) )

        t_nTagLLP1_r_q1q2_2T = nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2>1 ].astype(bool).sum()
        t_nTagLLP1_r_q1q2_2T_unc = 100./np.sqrt(nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2>1 ].astype(bool).sum())
        t_diff_nTagLLP1_r_q1q2_2T = 100*( ( (nTagLLP1_pred_r_q1q2_2T)- nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2>1 ].astype(bool).sum() )/ ( nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2>1 ].astype(bool).sum() ) )




        row = [
            b,
            str(t_nTagLLP1_r_q1) + " +- " + str( round( t_nTagLLP1_r_q1_unc,2) ) + "%",
            str(round(nTagLLP1_pred_r_q1,2) ) + " +- " + str( round( 100./np.sqrt(nTagLLP1_pred_r_q1),2) ) + "%",
            str( round(t_diff_nTagLLP1_r_q1, 2) ) + "%",

            str(t_nTagLLP1_r_q2) + " +- " + str( round( t_nTagLLP1_r_q2_unc,2) ) + "%",
            str(round(nTagLLP1_pred_r_q2,2) ) + " +- " + str( round( 100./np.sqrt(nTagLLP1_pred_r_q2),2) ) + "%",
            str( round(t_diff_nTagLLP1_r_q2, 2) ) + "%",

        ]

        table_r1_LLP1.add_row(row)

        row = [
            b,
            str(t_nTagLLP1_r_q1q2_1T) + " +- " + str( round( t_nTagLLP1_r_q1q2_1T_unc,2) ) + "%",
            str(round(nTagLLP1_pred_r_q1q2_1T,2) ) + " +- " + str( round( 100./np.sqrt(nTagLLP1_pred_r_q1q2_1T),2) ) + "%",
            str( round(t_diff_nTagLLP1_r_q1q2_1T, 2) ) + "%",

            str(t_nTagLLP1_r_q1q2_2T) + " +- " + str( round( t_nTagLLP1_r_q1q2_2T_unc,2) ) + "%",
            str(round(nTagLLP1_pred_r_q1q2_2T,2) ) + " +- " + str( round( 100./np.sqrt(nTagLLP1_pred_r_q1q2_2T),2) ) + "%",
            str( round(t_diff_nTagLLP1_r_q1q2_2T, 2) ) + "%",


        ]

        table_r2_LLP1.add_row(row)

        '''
        print "\n"
        print "True tags to LLP1, q1"
        print nTagLLP1_r_q1[nTagLLP1_r_q1>0].astype(bool).sum()
        print "Predicted q1 tags to LLP1"
        print nTagLLP1_pred_r_q1
        print "Wrong by: ", 100*( (nTagLLP1_pred_r_q1-nTagLLP1_r_q1[nTagLLP1_r_q1>0].astype(bool).sum())/ ( nTagLLP1_r_q1[nTagLLP1_r_q1>0].astype(bool).sum() ) ), "%"

        print "\n"
        print "True tags to LLP1, q2"
        print nTagLLP1_r_q2[nTagLLP1_r_q2>0].astype(bool).sum()
        print "Predicted q2 tags to LLP1"
        print nTagLLP1_pred_r_q2
        print "Wrong by: ", 100*( (nTagLLP1_pred_r_q2-nTagLLP1_r_q2[nTagLLP1_r_q2>0].astype(bool).sum())/ ( nTagLLP1_r_q2[nTagLLP1_r_q2>0].astype(bool).sum() ) ), "%"

        print "\n"
        print "True tags to LLP1, q1 q2 when 1 tag"
        print nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2==1 ].astype(bool).sum()
        print "Predicted q1q2 1 tag to LLP1"
        print nTagLLP1_pred_r_q1q2_1T
        print "Wrong by: ", 100*( ( (nTagLLP1_pred_r_q1q2_1T)- nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2==1 ].astype(bool).sum() )/ ( nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2==1 ].astype(bool).sum() ) ), "%"


        print "\n"
        print "True tags to LLP1, q1 q2 when >1 tag"
        print nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2>1 ].astype(bool).sum()
        print "Predicted q1q2 2 tags to LLP1"
        print nTagLLP1_pred_r_q1q2_2T
        print "Wrong by: ", 100*( ( (nTagLLP1_pred_r_q1q2_2T)- nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2>1 ].astype(bool).sum() )/ ( nTagLLP1_r_q1q2[ nTagLLP1_r_q1q2>1 ].astype(bool).sum() ) ), "%"
        '''

        #exit()


        #print "True tags to LLP1"
        #print nTagLLP1_r
        #print nTagLLP1_r.sum()

        #print "Predicted1 tags to LLP1"
        #print nTagLLP1_pred1_r
        #print "Predicted2 tags to LLP1"
        #print nTagLLP1_pred2_r
        #print "Tot predicted tags to LLP1"
        #print nTagLLP1_pred1_r + nTagLLP1_pred2_r


        #old approach
        '''
        r_LLP0_r = LLP0_r[LLP0_resolved]
        r_LLP0_z = LLP0_z[LLP0_resolved]
        r_LLP0_pt1 = LLP0_pt1[LLP0_resolved]
        r_LLP0_eta1 = LLP0_eta1[LLP0_resolved]
        r_LLP0_pt2 = LLP0_pt1[LLP0_resolved]
        r_LLP0_eta2 = LLP0_eta1[LLP0_resolved]

        r_LLP1_r = LLP1_r[LLP1_resolved]
        r_LLP1_z = LLP1_z[LLP1_resolved]
        r_LLP1_pt1 = LLP1_pt1[LLP1_resolved]
        r_LLP1_eta1 = LLP1_eta1[LLP1_resolved]
        r_LLP1_pt2 = LLP1_pt2[LLP1_resolved]
        r_LLP1_eta2 = LLP1_eta2[LLP1_resolved]


        bins_r_resolved =  maps['resolved']['bins_r']
        bins_z_resolved = maps['resolved']['bins_z']
        bins_pt_resolved = maps['resolved']['bins_pt']
        bins_eta_resolved = maps['resolved']['bins_eta']

        #digitize is the right function to use
        r_LLP0_r_idx = np.digitize(r_LLP0_r,maps['resolved']['bins_r'])-1
        r_LLP0_z_idx = np.digitize(r_LLP0_z,maps['resolved']['bins_z'])-1
        r_LLP0_pt1_idx = np.digitize(r_LLP0_pt1,maps['resolved']['bins_pt'])-1
        r_LLP0_eta1_idx = np.digitize(r_LLP0_eta1,maps['resolved']['bins_eta'])-1
        r_LLP0_pt2_idx = np.digitize(r_LLP0_pt2,maps['resolved']['bins_pt'])-1
        r_LLP0_eta2_idx = np.digitize(r_LLP0_eta2,maps['resolved']['bins_eta'])-1

        r_LLP1_r_idx = np.digitize(r_LLP1_r,maps['resolved']['bins_r'])-1
        r_LLP1_z_idx = np.digitize(r_LLP1_z,maps['resolved']['bins_z'])-1
        r_LLP1_pt1_idx = np.digitize(r_LLP1_pt1,maps['resolved']['bins_pt'])-1
        r_LLP1_eta1_idx = np.digitize(r_LLP1_eta1,maps['resolved']['bins_eta'])-1
        r_LLP1_pt2_idx = np.digitize(r_LLP1_pt2,maps['resolved']['bins_pt'])-1
        r_LLP1_eta2_idx = np.digitize(r_LLP1_eta2,maps['resolved']['bins_eta'])-1


        nTagLLP0_pred1_r = 0
        nTagLLP0_pred2_r = 0
        nTagLLP1_pred1_r = 0
        nTagLLP1_pred2_r = 0

        print maps['resolved']['eff_LLP0_q1'][3][1][0][2]
        print maps['resolved']['eff_LLP0_q1'].shape


        for it in range(len(r_LLP0_r_idx)):
            pred1 = maps['resolved']['eff_LLP0_q1'][ r_LLP0_r_idx[it] ][  r_LLP0_z_idx[it] ][  r_LLP0_pt1_idx[it] ][  r_LLP0_eta1_idx[it] ]
            nTagLLP0_pred1_r+= pred1


            pred2 = maps['resolved']['eff_LLP0_q2'][ r_LLP0_r_idx[it] ][  r_LLP0_z_idx[it] ][  r_LLP0_pt2_idx[it] ][  r_LLP0_eta2_idx[it] ]
            if pred2<0:
                print "pred2 negative!"
                print pred2
                #print maps['resolved']['eff_LLP0_q2']
                #exit()
            nTagLLP0_pred2_r+= pred2

        for it in range(len(r_LLP1_r_idx)):
            #print "Ev n. ", it
            #print "actual values"
            #print "r, z, pt, eta"
            #print r_LLP1_r[it], r_LLP1_z[it], r_LLP1_pt1[it], r_LLP1_eta1[it] 
            #print "bin content"
            #print "r ",bins_r_merged
            #print "z ",bins_z_merged
            #print "pt ",bins_pt_merged
            #print "eta ",bins_eta_merged
            pred1 = maps['resolved']['eff_LLP1_q1'][ r_LLP1_r_idx[it] ][  r_LLP1_z_idx[it] ][  r_LLP1_pt1_idx[it] ][  r_LLP1_eta1_idx[it] ]
            nTagLLP1_pred1_r+= pred1
            pred2 = maps['resolved']['eff_LLP1_q2'][ r_LLP1_r_idx[it] ][  r_LLP1_z_idx[it] ][  r_LLP1_pt2_idx[it] ][  r_LLP1_eta2_idx[it] ]
            nTagLLP1_pred2_r+= pred2


        print "\n"
        print "   R e s o l v e d"
        print "\n"

        print "True tags to LLP0"
        #print nTagLLP0_r
        print nTagLLP0_r.sum()

        print "Predicted1 tags to LLP0"
        print nTagLLP0_pred1_r
        print "Predicted2 tags to LLP0"
        print nTagLLP0_pred2_r
        print "Tot predicted tags to LLP0"
        print nTagLLP0_pred1_r + nTagLLP0_pred2_r


        print "True tags to LLP1"
        #print nTagLLP1_r
        print nTagLLP1_r.sum()

        print "Predicted1 tags to LLP1"
        print nTagLLP1_pred1_r
        print "Predicted2 tags to LLP1"
        print nTagLLP1_pred2_r
        print "Tot predicted tags to LLP1"
        print nTagLLP1_pred1_r + nTagLLP1_pred2_r

        '''

        #exit()

    #HERE
    print(table_m)
    print(table_r1_LLP0)
    print(table_r2_LLP0)
    print(table_r1_LLP1)
    print(table_r2_LLP1)


create_maps(sign,skip_even=True)
use_maps(sign,skip_odd=True)
exit()
#time for efficiency maps

########################################
### 
gen_match_q = "( abs(GenBquarks.eta)<1. && GenBquarks.pt>30 )"
gen_match_q0 = "( abs(GenBquarks[0].eta)<1. && GenBquarks[0].pt>30 )"
gen_match_q1 = "( abs(GenBquarks[1].eta)<1. && GenBquarks[1].pt>30 )"
gen_match_q2 = "( abs(GenBquarks[2].eta)<1. && GenBquarks[2].pt>30 )"
gen_match_q3 = "( abs(GenBquarks[3].eta)<1. && GenBquarks[3].pt>30 )"
gen_match_llp0 = "( ( abs(GenBquarks[0].eta)<1. && GenBquarks[0].pt>30 ) || ( abs(GenBquarks[1].eta)<1. && GenBquarks[1].pt>30 ) )"
gen_match_llp1 = "( ( abs(GenBquarks[2].eta)<1. && GenBquarks[2].pt>30 ) || ( abs(GenBquarks[3].eta)<1. && GenBquarks[3].pt>30 ) )"
gen_match_llp0_merged = "( ( abs(GenBquarks[0].eta)<1. && GenBquarks[0].pt>30 ) && ( abs(GenBquarks[1].eta)<1. && GenBquarks[1].pt>30 ) && GenHiggs[0].dRdaughters<0.4 )"
gen_match_llp0_resolved = "( ( abs(GenBquarks[0].eta)<1. && GenBquarks[0].pt>30 ) && ( abs(GenBquarks[1].eta)<1. && GenBquarks[1].pt>30 ) && GenHiggs[0].dRdaughters>=0.4)"

llp0_merged = "(GenHiggs[0].dRdaughters<0.4)"
llp0_merged_Higgs0_acceptance = "(GenHiggs[0].dRdaughters<0.4 && abs(GenHiggs[0].eta)<1. && GenHiggs[0].pt>30)"
llp0_merged_Bquarks_OR_acceptance = "(GenHiggs[0].dRdaughters<0.4 && ( (abs(GenBquarks[0].eta)<1. && GenBquarks[0].pt>30) || (abs(GenBquarks[1].eta)<1. && GenBquarks[1].pt>30) ) )"
llp0_merged_Bquarks_XOR_acceptance = "(GenHiggs[0].dRdaughters<0.4 && ( (abs(GenHiggs[0].eta)<1. && GenHiggs[0].pt>30) || (abs(GenBquarks[1].eta)<1. && GenBquarks[1].pt>30) && !( (abs(GenBquarks[0].eta)<1. && GenBquarks[0].pt>30) && (abs(GenBquarks[1].eta)<1. && GenBquarks[1].pt>30)  ) ) )"
llp0_merged_Bquarks_AND_acceptance = "(GenHiggs[0].dRdaughters<0.4 && ( (abs(GenBquarks[0].eta)<1. && GenBquarks[0].pt>30) && (abs(GenBquarks[1].eta)<1. && GenBquarks[1].pt>30) ) )"

llp0_resolved = "(GenHiggs[0].dRdaughters>=0.4)"


#Gen level acceptance must include also the noisy phi regions when looking at 2017/2018 data

#### MERGED

llp0_merged_Higgs0_acceptance = "(GenHiggs[0].dRdaughters<0.4 && abs(GenHiggs[0].eta)<1. && GenHiggs[0].pt>30)"
llp0_merged_Higgs0_acceptance_1_jet = "(GenHiggs[0].dRdaughters<0.4 && abs(GenHiggs[0].eta)<1. && GenHiggs[0].pt>30 && nJets_to_LLP0>0)"
llp0_merged_Higgs0_acceptance_1_tag_jet = "(GenHiggs[0].dRdaughters<0.4 && abs(GenHiggs[0].eta)<1. && GenHiggs[0].pt>30 && nTagJets_to_LLP0>0)"

#pl(sign,var="nTagJets_to_LLP0",cut_d=llp0_merged_Higgs0_acceptance,cut_n=llp0_merged_Higgs0_acceptance,eff_mode=False,add_label="_Higgs0_acceptance_merged")

pl(sign,var="GenHiggs[0].pt",cut_d=llp0_merged_Higgs0_acceptance,cut_n=llp0_merged_Higgs0_acceptance_1_jet,eff_mode=False,add_label="_Higgs0_acceptance_merged")
exit()

pl(sign,var="GenHiggs[0].pt",cut_d=llp0_merged_Higgs0_acceptance,cut_n=llp0_merged_Higgs0_acceptance_1_jet,eff_mode=True,add_label="_Higgs0_acceptance_merged_1_jet")
pl(sign,var="GenHiggs[0].pt",cut_d=llp0_merged_Higgs0_acceptance,cut_n=llp0_merged_Higgs0_acceptance_1_tag_jet,eff_mode=True,add_label="_Higgs0_acceptance_merged_1_tag_jet")

#pl(sign,var="GenHiggs[0].eta",cut_d=llp0_merged_Higgs0_acceptance,cut_n=llp0_merged_Higgs0_acceptance_1_jet,eff_mode=True,add_label="_Higgs0_acceptance_merged_1_jet")
#pl(sign,var="GenHiggs[0].eta",cut_d=llp0_merged_Higgs0_acceptance,cut_n=llp0_merged_Higgs0_acceptance_1_tag_jet,eff_mode=True,add_label="_Higgs0_acceptance_merged_1_tag_jet")

#pl(sign,var="GenHiggs[0].travelRadiusLLP",cut_d=llp0_merged_Higgs0_acceptance,cut_n=llp0_merged_Higgs0_acceptance_1_jet,eff_mode=True,add_label="_Higgs0_acceptance_merged_1_jet")
#pl(sign,var="GenHiggs[0].travelRadiusLLP",cut_d=llp0_merged_Higgs0_acceptance,cut_n=llp0_merged_Higgs0_acceptance_1_tag_jet,eff_mode=True,add_label="_Higgs0_acceptance_merged_1_tag_jet")

#pl(sign,var="GenHiggs[0].travelZLLP",cut_d=llp0_merged_Higgs0_acceptance,cut_n=llp0_merged_Higgs0_acceptance_1_jet,eff_mode=True,add_label="_Higgs0_acceptance_merged_1_jet")
#pl(sign,var="GenHiggs[0].travelZLLP",cut_d=llp0_merged_Higgs0_acceptance,cut_n=llp0_merged_Higgs0_acceptance_1_tag_jet,eff_mode=True,add_label="_Higgs0_acceptance_merged_1_tag_jet")

exit()

#If b-quarks closer than 0.4: parametrize vs gen Higgs --> merged
pl(sign,var="nTagJets_to_LLP0",cut_d=llp0_merged,cut_n=llp0_merged,eff_mode=False,add_label="_merged")
pl(sign,var="nTagJets_to_LLP0",cut_d=llp0_merged_Higgs0_acceptance,cut_n=llp0_merged_Higgs0_acceptance,eff_mode=False,add_label="_Higgs0_acceptance_merged")
pl(sign,var="nTagJets_to_LLP0",cut_d=llp0_merged_Bquarks_OR_acceptance,cut_n=llp0_merged_Bquarks_OR_acceptance,eff_mode=False,add_label="_Bquarks_OR_acceptance_merged")

pl(sign,var="nTagJets_to_LLP0",cut_d=llp0_merged_Bquarks_XOR_acceptance,cut_n=llp0_merged_Bquarks_XOR_acceptance,eff_mode=False,add_label="_Bquarks_XOR_acceptance_merged")

pl(sign,var="nTagJets_to_LLP0",cut_d=llp0_merged_Bquarks_AND_acceptance,cut_n=llp0_merged_Bquarks_AND_acceptance,eff_mode=False,add_label="_Bquarks_AND_acceptance_merged")
#If DR>0.4: parametrize vs individual gen b quarks --> resolved


'''
#switch n.1: merged/resolved
pl(sign,var="GenHiggs[0].dRdaughters",cut_d=gen_match_llp0,cut_n=gen_match_llp0,eff_mode=False,add_label="_switch_merged_resolved")

#acceptance included
pl(sign,var="nTagJets_to_LLP0",cut_d=gen_match_llp0_merged,cut_n=gen_match_llp0_merged,eff_mode=False,add_label="_acceptance_merged",norm=True)

pl(sign,var="nTagJets_to_LLP0",cut_d=gen_match_llp0_resolved,cut_n=gen_match_llp0_resolved,eff_mode=False,add_label="_acceptance_resolved",norm=True)

pl(sign,var="nJets_to_LLP0",cut_d=gen_match_llp0_merged,cut_n=gen_match_llp0_merged,eff_mode=False,add_label="_acceptance_merged",norm=True)

pl(sign,var="nJets_to_LLP0",cut_d=gen_match_llp0_resolved,cut_n=gen_match_llp0_resolved,eff_mode=False,add_label="_acceptance_resolved",norm=True)

#
pl(sign,var="nTagJets_to_LLP0",cut_d=llp0_merged,cut_n=llp0_merged,eff_mode=False,add_label="_merged",norm=True)

pl(sign,var="nTagJets_to_LLP0",cut_d=llp0_resolved,cut_n=llp0_resolved,eff_mode=False,add_label="_resolved",norm=True)

pl(sign,var="nJets_to_LLP0",cut_d=llp0_merged,cut_n=llp0_merged,eff_mode=False,add_label="_merged",norm=True)

pl(sign,var="nJets_to_LLP0",cut_d=llp0_resolved,cut_n=llp0_resolved,eff_mode=False,add_label="_resolved",norm=True)

exit()
pl(sign,var="Jets.ptGen",cut_d="( " + gen_match_q+" && Jets.isGenMatched && (Jets.ptGen == GenBquarks.pt) )",cut_n="( " + gen_match_q+" && Jets.isGenMatched && (Jets.ptGen == GenBquarks.pt) )",eff_mode=False,add_label="_LLP_bquark_eta1p0")

#Eff
pl(sign,var="dR_LLPs",cut_d="( (abs(GenBquarks[0].eta)<1. && GenBquarks[0].pt>30) || (abs(GenBquarks[1].eta)<1. && GenBquarks[1].pt>30))",cut_n="( (abs(GenBquarks[0].eta)<1. && GenBquarks[0].pt>30) || (abs(GenBquarks[1].eta)<1. && GenBquarks[1].pt>30)) && ( (nTagJets_to_LLP0==1 && nTagJets_to_LLP1==0) || (nTagJets_to_LLP0==0 && nTagJets_to_LLP1==1))",eff_mode=True,add_label="_LLP_bquark_eta1p0_1_tag_XOR")
'''
