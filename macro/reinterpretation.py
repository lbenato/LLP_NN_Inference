#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
import uproot
import root_numpy

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
(options, args) = parser.parse_args()
##if options.bash: 
gROOT.SetBatch(True)
gStyle.SetOptStat(0)

from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.reinterpretation_variables import *

ERA = "2017"
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

sign = [
    "SUSY_mh1000_ctau500_HH",
    #"SUSY_mh400_ctau500_HH",
    #"SUSY_mh127_ctau500_HH",
    #"SUSY_mh1000_ctau3000_HH",
    #"SUSY_mh400_ctau3000_HH",
    #"SUSY_mh127_ctau3000_HH",
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

def maps(sign, tree_name="tree",var="Jets.sigprob",cut_d="",cut_n="",do_radius=False,add_label="",eff_mode=True,max_eff=1.01):

    '''
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
    '''


    list_of_variables = ["nTagJets_0p996","isMC","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB","HLT*","MT","pt","MinJet*DPhi*","nCHSJets*","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","TriggerWeight","PUWeight","PUReWeight","GenLLPs.*","MeanNumInteractions","HT","Jets.nTrackConstituents","isLLP*","nLLP*","nJets*","nTagJets*","GenLLPs*","GenBquarks*","GenHiggs*"]

    #list_of_variables_data = list_of_variables  + ["dt_ecal_dist","min_dPhi_jets*"]

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
    chunk_size = 100000
    for b in sign:
        for i, ss in enumerate(samples[b]['files']):
            print "Uproot iterating over ", ss, " . . . "
            filename = NTUPLEDIR + ss + '.root'
            gen = uproot.iterate(filename,"tree",list_of_variables)
            for arrays in gen:
                
                n_events = -1

                #merged vs resolved
                genLLP0_merged = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][0:n_events] )))[0:1] < 0.4)
                genLLP1_merged = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][0:n_events] )))[1:2] < 0.4)

                genLLP0_resolved = np.logical_not(genLLP0_merged)
                genLLP1_resolved = np.logical_not(genLLP1_merged)


                genLLP0_r = np.transpose(np.stack(( arrays["GenLLPs.travelRadius"][0:n_events]/100. )))[0:1]
                genLLP1_r = np.transpose(np.stack(( arrays["GenLLPs.travelRadius"][0:n_events]/100. )))[1:2]

                genLLP0_z = np.transpose(np.stack(( arrays["GenLLPs.travelZ"][0:n_events]/100. )))[0:1]
                genLLP1_z = np.transpose(np.stack(( arrays["GenLLPs.travelZ"][0:n_events]/100. )))[1:2]

                genh0_pt = np.transpose(np.stack(( arrays["GenHiggs.pt"][0:n_events] )))[0:1]
                genh1_pt = np.transpose(np.stack(( arrays["GenHiggs.pt"][0:n_events] )))[1:2]

                genh0_eta = np.transpose(np.stack(( arrays["GenHiggs.eta"][0:n_events] )))[0:1]
                genh1_eta = np.transpose(np.stack(( arrays["GenHiggs.eta"][0:n_events] )))[1:2]

                genb0_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][0:n_events] )))[0:1]
                genb1_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][0:n_events] )))[1:2]
                genb2_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][0:n_events] )))[2:3]
                genb3_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][0:n_events] )))[3:]

                genb0_eta = np.transpose(np.stack(( arrays["GenBquarks.eta"][0:n_events] )))[0:1]
                genb1_eta = np.transpose(np.stack(( arrays["GenBquarks.eta"][0:n_events] )))[1:2]
                genb2_eta = np.transpose(np.stack(( arrays["GenBquarks.eta"][0:n_events] )))[2:3]
                genb3_eta = np.transpose(np.stack(( arrays["GenBquarks.eta"][0:n_events] )))[3:]


                #acceptance masks
                #merged
                acc_LLP0_merged_per_part = np.logical_and( np.logical_and(genh0_pt>30.,abs(genh0_eta)<1.) , genLLP0_merged)
                acc_LLP1_merged_per_part = np.logical_and( np.logical_and(genh1_pt>30.,abs(genh1_eta)<1.) , genLLP1_merged)

                acc_LLP0_merged = acc_LLP0_merged_per_part.reshape(acc_LLP0_merged_per_part.shape[1])
                acc_LLP1_merged = acc_LLP1_merged_per_part.reshape(acc_LLP1_merged_per_part.shape[1])


                acc_LLP0_resolved_per_part = np.logical_and( np.logical_or( np.logical_and(genb0_pt>30.,abs(genb0_eta)<1.), np.logical_and(genb1_pt>0.,abs(genb1_eta)<1.)), genLLP0_resolved)
                acc_LLP1_resolved_per_part = np.logical_and( np.logical_or( np.logical_and(genb2_pt>30.,abs(genb2_eta)<1.), np.logical_and(genb3_pt>0.,abs(genb3_eta)<1.)), genLLP1_resolved)

                acc_LLP0_resolved = acc_LLP0_resolved_per_part.reshape(acc_LLP0_resolved_per_part.shape[1])
                acc_LLP1_resolved = acc_LLP1_resolved_per_part.reshape(acc_LLP1_resolved_per_part.shape[1])

                #print acc_LLP0_merged_per_part
                #print acc_LLP1_merged_per_part
                #print acc_LLP0_merged
                #print acc_LLP1_merged

                zero_acceptance_merged = np.logical_and(acc_LLP0_merged==False, acc_LLP1_merged==False)
                one_acceptance_merged  = np.logical_xor(acc_LLP0_merged==True, acc_LLP1_merged==True)
                two_acceptance_merged  = np.logical_and(acc_LLP0_merged==True, acc_LLP1_merged==True)

                zero_acceptance_resolved = np.logical_and(acc_LLP0_resolved==False, acc_LLP1_resolved==False)
                one_acceptance_resolved  = np.logical_xor(acc_LLP0_resolved==True, acc_LLP1_resolved==True)
                two_acceptance_resolved  = np.logical_and(acc_LLP0_resolved==True, acc_LLP1_resolved==True)

                one_resolved_one_merged  = np.logical_and(one_acceptance_resolved==True, one_acceptance_merged==True)

                #print zero_acceptance
                #print "one_acceptance_merged[one_acceptance_merged]"
                #print one_acceptance_merged[one_acceptance_merged]
                #print "two_acceptance_merged[two_acceptance_merged]"
                #print two_acceptance_merged[two_acceptance_merged]

                #print "one_acceptance_resolved[one_acceptance_resolved]"
                #print one_acceptance_resolved[one_acceptance_resolved]
                #print "two_acceptance_resolved[two_acceptance_resolved]"
                #print two_acceptance_resolved[two_acceptance_resolved]

                #print "one_resolved_one_merged[one_resolved_one_merged]"
                #print one_resolved_one_merged[one_resolved_one_merged]

                #print genh0_pt[acc_LLP0_merged_per_part]
                #print genh1_pt[acc_LLP1_merged_per_part]
                #print genh0_eta[acc_LLP0_merged_per_part]
                #print genh1_eta[acc_LLP1_merged_per_part]

                print "LLP r and z merged"
                print genLLP0_r[acc_LLP0_merged_per_part]
                #print genLLP1_r[acc_LLP1_merged_per_part]

                print genLLP0_z[acc_LLP0_merged_per_part]
                #print genLLP1_z[acc_LLP1_merged_per_part]

                sorted_genLLP0_r_merged = np.sort(genLLP0_r[acc_LLP0_merged_per_part])
                sorted_genLLP1_r_merged = np.sort(genLLP1_r[acc_LLP1_merged_per_part])

                sorted_genLLP0_z_merged = np.sort(genLLP0_z[acc_LLP0_merged_per_part])
                sorted_genLLP1_z_merged = np.sort(genLLP1_z[acc_LLP1_merged_per_part])

                sorted_genLLP0_pt_merged = np.sort(genh0_pt[acc_LLP0_merged_per_part])
                sorted_genLLP1_pt_merged = np.sort(genh1_pt[acc_LLP1_merged_per_part])

                sorted_genLLP0_eta_merged = np.sort(genh0_eta[acc_LLP0_merged_per_part])
                sorted_genLLP1_eta_merged = np.sort(genh1_eta[acc_LLP1_merged_per_part])



                print "split sorted_genLLP0_r_merged"
                print np.array_split(sorted_genLLP0_r_merged,4)[0]
                print np.array_split(sorted_genLLP0_r_merged,4)[0].shape
                print np.array_split(sorted_genLLP0_r_merged,4)[1]
                print np.array_split(sorted_genLLP0_r_merged,4)[1].shape
                print np.array_split(sorted_genLLP0_r_merged,4)[2]
                print np.array_split(sorted_genLLP0_r_merged,4)[2].shape
                print np.array_split(sorted_genLLP0_r_merged,4)[3]
                print np.array_split(sorted_genLLP0_r_merged,4)[3].shape

                '''
                print "split sorted_genLLP0_z_merged"
                print np.array_split(sorted_genLLP0_z_merged,4)[0]
                print np.array_split(sorted_genLLP0_z_merged,4)[1]
                print np.array_split(sorted_genLLP0_z_merged,4)[2]
                print np.array_split(sorted_genLLP0_z_merged,4)[3]


                print "split sorted_genLLP0_pt_merged"
                print np.array_split(sorted_genLLP0_pt_merged,4)[0]
                print np.array_split(sorted_genLLP0_pt_merged,4)[0].shape
                print np.array_split(sorted_genLLP0_pt_merged,4)[1]
                print np.array_split(sorted_genLLP0_pt_merged,4)[1].shape
                print np.array_split(sorted_genLLP0_pt_merged,4)[2]
                print np.array_split(sorted_genLLP0_pt_merged,4)[2].shape
                print np.array_split(sorted_genLLP0_pt_merged,4)[3]
                print np.array_split(sorted_genLLP0_pt_merged,4)[3].shape

                
                print "split sorted_genLLP0_eta_merged"
                print np.array_split(sorted_genLLP0_eta_merged,4)[0]
                print np.array_split(sorted_genLLP0_eta_merged,4)[0].shape
                print np.array_split(sorted_genLLP0_eta_merged,4)[1]
                print np.array_split(sorted_genLLP0_eta_merged,4)[1].shape
                print np.array_split(sorted_genLLP0_eta_merged,4)[2]
                print np.array_split(sorted_genLLP0_eta_merged,4)[2].shape
                print np.array_split(sorted_genLLP0_eta_merged,4)[3]
                print np.array_split(sorted_genLLP0_eta_merged,4)[3].shape
                '''

                #First bin guess
                bins_eta_merged = np.array([-1., -0.4,  0.,   0.4,    1.])
                bins_pt_merged = np.array( [0.,  765.,  900., 1100.,  5000.])
                bins_z_merged = np.array(  [-5., -0.15, 0.,   0.15,   5.])
                bins_r_merged = np.array(  [0.,  0.1,   0.3,  0.6,    10.])

                #Now we need the 4D maps

                exit()
                print "LLP r and z resolved"
                print genLLP0_r[acc_LLP0_resolved_per_part]
                print genLLP1_r[acc_LLP1_resolved_per_part]

                print genLLP0_z[acc_LLP0_resolved_per_part]
                print genLLP1_z[acc_LLP1_resolved_per_part]

                exit()

                #resolved

                acc_LLP0 = np.logical_or( np.logical_and(genb0_pt>30.,abs(genb0_eta)<1.), np.logical_and(genb1_pt>0.,abs(genb1_eta)<1.))
                acc_LLP1 = np.logical_or( np.logical_and(genb2_pt>30.,abs(genb2_eta)<1.), np.logical_and(genb3_pt>0.,abs(genb3_eta)<1.))
                acc_LLP0 = acc_LLP0.reshape(acc_LLP0.shape[1])
                acc_LLP1 = acc_LLP1.reshape(acc_LLP1.shape[1])

                print acc_LLP0
                print acc_LLP1
                zero_acceptance = np.logical_and(acc_LLP0==False, acc_LLP1==False)
                one_acceptance  = np.logical_xor(acc_LLP0==True, acc_LLP1==True)
                two_acceptance  = np.logical_and(acc_LLP0==True, acc_LLP1==True)
                print zero_acceptance
                print one_acceptance
                print two_acceptance

                nTagJets_to_LLP0 = arrays["nTagJets_to_LLP0"][0:n_events]
                nTagJets_to_LLP1 = arrays["nTagJets_to_LLP1"][0:n_events]

                #n. tags per LLP masks

                exit()


maps(sign)
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
