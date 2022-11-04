#! /usr/bin/env python

import os, multiprocessing
import copy
import math
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, gPad
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH1D, TH2F, THStack, TGraph, TGraphAsymmErrors
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TBox, TGaxis, TAxis
from decimal import Decimal, ROUND_UP

#### IMPORT SAMPLES AND VARIABLES DICTIONARIES ####

#from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
#from NNInferenceCMSSW.LLP_NN_Inference.variables_default import *
from NNInferenceCMSSW.LLP_NN_Inference.variables import *
#from NNInferenceCMSSW.LLP_NN_Inference.skimmed_variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *

##################
#    PROJECT     #
##################

def project(samples,var, cut, cut_s, cut_d, weight, samplelist, pd, ntupledir, treename="ntuple/tree", formula="", alpha=1.):
#def project(var, cut, cut_s, weight, samplelist, pd, ntupledir, treename="trigger/tree"):
    # Create dict
    file = {}
    tree = {}
    chain = {}
    hist = {}
    
    ### Create and fill MC histograms ###
    for i, s in enumerate(samplelist):
        
        if "HIST" in cut: # Histogram written to file
            for j, ss in enumerate(samples[s]['files']):
                file[ss] = TFile(ntupledir + ss + ".root", "READ")
                hist[s] = file[ss].Get("ntuple/" + histDir[var[0:2]] + "/" + var) if not s in hist else hist[s].Add(file[ss].Get("ntuple/" + histDir[var[0:2]] + "/" + var))
        else: # Project from tree
            chain[s] = TChain(treename)
            for j, ss in enumerate(samples[s]['files']):
                if not 'data' in s or ('data' in s and ss in pd):
                    chain[s].Add(ntupledir + ss + ".root")
                    #print "Sample: ", ss
                    #print "current weight of chain: ", chain[s].GetWeight()
                    #filename = TFile(ntupledir + ss +'.root')
                    #tree = filename.Get("ntuple/tree") 
                    #print "real weight: ", tree.GetWeight()
                    #chain[s].SetWeight(tree.GetWeight(),"global")
                    #print "forcing weight of chain: ", chain[s].GetWeight()
            if variable[var]['nbins']>0: hist[s] = TH1D(s, ";"+variable[var]['title'], variable[var]['nbins'], variable[var]['min'], variable[var]['max']) # Init histogram
            else: hist[s] = TH1D(s, ";"+variable[var]['title'], len(variable[var]['bins'])-1, array('f', variable[var]['bins']))
            hist[s].Sumw2()
            tmpcut = cut
            tmpcut_s = cut_s
            tmpcut_d = cut_d

            #Scale up QCD
            #if s=="QCD":
            #    weight += "*( HT<400 ? ( HT<300 ? 0.6 : 0.7) :  ( HT>400 && HT<500 ? 0.9  :  1.1 ) )" 

            cutstring = "("+weight+")" + ("*("+tmpcut+")" if len(tmpcut)>0 else "")
            cutstring_s = "("+weight+")" + ("*("+tmpcut_s+")" if len(tmpcut_s)>0 else "")
            cutstring_d = "("+weight+")" + ("*("+tmpcut_d+")" if len(tmpcut_d)>0 else "")
            if "(" in formula:
                var_updated = formula+"("+var+"))"
            else:
                var_updated = formula+"("+var+")"

            if "VBFH_M" in s:#important bugfix! Not applying jet matching to signal!
                chain[s].Project(s, var, cutstring_s) if formula=="" else chain[s].Project(s, var_updated, cutstring_s)
            elif "ggH_M" in s:#important bugfix! Not applying jet matching to signal!
                chain[s].Project(s, var, cutstring_s) if formula=="" else chain[s].Project(s, var_updated, cutstring_s)
            elif "ZH_M" in s:#important bugfix! Not applying jet matching to signal!
                chain[s].Project(s, var, cutstring_s) if formula=="" else chain[s].Project(s, var_updated, cutstring_s)
            elif "SUSY" in s:#important bugfix! Not applying jet matching to signal!
                chain[s].Project(s, var, cutstring_s) if formula=="" else chain[s].Project(s, var_updated, cutstring_s)
            elif "Event" in s:#important bugfix! Not applying jet matching to signal!
                chain[s].Project(s, var, cutstring_s) if formula=="" else chain[s].Project(s, var_updated, cutstring_s)
            elif "HEM" in s:#important bugfix! Not applying jet matching to signal!
                chain[s].Project(s, var, cutstring_s) if formula=="" else chain[s].Project(s, var_updated, cutstring_s)
            elif "SMS" in s:#important bugfix! Not applying jet matching to signal!
                chain[s].Project(s, var, cutstring_s) if formula=="" else chain[s].Project(s, var_updated, cutstring_s)

            #elif ("MET" in s):#important bugfix! Not applying jet matching to signal!
            #    chain[s].Project(s, var, cutstring_d) if formula=="" else chain[s].Project(s, var_updated, cutstring_d)
            elif ("JetHTMC" in s):#important bugfix! Not applying jet matching to signal!
                chain[s].Project(s, var, cutstring_d) if formula=="" else chain[s].Project(s, var_updated, cutstring_d)
            #elif ("BH" in s):#important bugfix! Not applying jet matching to signal!
            #    chain[s].Project(s, var, cutstring_s) if formula=="" else chain[s].Project(s, var_updated, cutstring_d)
            else:
                chain[s].Project(s, var, cutstring) if formula=="" else chain[s].Project(s, var_updated, cutstring)
                print s, " percentage passing :", 100*float(chain[s].GetEntries(cutstring))/float(chain[s].GetEntries()) if chain[s].GetEntries()>0 else 0, " entries"
                if s=="HighMETBH":
                    chain[s].Project(s, var, cutstring_d) if formula=="" else chain[s].Project(s, var_updated, cutstring_d)
                if s=="HighMETCopy":
                    chain[s].Project(s, var, cutstring_d) if formula=="" else chain[s].Project(s, var_updated, cutstring_d)
                    
            #print s, " has total of ", chain[s].GetEntries(), " entries"
            #print s, " with cut: ", chain[s].GetEntries(tmpcut), " entries"
            #print s, " with weight and cut: ", chain[s].GetEntries(cutstring), " entries"
            #print s, " histogram ", hist[s].Integral(), " entries"

            hist[s].SetOption("%s" % chain[s].GetTree().GetEntriesFast())
            hist[s].Scale(samples[s]['weight'] if hist[s].Integral() >= 0 else 0)
            #if s in sign:
                #print "Is it empty?"
                
            #print "dont normalize!"
            print "normalize!"
            ##print "Entries: ", s, hist[s].GetEntries()
            ##print "Integral pre: ", s, hist[s].Integral()
            if hist[s].Integral()>0:
                hist[s].Scale(1./hist[s].Integral())
            ##print "Integral: ", s, hist[s].Integral()

        hist[s].SetFillColorAlpha(samples[s]['fillcolor'],alpha)
        hist[s].SetFillStyle(samples[s]['fillstyle'])
        hist[s].SetLineColor(samples[s]['linecolor'])
        hist[s].SetLineStyle(samples[s]['linestyle'])
    
    if "HIST" in cut: hist["files"] = file
    return hist


##################
#      DRAW      #
##################

def draw(samples, hist, data, back, sign, snorm=1, ratio=0, poisson=False, log=False, data_tag='data_obs'):
    # If not present, create BkgSum
    if not 'BkgSum' in hist.keys():
        hist['BkgSum'] = hist[data_tag].Clone("BkgSum") if data_tag in hist else hist[back[0]].Clone("BkgSum")
        hist['BkgSum'].Reset("MICES")
        for i, s in enumerate(back): hist['BkgSum'].Add(hist[s])
    hist['BkgSum'].SetMarkerStyle(0)

    # Some style
    for i, s in enumerate(data):
        hist[s].SetMarkerStyle(21)
        hist[s].SetMarkerSize(1.25)
    for i, s in enumerate(sign):
        hist[s].SetLineWidth(3)
        hist[s].SetLineStyle(samples[s]['linestyle'])

        
    for i, s in enumerate(data+back+sign+['BkgSum']):
        addOverflow(hist[s], False) # Add overflow # False before
    
    # Set Poisson error bars
    #if len(data) > 0: hist['data_obs'].SetBinErrorOption(1) # doesn't work
    
    # Poisson error bars for data
    if poisson:
        alpha = 1 - 0.6827
        hist[data_tag].SetBinErrorOption(TH1.kPoisson)
        data_graph = TGraphAsymmErrors(hist[data_tag].GetNbinsX())
        data_graph.SetMarkerStyle(hist[data_tag].GetMarkerStyle())
        data_graph.SetMarkerSize(hist[data_tag].GetMarkerSize())
        res_graph = data_graph.Clone()
        for i in range(hist[data_tag].GetNbinsX()):
            N = hist[data_tag].GetBinContent(i+1)
            B = hist['BkgSum'].GetBinContent(i+1)
            L =  0 if N==0 else ROOT.Math.gamma_quantile(alpha/2,N,1.)
            U =  ROOT.Math.gamma_quantile_c(alpha/2,N+1,1)
            data_graph.SetPoint(i, hist[data_tag].GetXaxis().GetBinCenter(i+1), N if not N==0 else -1.e99)
            data_graph.SetPointError(i, hist[data_tag].GetXaxis().GetBinWidth(i+1)/2., hist[data_tag].GetXaxis().GetBinWidth(i+1)/2., N-L, U-N)
            res_graph.SetPoint(i, hist[data_tag].GetXaxis().GetBinCenter(i+1), N/B if not B==0 and not N==0 else -1.e99)
            res_graph.SetPointError(i, hist[data_tag].GetXaxis().GetBinWidth(i+1)/2., hist[data_tag].GetXaxis().GetBinWidth(i+1)/2., (N-L)/B if not B==0 else -1.e99, (U-N)/B if not B==0 else -1.e99)
    
    
    # Create stack
    bkg = THStack("Bkg", ";"+hist['BkgSum'].GetXaxis().GetTitle()+";Events")
    for i, s in enumerate(back): bkg.Add(hist[s])
    
    # Legend
    n = len([x for x in data+back+['BkgSum']+sign if samples[x]['plot']])
    for i, s in enumerate(sign):
        if 'sublabel' in samples[s]: n+=1
        if 'subsublabel' in samples[s]: n+=1
    #leg = TLegend(0.68, 0.9-0.05*n, 0.93, 0.9)
    #leg = TLegend(0.68-0.13, 0.9-0.05*n, 0.93, 0.9)#DCMS
    leg = TLegend(0.65, 0.9-0.05*n, 0.95, 0.9)#new, more rectangular
    leg.SetTextSize(0.03)#DCMS
    if n<4:
        leg = TLegend(0.65, 0.6, 0.95, 0.8)#more readable for less entries
        leg = TLegend(0.65, 0.6-0.05, 0.95, 0.8-0.05)#more readable for less entries
        leg.SetTextSize(0.04)#DCMS
    leg.SetBorderSize(0)
    leg.SetFillStyle(0) #1001
    leg.SetFillColor(0)
    #if len(sign)>0:
    #    leg.SetHeader("Signal x-sec=%.0f pb"%(1*snorm))
    if len(data) > 0:
        leg.AddEntry(hist[data[0]], samples[data[0]]['label'], "ple1")
    for i, s in reversed(list(enumerate(['BkgSum']+back))):
        leg.AddEntry(hist[s], samples[s]['label'], "f")    
    for i, s in enumerate(sign):
        leg.AddEntry(hist[s], samples[s]['label'], "f")

    
    # --- Display ---
    c1 = TCanvas("c1", hist.values()[-1].GetXaxis().GetTitle(), 1200, 800 if ratio else 800)#else  700
    
    if ratio:
        c1.Divide(1, 2)
        setTopPad(c1.GetPad(1), ratio)
        setBotPad(c1.GetPad(2), ratio)
    c1.cd(1)
    c1.GetPad(bool(ratio)).SetTopMargin(0.06)
    c1.GetPad(bool(ratio)).SetRightMargin(0.05)
    c1.GetPad(bool(ratio)).SetTicks(1, 1)
    if log:
        c1.GetPad(bool(ratio)).SetLogy()
        #c1.GetPad(bool(ratio)).SetLogx()
        
    # Draw
    bkg.Draw("HIST") # stack
    #a = bkg.GetXaxis()
    #a.ChangeLabel(1,-1,-1,-1,-1,-1," ");
    #a.ChangeLabel(3,-1,-1,-1,-1,-1," ");
    #a.ChangeLabel(5,-1,-1,-1,-1,-1," ");
    #a.ChangeLabel(2,-1,-1,-1,-1,-1,"0 tag");
    #a.ChangeLabel(4,-1,-1,-1,-1,-1,"1 tag");
    #a.ChangeLabel(6,-1,-1,-1,-1,-1,"#geq 2 tags");
    #a.ChangeLabel(-1,-1,-1,-1,-1,-1," ");
    hist['BkgSum'].Draw("SAME, E2") # sum of bkg

    if poisson: data_graph.Draw("SAME, PE")
    elif len(data) > 0: hist[data_tag].Draw("SAME, PE")
    for i, s in enumerate(sign):
        if samples[s]['plot']:
            hist[s].DrawNormalized("SAME, HIST", hist[s].Integral()*snorm) # signals


    # Determine range
    if data_tag in hist:
        print "Skip data range y axis assignement of min max"
        bkg.SetMaximum((2.5 if log else 1.2)*max(bkg.GetMaximum(), hist[data_tag].GetBinContent(hist[data_tag].GetMaximumBin())+hist[data_tag].GetBinError(hist[data_tag].GetMaximumBin())))
        bkg.SetMinimum(max(min(hist['BkgSum'].GetBinContent(hist['BkgSum'].GetMinimumBin()), hist[data_tag].GetMinimum()), 5.e-1)  if log else 0.)
        bkg.SetMinimum(5.e-1)#!!

        #This is a good range for normalized stuff such as beam halo
        #
        #print "max bkg and data"
        #print hist['BkgSum'].GetMaximum()
        #print hist[data_tag].GetMaximum()
        #print "min bkg and data"
        #print hist['BkgSum'].GetMinimum()
        #print hist[data_tag].GetMinimum()
        
        
        bkg.SetMaximum(max(hist['BkgSum'].GetMaximum(),hist[data_tag].GetMaximum())*1.1)#5)#!!
        if min(hist['BkgSum'].GetMinimum(),hist[data_tag].GetMinimum())>0:
            bkg.SetMinimum(min(hist['BkgSum'].GetMinimum(),hist[data_tag].GetMinimum())*0.9)#!!
        else:#new
            if max(hist['BkgSum'].GetMinimum(),hist[data_tag].GetMinimum())>0:
                bkg.SetMinimum(max(hist['BkgSum'].GetMinimum(),hist[data_tag].GetMinimum()))
            else:
                bkg.SetMinimum(0.001)
        
        ##bkg.GetYaxis().SetTitleOffset(bkg.GetYaxis().GetTitleOffset()*1.075)
        

    else:
        #bkg.SetMaximum(bkg.GetMaximum()*(2.5 if log else 1.2))
        #bkg.SetMinimum(5.e-1 if log else 0.)
        #bkg.SetMinimum(5.e-1)#!!
        bkg.GetYaxis().SetTitleOffset(bkg.GetYaxis().GetTitleOffset()*1.075)
    if log:
        bkg.GetYaxis().SetNoExponent(bkg.GetMaximum() < 1.e4)
        bkg.GetYaxis().SetMoreLogLabels(True)
        print "Wanna change thiiiis"
        #bkg.SetMinimum(5.e-7)#!!
        bkg.SetMinimum(5.e-3)#!!
        hist[data_tag].SetMinimum(5.e-3)#!!
        bkg.GetYaxis().SetTitleOffset(bkg.GetYaxis().GetTitleOffset()*1.075)



    #w#bkg.SetMaximum(2.e7)
    leg.Draw()
    #drawCMS(LUMI, "Preliminary")
    #drawRegion(channel)
    #drawAnalysis("LL")
    
    setHistStyle(bkg, 1.2 if ratio else 1.1)
    setHistStyle(hist['BkgSum'], 1.2 if ratio else 1.1)


    if ratio:
        c1.cd(2)
        err = hist['BkgSum'].Clone("BkgErr;")
        err.SetTitle("")
        err.GetYaxis().SetTitle("Data / Bkg")
        for i in range(1, err.GetNbinsX()+1):
            err.SetBinContent(i, 1)
            if hist['BkgSum'].GetBinContent(i) > 0:
                err.SetBinError(i, hist['BkgSum'].GetBinError(i)/hist['BkgSum'].GetBinContent(i))
        setBotStyle(err)
        errLine = err.Clone("errLine")
        errLine.SetLineWidth(2)
        errLine.SetFillStyle(0)
        errLine.SetLineColor(2)#L#
        errLine.SetLineStyle(2)#L#
        #err.GetXaxis().SetLabelOffset(err.GetXaxis().GetLabelOffset()*5)
        #err.GetXaxis().SetTitleOffset(err.GetXaxis().GetTitleOffset()*2)
        err.Draw("E2")
        errLine.Draw("SAME, HIST")
        if data_tag in hist:
            res = hist[data_tag].Clone("Residues")
            for i in range(0, res.GetNbinsX()+1):
                if hist['BkgSum'].GetBinContent(i) > 0: 
                    res.SetBinContent(i, res.GetBinContent(i)/hist['BkgSum'].GetBinContent(i))
                    res.SetBinError(i, res.GetBinError(i)/hist['BkgSum'].GetBinContent(i))
            setBotStyle(res)
            if poisson: res_graph.Draw("SAME, PE0")
            else: res.Draw("SAME, PE0")
            if len(err.GetXaxis().GetBinLabel(1))==0: # Bin labels: not a ordinary plot
                drawRatio(hist[data_tag], hist['BkgSum'])
                drawKolmogorov(hist[data_tag], hist['BkgSum'])
                #drawRatio(hist["SingleMuonBH"], hist['BkgSum'])
                #drawKolmogorov(hist["SingleMuonBH"], hist['BkgSum'])
        else: res = None
    c1.Update()
    

    # return list of objects created by the draw() function
    return [c1, bkg, leg, err if ratio else None, errLine if ratio else None, res if ratio else None, data_graph if poisson else None, res_graph if poisson else None]




def drawSignal(samples, hist, sign, log=False, logx=False):
    
    # Legend
    n = len(sign)
    leg = TLegend(0.6, 0.9-0.05*n, 0.9, 0.9)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0) #1001
    leg.SetFillColor(0)
    for i, s in enumerate(sign): leg.AddEntry(hist[s], samples[s]['label'], "fl")
    
    
    # --- Display ---
    c1 = TCanvas("c1", hist.values()[-1].GetXaxis().GetTitle(), 800, 600)
    
    c1.cd(1)
    c1.GetPad(0).SetTopMargin(0.06)
    c1.GetPad(0).SetRightMargin(0.05)
    c1.GetPad(0).SetTicks(1, 1)
    if log:
        c1.GetPad(0).SetLogy()
        #c1.GetPad(0).SetLogx()

    # Draw
    for i, s in enumerate(sign): 
        hist[s+'Err'] = hist[s].Clone(s+'Err')
        hist[s+'Err'].Reset("MICES")
        hist[s+'Err'].Add(hist[s])
        hist[s+'Err'].SetMarkerStyle(0)
        hist[s+'Err'].SetFillStyle(3003)
        hist[s+'Err'].SetFillColor(samples[s]['fillcolor'] if samples[s]['fillcolor']!=0 else 922)

    max_val = 0
    for i, s in enumerate(sign): 
        hist[s].SetLineWidth(3)
        hist[s].SetLineStyle(samples[s]['linestyle'])
        hist[s].Draw("SAME, HIST" if i>0 else "HIST") # signals
        #hist[s].Draw("SAME, PE" if i>0 else "PE") # signals
        max_val = max(max_val,hist[s].GetMaximum())
        addOverflow(hist[s], True) # Add overflow
        hist[s+'Err'].Draw("SAMES,E2")
        #hist[s].SetMinimum(0.01)#(0.01)
    #?#hist[sign[0]].GetXaxis().SetRangeUser(0., 1500)
    #?hist[sign[0]].GetYaxis().SetTitleOffset(hist[sign[-1]].GetYaxis().GetTitleOffset()*1.075)
    #?hist[sign[0]].SetMaximum(max(hist[sign[0]].GetMaximum(), hist[sign[-1]].GetMaximum())*1.25)
    #?hist[sign[0]].SetMinimum(0.)

    hist[sign[0]].GetYaxis().SetTitleOffset(hist[sign[-1]].GetYaxis().GetTitleOffset()*1.075)
    hist[sign[0]].SetMaximum(max_val*1.25)
    #hist[sign[0]].SetMinimum(0.)
    
    if log:
        hist[sign[0]].GetYaxis().SetNoExponent(hist[sign[0]].GetMaximum() < 1.e4)
        hist[sign[0]].GetYaxis().SetMoreLogLabels(True)

    if log:
        c1.GetPad(0).SetLogy()
        #c1.GetPad(0).SetLogx()
    if logx:
        c1.GetPad(0).SetLogx()

    
    leg.Draw()
    #drawCMS(LUMI, "Preliminary")
    
    c1.Update()
    
    # return list of objects created by the draw() function
    return [c1, leg]

def drawRatio(data, bkg):
    errData = array('d', [1.0])
    errBkg = array('d', [1.0])
    intData = data.IntegralAndError(1, data.GetNbinsX(), errData)
    intBkg = bkg.IntegralAndError(1, bkg.GetNbinsX(), errBkg)
    ratio = intData / intBkg if intBkg!=0 else 0.
    error = math.hypot(errData[0]*ratio/intData,  errBkg[0]*ratio/intBkg) if intData>0 and intBkg>0 else 0
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextColor(1)
    latex.SetTextFont(62)
    latex.SetTextSize(0.085)
    latex.DrawLatex(0.25, 0.85, "Data/Bkg = %.3f #pm %.3f" % (ratio, error))
    print "  Ratio:\t%.3f +- %.3f" % (ratio, error)
    #return [ratio, error]

def drawKolmogorov(data, bkg, fontsize=0.085):
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextColor(1)
    latex.SetTextFont(62)
    latex.SetTextSize(fontsize)
    latex.DrawLatex(0.55, 0.85, "#chi^{2}/ndf = %.2f,   K-S = %.3f" % (data.Chi2Test(bkg, "CHI2/NDF"), data.KolmogorovTest(bkg)))

def printTable(samples, hist, sign=[], SIGNAL=1, data_tag="data_obs"):
    #samplelist = [x for x in hist.keys() if not data_tag in x and not 'BkgSum' in x and not x in sign and not x=="files"]
    samplelist = [x for x in hist.keys() if not x==data_tag in x and not 'BkgSum' in x and not x in sign and not x=="files"]
    print "Sample                  Events          Entries         %"
    print "-"*80
    for i, s in enumerate([data_tag]+samplelist+['BkgSum']):
        if i==1 or i==len(samplelist)+1: print "-"*80
        print "%-20s" % s, "\t%-10.2f" % hist[s].Integral(), "\t%-10.0f" % (hist[s].GetEntries()-2), "\t%-10.2f" % (100.*hist[s].Integral()/hist['BkgSum'].Integral()) if hist['BkgSum'].Integral() > 0 else 0, "%"
    print "-"*80
    #for i, s in enumerate(sign):
    for s in sorted(sign):
        if not samples[s]['plot']: continue
        print "%-20s" % s, "\t%-10.2f" % (hist[s].Integral()*SIGNAL), "\t%-10.0f" % (hist[s].GetEntries()-2), "\t%-10.2f" % (100.*hist[s].GetEntries()/float(hist[s].GetOption())) if float(hist[s].GetOption()) > 0 else 0, "%"    
    print "-"*80




##################
#     OTHERS     #
##################

def getPrimaryDataset(samples, cut, data_tag="data_obs"):
    pd = []
#    if 'HLT_PFMET' in cut: pd += [x for x in samples[data_tag]['files'] if "MET" in x]
#    if 'HLT_' in cut: pd += [x for x in samples[data_tag]['files'] if "MET" in x]
    pd += [x for x in samples[data_tag]['files'] if ("MET" in x or "DisplacedJet" in x or "SingleMuon" in x or "JetHT" in x or 'EGamma' in x or 'MuonEG' in x or 'SingleElectron' in x or 'SinglePhoton' in x)]
    return pd


def addOverflow(hist, addUnder=True):
    n = hist.GetNbinsX()
    hist.SetBinContent(n, hist.GetBinContent(n) + hist.GetBinContent(n+1))
    hist.SetBinError(n, math.sqrt( hist.GetBinError(n)**2 + hist.GetBinError(n+1)**2 ) )
    hist.SetBinContent(n+1, 0.)
    hist.SetBinError(n+1, 0.)
    if addUnder:
        hist.SetBinContent(1, hist.GetBinContent(0) + hist.GetBinContent(1))
        hist.SetBinError(1, math.sqrt( hist.GetBinError(0)**2 + hist.GetBinError(1)**2 ) )
        hist.SetBinContent(0, 0.)
        hist.SetBinError(0, 0.)

def setTopPad(TopPad, r=4):
    TopPad.SetPad("TopPad", "", 0., 1./r, 1.0, 1.0, 0, -1, 0)
    TopPad.SetTopMargin(0.24/r)
    TopPad.SetBottomMargin(0.04/r)
    TopPad.SetRightMargin(0.05)
    TopPad.SetTicks(1, 1)

def setBotPad(BotPad, r=4, forcetop=0):
    BotPad.SetPad("BotPad", "", 0., 0., 1.0, 1./r, 0, -1, 0)
    if forcetop==0:
        forcetop = r/100
    BotPad.SetTopMargin(forcetop)
    BotPad.SetBottomMargin(r/10.)
    BotPad.SetRightMargin(0.05)
    BotPad.SetTicks(1, 1)

def setHistStyle(hist, r=1.1):
    hist.GetXaxis().SetTitleSize(hist.GetXaxis().GetTitleSize()*r*r)
    hist.GetYaxis().SetTitleSize(hist.GetYaxis().GetTitleSize()*r*r)
    hist.GetXaxis().SetLabelSize(hist.GetXaxis().GetLabelSize()*r)
    hist.GetYaxis().SetLabelSize(hist.GetYaxis().GetLabelSize()*r)
    hist.GetXaxis().SetLabelOffset(hist.GetXaxis().GetLabelOffset()*r*r*r*r)
    hist.GetXaxis().SetTitleOffset(hist.GetXaxis().GetTitleOffset()*r)
    hist.GetYaxis().SetTitleOffset(hist.GetYaxis().GetTitleOffset())
    if hist.GetXaxis().GetTitle().find("GeV") != -1: # and not hist.GetXaxis().IsVariableBinSize()
        div = (hist.GetXaxis().GetXmax() - hist.GetXaxis().GetXmin()) / hist.GetXaxis().GetNbins()
        hist.GetYaxis().SetTitle("Events / %.1f GeV" % div)

def setBotStyle(h, r=4, fixRange=True, miny=0., maxy=2.):
    h.GetXaxis().SetLabelSize(h.GetXaxis().GetLabelSize()*(r-1));
    h.GetXaxis().SetLabelOffset(h.GetXaxis().GetLabelOffset()*(r-1));
    h.GetXaxis().SetTitleSize(h.GetXaxis().GetTitleSize()*(r-1));
    h.GetYaxis().SetLabelSize(h.GetYaxis().GetLabelSize()*(r-1));
    h.GetYaxis().SetNdivisions(505);
    h.GetYaxis().SetTitleSize(h.GetYaxis().GetTitleSize()*(r-1));
    h.GetYaxis().SetTitleOffset(h.GetYaxis().GetTitleOffset()/(r-1));
    if fixRange:
        h.GetYaxis().SetRangeUser(miny, maxy)
        for i in range(1, h.GetNbinsX()+1):
            if h.GetBinContent(i)<1.e-6:
                h.SetBinContent(i, -1.e-6)

##################
### DRAW UTILS ###
##################

def drawCMS(samples, LUMI, text, ERA="", onTop=False, left_marg_CMS=0.15,data_obs=[],text_size=0.045):
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(text_size)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(33)
    era_str = ""
    if ERA!="":
        era_str = ", "+ERA
    if (type(LUMI) is float or type(LUMI) is int) and float(LUMI) > 0:
        latex.DrawLatex(0.95, 0.985, ("%s fb^{-1}  (13 TeV%s)") % ( Decimal( str(LUMI/1000.) ).quantize(Decimal('1.'), rounding=ROUND_UP) ,era_str ) )
    elif type(LUMI) is str:
        latex.DrawLatex(0.95, 0.985, ("%s fb^{-1}  (13 TeV%s)" % (LUMI,era_str)) )
    if not onTop:
        latex.SetTextAlign(11)
    latex.SetTextFont(62)
    latex.SetTextSize(text_size if len(text)>0 else 0.06)
    if not onTop:
        #in a different line:
        #latex.DrawLatex(left_marg_CMS, 0.8 if len(text)>0 else 0.84, "CMS")
        #latex.DrawLatex(left_marg_CMS, 0.89 if len(text)>0 else 0.84, "CMS")
        latex.DrawLatex(left_marg_CMS, 0.84 if len(text)>0 else 0.84, "CMS")#gen level is ugly otherwise
    else:
        latex.DrawLatex(0.2, 0.9, "CMS")#DCMS
    latex.SetTextSize(text_size)
    latex.SetTextFont(52)
    if not onTop:
        latex.DrawLatex(left_marg_CMS, 0.89, text)#0.7
    else:
        #latex.DrawLatex(0.40, 0.98, text)
        latex.DrawLatex(0.33+text_size*2, 0.9, text)#DCMS
    dat = ""
    if len(data_obs)>0:
        print samples[data_obs[0]]['files'][0]
        if "SingleMuon" in (samples[data_obs[0]]['files'][0]):
            dat = "SingleMuon dataset"
        elif "SingleElectron" in (samples[data_obs[0]]['files'][0]):
            dat = "SingleElectron dataset"
        elif "EGamma" in (samples[data_obs[0]]['files'][0]):
            dat = "EGamma dataset"
        elif "DisplacedJet" in (samples[data_obs[0]]['files'][0]):
            dat = "DisplacedJet dataset"
        elif "MET" in (samples[data_obs[0]]['files'][0]):
            dat = "MET dataset"
        elif "HighMET" in (samples[data_obs[0]]['files'][0]):
            dat = "HighMET dataset"
        elif "JetHT" in (samples[data_obs[0]]['files'][0]):
            dat = "JetHT dataset"
        print "dat: ", dat
        latex2 = TLatex()
        latex2.SetNDC()
        latex2.SetTextFont(72) #52
        latex2.SetTextSize(text_size)
        latex2.SetTextAlign(10)
        #latex2.DrawLatex(0.45, 0.95, dat)
        latex2.DrawLatex(0.65, 0.9, dat)


def drawCMS_supplementary(LUMI, text, ERA="", onTop=False, top_margin=0.84,left_marg_CMS=0.15,data_obs=[],text_size=0.045):
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(text_size)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(33)
    era_str = ""
    if ERA!="":
        era_str = ", "+ERA
    if (type(LUMI) is float or type(LUMI) is int) and float(LUMI) > 0:
        latex.DrawLatex(0.95, 0.985, ("%s fb^{-1}  (13 TeV%s)") % ( Decimal( str(LUMI/1000.) ).quantize(Decimal('1.'), rounding=ROUND_UP) ,era_str ) )
    elif type(LUMI) is str:
        latex.DrawLatex(0.95, 0.985, ("%s fb^{-1}  (13 TeV%s)" % (LUMI,era_str)) )
    #latex.SetTextAlign(11)
    latex.SetTextFont(62)
    latex.SetTextSize(text_size if len(text)>0 else 0.06)
    latex.DrawLatex(left_marg_CMS, top_margin, "CMS")
    latex.SetTextSize(text_size)
    latex.SetTextFont(52)
    latex.DrawLatex(left_marg_CMS+0.25*text_size/0.045, top_margin, text)

def drawCMS_simple(LUMI, text, ERA="",onTop=False, left_marg_CMS=0.20,left_marg_LUMI=0.95,text_size=0.045,cms_text_size=0.06,lumi_text_size=0.04,custom_spacing=0):
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(lumi_text_size)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(33)
    era_str = ""
    if ERA!="":
        era_str = ", "+ERA
    if (type(LUMI) is float or type(LUMI) is int) and float(LUMI) > 0:
        latex.DrawLatex(left_marg_LUMI, 0.985, ("%s fb^{-1}  (13 TeV%s)") % ( Decimal( str(LUMI/1000.) ).quantize(Decimal('1.'), rounding=ROUND_UP) ,era_str ) )#( round(float(LUMI)/1000.,0),era_str) )
    elif type(LUMI) is str:
        latex.DrawLatex(left_marg_LUMI, 0.985, ("%s fb^{-1}  (13 TeV%s)" % (LUMI,era_str)) )
    if not onTop: latex.SetTextAlign(11)
    latex.SetTextFont(62)
    #latex.SetTextSize(0.05 if len(text)>0 else 0.06)
    latex.SetTextSize(cms_text_size)
    latex.DrawLatex(left_marg_CMS, 0.98, "CMS")
    latex.SetTextFont(52)#times 12.5
    spacing = left_marg_CMS+0.3*(cms_text_size/0.06)
    if len(text)>11:
        spacing+=(len(text)/100.)
    print text
    print len(text)
    if custom_spacing!=0:
        spacing = left_marg_CMS+custom_spacing
    latex.DrawLatex(spacing, 0.98, text)

def drawAnalysis(s, center=False):
    analyses = {
        "LL" : "VBF H #rightarrow #pi #pi #rightarrow b#bar{b} b#bar{b}",
        "LLZH" : "ZH #rightarrow #pi #pi #rightarrow b#bar{b} b#bar{b}",
        "LLVBF" : "VBF H #rightarrow #pi #pi #rightarrow b#bar{b} b#bar{b}",
        "LLVBFH" : "VBF H #rightarrow #pi #pi #rightarrow b#bar{b} b#bar{b}",
        "LLggH" : "ggH #rightarrow #pi #pi #rightarrow b#bar{b} b#bar{b}",
        "LLSUSY" : " #chi #chi #rightarrow  #tilde{G} h #tilde{G} h #rightarrow b#bar{b} b#bar{b}",
        "LLggHeavyHiggs" : "gg H2 #rightarrow S S #rightarrow b#bar{b} b#bar{b}",
        "LLsplitSUSY" : "#tilde{g} #tilde{g} #rightarrow tbs tbs",
        "LLgluinoGMSB" : "#tilde{g} #tilde{g} #rightarrow g #tilde{G} g #tilde{G}",
        "LLJetJet" : "Z^{*} #rightarrow XX #rightarrow q#bar{q} q#bar{q}",
        }
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    #latex.SetTextAlign(33)
    latex.DrawLatex(0.15 if not center else 0.3, 0.95, s if not s in analyses else analyses[s])


def drawBR(h, center=False, x_pos=0.15,y_pos=0.95):
    #br_string = "B(#chi #rightarrow h #tilde{G}) = "+str(h)+"%; B(#chi #rightarrow Z #tilde{G}) = "+str(100-h)+"%"
    if h==100:
        br_string = "B(#chi #rightarrow H #tilde{G}) = "+str(h)+"%"
    elif h==0:
        br_string = "B(#chi #rightarrow Z #tilde{G}) = "+str(100)+"%"
    else:
        br_string = "B(#chi #rightarrow H #tilde{G}) = "+str(h)+"%; B(#chi #rightarrow Z #tilde{G}) = "+str(100-h)+"%"
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    #latex.SetTextAlign(33)
    #latex.DrawLatex(0.15 if not center else 0.3, 0.95, s if not s in analyses else analyses[s])
    latex.DrawLatex(x_pos if not center else 0.3, y_pos, br_string)

def drawRegion(channel, left=False, left_marg_CMS=0.15, top=0.75, setNDC=True, color=1):
    region = {
        "SR_yes_BH" : "SR + Beam Halo",
        "VBFtrigger": "VBF triggers",
        "VBF": "VBF",
        "DisplacedJets" : "Displaced Jets phase-space",
        "MET" : "MET phase-space",
        "BTagCSV" : "BTagCSV phase-space",
        "VBFplusDisplacedTrigger" : "VBF + Displaced Jet trigger",
        "VBFplusDisplacedHadronicTrigger" : "VBF + Displaced hadronic Jet trigger",
        "VBFplusMETTrigger" : "VBF + MET trigger",
        "VBFplusVBFTrigger" : "VBF + VBF trigger",
        "ZtoMMCR": "Z #rightarrow #mu#mu CR",
        "ZtoEECR": "Z #rightarrow ee CR",
        "ZtoMMVBFCR": "VBF + Z #rightarrow #mu#mu CR",
        "ZtoEEVBFCR": "VBF + Z #rightarrow ee CR",
        "WtoMNCR": "W #rightarrow #mu#nu CR",
        "TopEMCR": "t #rightarrow #mue+X CR",

        "ZtoMM_CR": "Z #rightarrow #mu#mu CR",
        "ZtoEE_CR": "Z #rightarrow ee CR",
        "WtoEN_CR": "W #rightarrow e#nu CR",
        "WtoMN_CR": "W #rightarrow #mu#nu CR",
        "TtoEM_CR": "t #rightarrow #mue+X CR",

        "ZtoMM": "Z #rightarrow #mu#mu MR",
        "ZtoEE": "Z #rightarrow ee MR",
        "ZtoLL": "Z #rightarrow ll MR",
        "ZtoLLPho": "Z #rightarrow ll#gamma",
        "E": "1 e overlapping 1 jet + MET",
        "WtoEN": "W #rightarrow e#nu MR",
        "WtoMN": "W #rightarrow #mu#nu MR",
        "WtoLN": "W #rightarrow l#nu MR",
        "EN": "1 e + e trigger + MET>200 MR",
        "MN": "1 #mu + #mu trigger + MET>200 MR",
        "MR": "1 l + MET trigger + MET>200 MR",
        "TtoEM": "t #rightarrow #mue+X MR",
        "JetMET_Lep" : "MET>100, single jet triggers, 1 lepton MR",
        "JetMET_low_dPhi_500" : "MET>100, PFJet500 triggers, low #Delta#varphi(lead. jet,MET) MR",
        "JetMET_low_dPhi_MR" : "MET>100, single jet triggers, low #Delta#varphi(lead. jet,MET) MR",
        "JetMET_low_dPhi_low_MET_MR" : "100<MET<200, single jet triggers, low #Delta#varphi(lead. jet,MET) MR",
        "JetMET_low_dPhi_MET_200" : "MET>200, single jet triggers, low #Delta#varphi(lead. jet,MET) MR",
        "JetHT140" : "HLTJet140, MET<30 GeV",
        "DiJetMET140" : "HLTJet140, 2 jets, #Delta#varphi(jet[1],MET)<0.4, 60<MET<100 GeV",

        "DisplacedZtoMMCR": "VBF + Displaced jets + Z #rightarrow #mu#mu CR",
        "DisplacedWtoMNCR": "VBF + Displaced jets + W #rightarrow #mu#nu CR",
        "DisplacedTopEMCR": "VBF + Displaced jets + 1 #mu 1 e",
        "DisplacedCR0Tag" : "VBF + Displaced jets CR (0 calo tag)",
        "DisplacedHadronicCR0Tag" : "VBF + Displaced hadronic jets CR (0 calo tag)",
        "TSG" : "IsoMu24 + 1#mu",
        "L1seed" : "IsoMu24 + 1#mu + L1seed",
        "hltTripleJet50" : "IsoMu24 + 1#mu + L1seed + TripleJet50",
        "VBFplusPFMETNoMuTrigger" : "VBF + PFMETNoMu120 trigger",
        "VBFplusDisplacedHadronicTrigger" : "VBF + Displaced jet trigger",
        #"ZtoMM": "ZH #rightarrow #mu#mu H",
        #"ZtoEE": "ZH #rightarrow ee H",
        #"ZHMM": "ZH #rightarrow #mu#mu H",
        #"ZHEE": "ZH #rightarrow ee H",
        "ZH": "ZH #rightarrow ll H",
        "METMuCR": "MEt.pt>250 & HT>200 & 1 muon CR",
        "METCR": "MEt.pt>250 & HT>200 & veto leptons",
        "METHT": "E_{T}^{miss}>200 GeV & H_{T}>200 GeV",
        "METHTVeto": "E_{T}^{miss}>200 GeV & H_{T}>200 GeV & veto #l, #gamma",
        "METHTNoVeto": "MEt.pt>200 & HT>100, no veto",
        "METPreSel": "E_{T}^{miss}>200 GeV & H_{T}>100 GeV & veto #l, #gamma",
        "METPreSelSUSYAODAK4ECAL" : "LL decay in ECAL barr.",
        "METPreSelSUSYAODAK4HCAL" : "LL decay in HCAL barr.",
        "METPreSelSUSYAODAK8ECAL" : "LL decay in ECAL barr.",
        "METPreSelSUSYAODAK8HCAL" : "LL decay in HCAL barr.",
        "SR_veto_bin2_and_1_jet" : "Veto bin 2, at least 1 jet in acceptance",
        }
    
    text = ""
    if channel in region:
        text = region[channel]
    else:
        text = ""
    latex = TLatex()
    if setNDC:
        latex.SetNDC()
    latex.SetTextFont(72) #52
    latex.SetTextColor(color)
    latex.SetTextSize(0.035)
    if left: latex.DrawLatex(left_marg_CMS, top, text)
    else:
        latex.SetTextAlign(10)
        #latex.DrawLatex(0.12, 0.75, text)
        latex.DrawLatex(0.15, top, text)#DCMS

def drawTagVar(t, left=False, left_marg_CMS=0.15):

    tagvarlist = {
        "nCaloTagJets" : "",
        "nLooseCaloTagJets" : "",
        "nCaloTagJetsRebuilt" : "",
        "nHardCaloTagJets" : "",
        "nLeadingCaloTagJets" : "",
        "nGenMatchedJets" : "",
        "1Loose1Tight" : "",
        "cHadEFrac" : "cHadEFrac",
        "LEADER" : "DNN output",
        }

    text = t if not t in tagvarlist else tagvarlist[t]
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(62) #52
    latex.SetTextSize(0.035)
    if left: latex.DrawLatex(left_marg_CMS, 0.65, text)
    else:
        latex.SetTextAlign(10)
        latex.DrawLatex(0.12, 0.75, text)


def drawBox(x1, y1, x2, y2, t="", fillstyle=3005):
    box = TBox(x1, y1, x2, y2)
    box.SetFillColor(1)
    box.SetFillStyle(fillstyle)
    box.Draw()
    if not t=="":
        text = TLatex()
        text.SetTextColor(1)
        text.SetTextFont(42)
        text.SetTextAlign(23)
        text.SetTextSize(0.04)
        text.DrawLatex((x1+x2)/2., y2/1.15, t)
        text.Draw()
    return box

def drawLine(x1, y1, x2, y2,color=1):
    line = TLine(x1, y1, x2, y2)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.SetLineColor(color)
    line.Draw()
    return line

def drawText(x, y, t, col=1):
    text = TLatex()
    text.SetTextColor(col)
    text.SetTextFont(42)
    text.SetTextAlign(23)
    text.SetTextSize(0.04)
    text.DrawLatex(x, y, t)
    text.Draw()
    return text



