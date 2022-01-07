#! /usr/bin/env python
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
from NNInferenceCMSSW.LLP_NN_Inference.combine_settings import *

import optparse
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-b', '--batch', action='store_true', default=False, dest='batch')
parser.add_option('-B', '--blind', action='store_true', default=True, dest='blind')
parser.add_option('-c', '--channel', action='store', type='string', dest='channel', default='')
parser.add_option('-t', '--tagvar', action='store', type='string', dest='tagvar', default='')
parser.add_option('-F', '--fom', action='store', type='string', dest='fom', default='')
parser.add_option('-k', '--key', action='store', type='string', dest='key', default='AK4_0p994')
parser.add_option('-a', '--abcd', action='store_true', default=False, dest='abcd')
parser.add_option('-v', '--verbose', action='store_true', default=False, dest='verbose')
(options, args) = parser.parse_args()

if SIGNAL_SUPPRESSION_FACTOR == 1000:
    unit = "fb"
else:
    unit = "pb"

unit = "fb"
PRELIMINARY = True


# Combine output with data
# 0 - Observed Limit
# 1 - Expected  2.5%
# 2 - Expected 16.0%
# 3 - Expected 50.0%
# 4 - Expected 84.0%
# 5 - Expected 97.5%
# 6 - Significance
# 7 - p-value
# 8 - Best fit r
# 9 - Best fit r down
#10 - Best fit r up

# Combine output blind
# 0 - Expected  2.5%
# 1 - Expected 16.0%
# 2 - Expected 50.0%
# 3 - Expected 84.0%
# 4 - Expected 97.5%

if options.channel=="SUSY":
    mass_string = "mh"
    ctau_string = "pl"
    chan = "SUSY"
    isMM = isEE = isComb = False
    xs = 1#0.1#0.1#0.1#1
elif options.channel=="ggHeavyHiggs":
    mass_string = "MH"+str(MH)+"_MS"
    ctau_string = "ctau"
    chan = "ggH"
    isMM = isEE = isComb = False
    xs = 1#0.1#0.1#0.1#1
else:
    mass_string = "M"
    ctau_string = "ctau"



def combine_limits_vs_ctau(graph_list , channel, mH, mS,FOM,add_label=""):
    particle = "#chi" if "SUSY" in options.channel else "#pi"
    if "HeavyHiggs" in options.channel: particle = "S"
    colors = [881,801,856,920,825,2,602,880,798,5]#920 is grey
    colors = [881,801,856,825,2,602,880,798,5]#920 is grey
    colors = [801,856,825,881,2,602,880,798,5]#920 is grey
    gStyle.SetLegendFillColor(0)
    c2 = TCanvas("c2", "Significance", 1200, 600)#800,600
    c2.cd()
    c2.SetGrid()
    c2.GetPad(0).SetTopMargin(0.06)
    c2.GetPad(0).SetRightMargin(0.05)
    c2.GetPad(0).SetTicks(1, 1)
    c2.GetPad(0).SetGridx()
    c2.GetPad(0).SetGridy()
    c2.GetPad(0).SetLogy()
    c2.GetPad(0).SetLogx()
    top = 0.9
    nitems = len(graph_list)
    #top right
    #leg = TLegend(0.45+0.05, top-nitems*0.5/5., 0.75+0.05+0.1, top)
    #left
    leg = TLegend(0.05+0.05, top-nitems*0.5/5., 0.75+0.05+0.1-0.05, top)
    #smaller
    leg = TLegend(0.05+0.05, top-nitems*0.5/5., 0.75+0.05+0.1-0.15, top)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    #leg.SetHeader("95% CL expected limits")
    if FOM=="Limits":
        leg.SetHeader("#splitline{95% CL expected limits}{m_{H2} = "+str(mH)+" GeV; m_{S} = "+str(mS)+" GeV}")
    elif FOM=="Significance":
        leg.SetHeader("#splitline{Significance}{m_{H2} = "+str(mH)+" GeV; m_{S} = "+str(mS)+" GeV}")
    elif FOM=="yield":
        leg.SetHeader("#splitline{Event yield}{m_{H2} = "+str(mH)+" GeV; m_{S} = "+str(mS)+" GeV}")
    leg.SetTextSize(0.04)

    graph_exp = {}
    graph_1sigma = {}
    gS = {}
    gB = {}
    max_y = 0.1
    min_y = 10000
    for i,l in enumerate(graph_list):
        print l
        print i
        FILEDIR = RESULTDIR+l+"/"
        if FOM=="Limits":
            filename = FILEDIR+"Exclusion_mH"+str(mH)+"_mS"+str(mS)+".root"
        elif FOM=="Significance":
            filename = FILEDIR+"Significance_mH"+str(mH)+"_mS"+str(mS)+".root"
        elif FOM=="yield":
            filename = FILEDIR+"Yield_mH"+str(mH)+"_mS"+str(mS)+".root"
        print "opening this file: ", filename
        oldFile = TFile(filename, "READ")
        if FOM=="Limits":
            y_min = 0.003
            y_min = 0.1
            y_max = 0.03
            y_max = 100
            graph_1sigma[l] = TGraph()
            graph_1sigma[l] = oldFile.Get("mH"+str(mH)+"_mS"+str(mS)+"_1sigma")
            graph_exp[l] = TGraph()
            graph_exp[l] = oldFile.Get("mH"+str(mH)+"_mS"+str(mS)+"_exp")
            #graph_list.append(graph)
            graph_1sigma[l].SetLineColor(colors[i])
            graph_1sigma[l].SetLineStyle(2)
            graph_1sigma[l].SetLineWidth(3)
            graph_1sigma[l].SetFillStyle(3002)
            graph_1sigma[l].SetFillColorAlpha(colors[i],0.3)
            graph_1sigma[l].GetYaxis().SetTitle("#sigma(H) B("+particle+particle+" #rightarrow b #bar{b} b #bar{b})/#sigma_{SM}")
            graph_1sigma[l].GetYaxis().SetTitle("#sigma(H) B("+particle+particle+" #rightarrow b #bar{b} b #bar{b}) ("+unit+")")
            graph_1sigma[l].GetXaxis().SetNoExponent(True)
            graph_1sigma[l].GetXaxis().SetMoreLogLabels(True)
            graph_1sigma[l].GetXaxis().SetTitleSize(0.048)
            graph_1sigma[l].GetYaxis().SetTitleSize(0.048)
            graph_1sigma[l].GetYaxis().SetTitleOffset(0.8)
            graph_1sigma[l].GetXaxis().SetTitleOffset(0.9)
            graph_1sigma[l].GetXaxis().SetTitle("c#tau_{"+particle+"} (mm)")
            graph_1sigma[l].SetMinimum(y_min)
            graph_1sigma[l].SetMaximum(y_max)

            graph_exp[l].SetLineColor(colors[i])
            graph_exp[l].SetFillColorAlpha(colors[i],0.3)
            graph_exp[l].GetYaxis().SetTitle("#sigma(H) B("+particle+particle+" #rightarrow b #bar{b} b #bar{b})/#sigma_{SM}")
            if (options.channel=="ggHeavyHiggs" or options.channel=="SUSY"): graph_exp[l].GetYaxis().SetTitle("#sigma(H) B("+particle+particle+" #rightarrow b #bar{b} b #bar{b}) ("+unit+")")
            graph_exp[l].GetXaxis().SetTitle("c#tau_{"+particle+"} (mm)")
            graph_exp[l].GetXaxis().SetNoExponent(True)
            graph_exp[l].GetXaxis().SetMoreLogLabels(True)
            graph_exp[l].GetXaxis().SetTitleSize(0.048)
            graph_exp[l].GetYaxis().SetTitleSize(0.048)
            graph_exp[l].GetYaxis().SetTitleOffset(0.8)
            graph_exp[l].GetXaxis().SetTitleOffset(0.9)
            graph_exp[l].SetMinimum(y_min)
            graph_exp[l].SetMaximum(y_max)
            print graph_exp[l].Print()
            if i == 0:
                graph_1sigma[l].Draw("AL3")
                graph_exp[l].Draw("SAME,L3")
            else:
                graph_1sigma[l].Draw("SAME,L3")
                graph_exp[l].Draw("SAME,L3")
            print "Drawn: expected ", l
            leg.AddEntry(graph_exp[l],l,"L")
            
        elif FOM=="Significance":
            graph_exp[l] = TGraph()
            graph_exp[l] = oldFile.Get("mH"+str(mH)+"_mS"+str(mS)+"_sign")
            graph_exp[l].SetLineColor(colors[i])
            graph_exp[l].SetFillColorAlpha(colors[i],0.3)
            graph_exp[l].GetYaxis().SetTitle("Significance; H #rightarrow "+particle+particle+" #rightarrow b #bar{b} b #bar{b}")
            graph_exp[l].GetXaxis().SetTitle("c#tau_{"+particle+"} (mm)")
            graph_exp[l].GetXaxis().SetNoExponent(True)
            graph_exp[l].GetXaxis().SetMoreLogLabels(True)
            graph_exp[l].GetXaxis().SetTitleSize(0.048)
            graph_exp[l].GetYaxis().SetTitleSize(0.048)
            graph_exp[l].GetYaxis().SetTitleOffset(0.8)
            graph_exp[l].GetXaxis().SetTitleOffset(0.9)
            if mH>600:
                graph_exp[l].SetMinimum(0.0001)
                graph_exp[l].SetMaximum(0.01)
                if (mH==1000):
                    graph_exp[l].SetMinimum(0.00001)
                    graph_exp[l].SetMaximum(0.001)
            else:
                graph_exp[l].SetMinimum(0.0000001)
                graph_exp[l].SetMaximum(0.0001)
                if (mH==600 and mS==150):
                    graph_exp[l].SetMinimum(0.000001)
                    graph_exp[l].SetMaximum(0.001)
            if i == 0:
                graph_exp[l].Draw("AL3")
            else:
                graph_exp[l].Draw("SAME,L3")
            print "Drawn: expected ", l
            graph_exp[l].GetXaxis().SetRangeUser(0.0001,11000)
            leg.AddEntry(graph_exp[l],l,"L")           

        elif FOM=="yield":
            gS[l] = TGraph()
            gS[l] = oldFile.Get("mH"+str(mH)+"_mS"+str(mS)+"_S")
            gS[l].SetLineColor(colors[i])
            gS[l].SetFillColorAlpha(colors[i],0.3)
            gS[l].GetYaxis().SetTitle("Yield; H #rightarrow "+particle+particle+" #rightarrow b #bar{b} b #bar{b}")
            gS[l].GetXaxis().SetTitle("c#tau_{"+particle+"} (mm)")
            gS[l].GetXaxis().SetNoExponent(True)
            gS[l].GetXaxis().SetMoreLogLabels(True)
            gS[l].GetXaxis().SetTitleSize(0.048)
            gS[l].GetYaxis().SetTitleSize(0.048)
            gS[l].GetYaxis().SetTitleOffset(0.8)
            gS[l].GetXaxis().SetTitleOffset(0.9)
            max_y = max(max_y,gS[l].GetMaximum())
            min_y = min(min_y,gS[l].GetMinimum())
            #gS[l].SetMinimum(min_y)
            #gS[l].SetMaximum(max_y)
            gS[l].SetMinimum(0.01)
            gS[l].SetMaximum(100)

            gB[l] = TGraph()
            gB[l] = oldFile.Get("mH"+str(mH)+"_mS"+str(mS)+"_B")
            gB[l].SetLineColor(colors[i])
            gB[l].SetLineStyle(2)
            gB[l].SetFillColorAlpha(colors[i],0.3)
            gB[l].GetYaxis().SetTitle("Yield; H #rightarrow "+particle+particle+" #rightarrow b #bar{b} b #bar{b}")
            gB[l].GetXaxis().SetTitle("c#tau_{"+particle+"} (mm)")
            gB[l].GetXaxis().SetNoExponent(True)
            gB[l].GetXaxis().SetMoreLogLabels(True)
            gB[l].GetXaxis().SetTitleSize(0.048)
            gB[l].GetYaxis().SetTitleSize(0.048)
            gB[l].GetYaxis().SetTitleOffset(0.8)
            gB[l].GetXaxis().SetTitleOffset(0.9)
            max_y = max(max_y,gB[l].GetMaximum())
            min_y = min(min_y,gB[l].GetMinimum())
            #gB[l].SetMinimum(min_y)
            #gB[l].SetMaximum(max_y)
            gB[l].SetMinimum(0.01)
            gB[l].SetMaximum(100)
            if i == 0:
                gB[l].Draw("AL")
                gS[l].Draw("SAME,L")
            else:
                gB[l].Draw("SAME,L")
                gS[l].Draw("SAME,L")
            leg.AddEntry(gS[l],l+";S","L")           
            #leg.AddEntry(gB[l],l+";B","L")           
            
            
        oldFile.Close()
        print "\n"

    if FOM=="yield":
        dummy = TGraph()
        dummy.SetLineColor(1)
        dummy.SetLineWidth(2)
        dummy.SetLineStyle(2)
        leg.AddEntry(dummy,"Correspb. B","L")

    #lineY = TLine(massPoints[0],1.,massPoints[len(massPoints)-1],1.)
    lineY = TLine(ctauPoints_HH[0],1.,ctauPoints_HH[len(ctauPoints_HH)-1],1.)
    lineY.SetLineColor(1)
    lineY.SetLineWidth(2)
    lineY.SetLineStyle(7)
    if not FOM=="yield":
        lineY.Draw()

    leg.Draw()
    if PRELIMINARY:
        drawCMS(samples, LUMI, "Preliminary",left_marg_CMS=0.5)
        #drawCMS(samples, LUMI, "Work in Progress",left_marg_CMS=0.2)
    else:
        drawCMS(samples, LUMI, "",left_marg_CMS=0.32)
    drawAnalysis("LL"+chan)
    drawRegion(channel)

    if FOM=="Limits":
        OUTSTRING = RESULTDIR+"/Compare_Exclusion_vs_ctau_"
    elif FOM=="Significance":
        OUTSTRING = RESULTDIR+"/Compare_Significance_vs_ctau_"
    elif FOM=="yield":
        OUTSTRING = RESULTDIR+"/Compare_Yield_vs_ctau_"
    OUTSTRING+=chan+"_mH"+str(mH)+"_mS"+str(mS)
    if isMM:
        OUTSTRING +"_MM"
    elif isEE:
        OUTSTRING +"_EE"
    elif isComb:
        OUTSTRING +"_comb"
    if xs<1.:
        OUTSTRING+="_BR"+str(xs).replace('.','p')
    c2.Print(OUTSTRING + add_label+".png")
    c2.Print(OUTSTRING + add_label+".pdf")

    if not gROOT.IsBatch(): raw_input("Press Enter to continue...")


def combine_limits_vs_mass(graph_list , channel, MChi, FOM, add_label=""):
    particle = "#chi" if "SUSY" in options.channel else "#pi"
    if "HeavyHiggs" in options.channel: particle = "S"
    massPoints = MChi
    colors = [881,801,856,920,825,2,602,880,798,5]#920 is grey
    colors = [881,801,856,825,2,602,880,798,5]#920 is grey
    colors = [801,856,825,881,2,602,880,798,5]#920 is grey
    gStyle.SetLegendFillColor(0)
    c2 = TCanvas("c2", "Significance", 1200, 600)#800,600
    c2.cd()
    c2.SetGrid()
    c2.GetPad(0).SetTopMargin(0.06)
    c2.GetPad(0).SetRightMargin(0.05)
    c2.GetPad(0).SetTicks(1, 1)
    c2.GetPad(0).SetGridx()
    c2.GetPad(0).SetGridy()
    c2.GetPad(0).SetLogy()
    c2.GetPad(0).SetLogx()
    top = 0.9
    nitems = len(graph_list)
    #top right
    #leg = TLegend(0.45+0.05, top-nitems*0.5/5., 0.75+0.05+0.1, top)
    #left
    leg = TLegend(0.05+0.05, top-nitems*0.5/5., 0.75+0.05+0.1-0.05, top)
    #smaller
    leg = TLegend(0.05+0.05, top-nitems*0.5/5., 0.75+0.05+0.1-0.15, top)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    #leg.SetHeader("95% CL expected limits")
    if FOM=="Limits":
        leg.SetHeader("#splitline{95% CL expected limits}{c_{#tau} = "+str(ctauPoint[0])+"}")
    elif FOM=="Significance":
        leg.SetHeader("#splitline{Significance}{c_{#tau} = "+str(ctauPoint[0])+"}")
    elif FOM=="yield":
        leg.SetHeader("#splitline{Event yield}{c_{#tau} = "+str(ctauPoint[0])+"}")
    leg.SetTextSize(0.04)

    graph_exp = {}
    graph_1sigma = {}
    gS = {}
    gB = {}
    max_y = 0.1
    min_y = 10000
    for i,l in enumerate(graph_list):
        print l
        print i
        FILEDIR = RESULTDIR+l+"/"
        if FOM=="Limits":
            filename = FILEDIR+"Exclusion_ctau"+str(ctauPoint[0])+".root"
        elif FOM=="Significance":
            filename = FILEDIR+"Significance_ctau"+str(ctauPoint[0])+".root"
        #elif FOM=="yield":
        #    filename = FILEDIR+"Yield_ctau"+str(ctauPoint[0])+".root"
        print "opening this file: ", filename
        oldFile = TFile(filename, "READ")
        if FOM=="Limits":
            y_min = 0.003
            y_min = 0.01
            y_max = 0.1
            #y_max = 1
            graph_1sigma[l] = TGraph()
            graph_1sigma[l] = oldFile.Get("pl"+str(ctauPoint[0])+"_1sigma")
            graph_exp[l] = TGraph()
            graph_exp[l] = oldFile.Get("pl"+str(ctauPoint[0])+"_exp")
            #graph_list.append(graph)
            graph_1sigma[l].SetLineColor(colors[i])
            graph_1sigma[l].SetLineStyle(2)
            graph_1sigma[l].SetLineWidth(3)
            graph_1sigma[l].SetFillStyle(3002)
            graph_1sigma[l].SetFillColorAlpha(colors[i],0.3)
            graph_1sigma[l].GetYaxis().SetTitle("#sigma("+particle+particle+") B(h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b})/#sigma_{SUSY}")
            #graph_1sigma[l].GetYaxis().SetTitle("#sigma(H) B("+particle+particle+" #rightarrow b #bar{b} b #bar{b}) ("+unit+")")
            graph_1sigma[l].GetXaxis().SetNoExponent(True)
            graph_1sigma[l].GetXaxis().SetMoreLogLabels(True)
            graph_1sigma[l].GetXaxis().SetTitleSize(0.048)
            graph_1sigma[l].GetYaxis().SetTitleSize(0.048)
            graph_1sigma[l].GetYaxis().SetTitleOffset(0.8)
            graph_1sigma[l].GetXaxis().SetTitleOffset(0.9)
            graph_1sigma[l].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
            graph_1sigma[l].SetMinimum(y_min)
            graph_1sigma[l].SetMaximum(y_max)

            graph_exp[l].SetLineColor(colors[i])
            graph_exp[l].SetFillColorAlpha(colors[i],0.3)
            graph_exp[l].GetYaxis().SetTitle("#sigma(H) B("+particle+particle+" #rightarrow b #bar{b} b #bar{b})/#sigma_{SM}")
            if (options.channel=="ggHeavyHiggs"): graph_exp[l].GetYaxis().SetTitle("#sigma(H) B("+particle+particle+" #rightarrow b #bar{b} b #bar{b}) ("+unit+")")
            if options.channel=="SUSY": graph_exp[l].GetYaxis().SetTitle("#sigma("+particle+particle+") B(h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b})/#sigma_{SUSY}")
            graph_exp[l].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
            graph_exp[l].GetXaxis().SetNoExponent(True)
            graph_exp[l].GetXaxis().SetMoreLogLabels(True)
            graph_exp[l].GetXaxis().SetTitleSize(0.048)
            graph_exp[l].GetYaxis().SetTitleSize(0.048)
            graph_exp[l].GetYaxis().SetTitleOffset(0.8)
            graph_exp[l].GetXaxis().SetTitleOffset(0.9)
            graph_exp[l].SetMinimum(y_min)
            graph_exp[l].SetMaximum(y_max)
            print graph_exp[l].Print()
            if i == 0:
                graph_1sigma[l].Draw("AL3")
                graph_exp[l].Draw("SAME,L3")
            else:
                graph_1sigma[l].Draw("SAME,L3")
                graph_exp[l].Draw("SAME,L3")
            print "Drawn: expected ", l
            leg.AddEntry(graph_exp[l],l,"L")
            
        elif FOM=="Significance":
            graph_exp[l] = TGraph()
            graph_exp[l] = oldFile.Get("pl"+str(ctauPoint[0])+"_sign")
            graph_exp[l].SetLineColor(colors[i])
            graph_exp[l].SetFillColorAlpha(colors[i],0.3)
            graph_exp[l].GetYaxis().SetTitle("Significance; "+particle+particle+" #rightarrow h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b}")
            graph_exp[l].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
            graph_exp[l].GetXaxis().SetNoExponent(True)
            graph_exp[l].GetXaxis().SetMoreLogLabels(True)
            graph_exp[l].GetXaxis().SetTitleSize(0.048)
            graph_exp[l].GetYaxis().SetTitleSize(0.048)
            graph_exp[l].GetYaxis().SetTitleOffset(0.8)
            graph_exp[l].GetXaxis().SetTitleOffset(0.9)
            if mH>600:
                graph_exp[l].SetMinimum(0.0001)
                graph_exp[l].SetMaximum(0.01)
                if (mH==1000):
                    graph_exp[l].SetMinimum(0.00001)
                    graph_exp[l].SetMaximum(0.001)
            else:
                graph_exp[l].SetMinimum(0.0000001)
                graph_exp[l].SetMaximum(0.0001)
                if (mH==600 and mS==150):
                    graph_exp[l].SetMinimum(0.000001)
                    graph_exp[l].SetMaximum(0.001)
            if i == 0:
                graph_exp[l].Draw("AL3")
            else:
                graph_exp[l].Draw("SAME,L3")
            print "Drawn: expected ", l
            graph_exp[l].GetXaxis().SetRangeUser(0.0001,11000)
            leg.AddEntry(graph_exp[l],l,"L")           

        '''
        elif FOM=="yield":
            gS[l] = TGraph()
            gS[l] = oldFile.Get("ctau"+str(ctauPoint[0])+"_S")
            gS[l].SetLineColor(colors[i])
            gS[l].SetFillColorAlpha(colors[i],0.3)
            gS[l].GetYaxis().SetTitle("Yield; H #rightarrow "+particle+particle+" #rightarrow b #bar{b} b #bar{b}")
            gS[l].GetXaxis().SetTitle("c#tau_{"+particle+"} (mm)")
            gS[l].GetXaxis().SetNoExponent(True)
            gS[l].GetXaxis().SetMoreLogLabels(True)
            gS[l].GetXaxis().SetTitleSize(0.048)
            gS[l].GetYaxis().SetTitleSize(0.048)
            gS[l].GetYaxis().SetTitleOffset(0.8)
            gS[l].GetXaxis().SetTitleOffset(0.9)
            max_y = max(max_y,gS[l].GetMaximum())
            min_y = min(min_y,gS[l].GetMinimum())
            #gS[l].SetMinimum(min_y)
            #gS[l].SetMaximum(max_y)
            gS[l].SetMinimum(0.01)
            gS[l].SetMaximum(100)

            gB[l] = TGraph()
            gB[l] = oldFile.Get("ctau"+str(ctauPoint[0])+"_B")
            gB[l].SetLineColor(colors[i])
            gB[l].SetLineStyle(2)
            gB[l].SetFillColorAlpha(colors[i],0.3)
            gB[l].GetYaxis().SetTitle("Yield; H #rightarrow "+particle+particle+" #rightarrow b #bar{b} b #bar{b}")
            gB[l].GetXaxis().SetTitle("c#tau_{"+particle+"} (mm)")
            gB[l].GetXaxis().SetNoExponent(True)
            gB[l].GetXaxis().SetMoreLogLabels(True)
            gB[l].GetXaxis().SetTitleSize(0.048)
            gB[l].GetYaxis().SetTitleSize(0.048)
            gB[l].GetYaxis().SetTitleOffset(0.8)
            gB[l].GetXaxis().SetTitleOffset(0.9)
            max_y = max(max_y,gB[l].GetMaximum())
            min_y = min(min_y,gB[l].GetMinimum())
            #gB[l].SetMinimum(min_y)
            #gB[l].SetMaximum(max_y)
            gB[l].SetMinimum(0.01)
            gB[l].SetMaximum(100)
            if i == 0:
                gB[l].Draw("AL")
                gS[l].Draw("SAME,L")
            else:
                gB[l].Draw("SAME,L")
                gS[l].Draw("SAME,L")
            leg.AddEntry(gS[l],l+";S","L")           
            #leg.AddEntry(gB[l],l+";B","L")           
        '''    
            
        oldFile.Close()
        print "\n"

    '''
    if FOM=="yield":
        dummy = TGraph()
        dummy.SetLineColor(1)
        dummy.SetLineWidth(2)
        dummy.SetLineStyle(2)
        leg.AddEntry(dummy,"Correspb. B","L")
    '''

    lineY = TLine(massPoints[0],1.,massPoints[len(massPoints)-1],1.)
    #lineY = TLine(ctauPoints_HH[0],1.,ctauPoints_HH[len(ctauPoints_HH)-1],1.)
    lineY.SetLineColor(1)
    lineY.SetLineWidth(2)
    lineY.SetLineStyle(7)
    if not FOM=="yield":
        lineY.Draw()

    leg.Draw()
    if PRELIMINARY:
        drawCMS(samples, LUMI, "Preliminary",left_marg_CMS=0.5)
        #drawCMS(samples, LUMI, "Work in Progress",left_marg_CMS=0.2)
    else:
        drawCMS(samples, LUMI, "",left_marg_CMS=0.32)
    drawAnalysis("LL"+chan)
    drawRegion(channel)

    if FOM=="Limits":
        OUTSTRING = RESULTDIR+"/Compare_Exclusion_vs_mass_"
    elif FOM=="Significance":
        OUTSTRING = RESULTDIR+"/Compare_Significance_vs_mass_"
    #elif FOM=="yield":
    #    OUTSTRING = RESULTDIR+"/Compare_Yield_vs_ctau_"
    OUTSTRING+=chan+"_ctau"+str(ctauPoint[0])
    if isMM:
        OUTSTRING +"_MM"
    elif isEE:
        OUTSTRING +"_EE"
    elif isComb:
        OUTSTRING +"_comb"
    if xs<1.:
        OUTSTRING+="_BR"+str(xs).replace('.','p')
    c2.Print(OUTSTRING + add_label+".png")
    c2.Print(OUTSTRING + add_label+".pdf")

    if not gROOT.IsBatch(): raw_input("Press Enter to continue...")
    

'''
        
for h in MH.keys():
    for s in MH[h]:
'''


if options.channel=="SUSY":
    for c in ctauPoint:
        probe = ["AK4_0p996_JJ","AK4_0p997_miniAOD"]
        #for p in probe:
        #    print "\n"
        #    os.system("python macro/plot_limits_SUSY.py -b -c "+options.channel+" -F "+options.fom+" -k " + p)
        combine_limits_vs_mass(probe,options.channel,MChi,options.fom,"_AK4_compare_miniAOD")

elif options.channel=="ggHeavyHiggs":
    for h in MH.keys():
        for s in MH[h]:

            probe = ["AK4_0p996_JJ","AK4_0p997_miniAOD"]
            #probe = ["AK4_0p997_miniAOD"]
            combine_limits_vs_ctau(probe,options.channel,h,s,options.fom,"_AK4_compare_miniAOD")


