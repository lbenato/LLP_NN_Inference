#include "TFile.h"
#include "TTree.h"
#include "TMath.h"
#include "TH1F.h"

#include <vector>
#include <iostream>
#include <map>
#include <chrono>//compute time
#include <ctime>//compute time

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

void NormalizeHist(TH1F *hist)
{
  Double_t norm = 0;
  for (UInt_t b=0; int(b)<hist->GetXaxis()->GetNbins()+2; ++b) {
    norm += hist->GetBinContent(b);
  }
  for (UInt_t b=0; int(b)<hist->GetXaxis()->GetNbins()+2; ++b) {
    hist->SetBinContent(b,hist->GetBinContent(b) / norm);
    hist->SetBinError(b,hist->GetBinError(b) / norm);
  }
}

void DivideHist(TH1 *ratio, TH1 *num, TH1 *den) {  
  for (UInt_t b=0; int(b)<num->GetXaxis()->GetNbins()+2; ++b) {
    if ( den->GetBinContent(b) > 1.0e-4 ) {
      cout << "Bin: " << b << " " << ratio->GetXaxis()->GetBinCenter(b) << " : " << num->GetBinContent(b) << " / " << den->GetBinContent(b) << " = " << num->GetBinContent(b) / den->GetBinContent(b) << "\n";
      ratio->SetBinContent(b,num->GetBinContent(b) / den->GetBinContent(b));    
      ratio->SetBinError(b, (num->GetBinContent(b) / den->GetBinContent(b))*sqrt( pow(num->GetBinError(b)/num->GetBinContent(b),2) + pow(den->GetBinError(b)/den->GetBinContent(b),2)));
    } else {
      ratio->SetBinContent(b,0);
      ratio->SetBinError(b,0);
    }
  }
}

void MultiplyHist(TH1 *mult, TH1 *fact1, TH1 *fact2) {  
  for (UInt_t b=0; int(b)<fact1->GetXaxis()->GetNbins()+2; ++b) {
    if ( fact2->GetBinContent(b) > 1.0e-4 ) {
      mult->SetBinContent(b,fact1->GetBinContent(b) * fact2->GetBinContent(b));    
      mult->SetBinError(b, (fact1->GetBinContent(b) * fact2->GetBinContent(b))*sqrt( pow(fact1->GetBinError(b)*fact1->GetBinContent(b),2) + pow(fact2->GetBinError(b)*fact2->GetBinContent(b),2)));
    } else {
      mult->SetBinContent(b,0);
      mult->SetBinError(b,0);
    }
  }
}


void calculateMCPUWeight(
		   std::string inFilename=
		   "/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_2018_31December2020/ZZ_TuneCP5_13TeV-pythia8/crab_ZZ_TuneCP5_13TeV-pythia8-v2/210103_142219/0000/output_1.root",
		   std::string outFilename=
		   "out_test.root",
		   std::string mcFilename=
		   "PU.root",
		   std::string dataFilename = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/Analyzer/LLP2018/dataAOD/PU_69200_2018.root",
		   std::string dataFilenameUp = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/Analyzer/LLP2018/dataAOD/PU_72380_2018.root",
		   std::string dataFilenameDown = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/Analyzer/LLP2018/dataAOD/PU_66020_2018.root",
		   Long64_t first_event=0,
		   Long64_t last_event=-1
		   )

{

  auto start = std::chrono::system_clock::now();//time!
  // =================
  // Input
  // =================
  TFile *inFile = TFile::Open(inFilename.data(),"READ"); if (!inFile) return;
  TTree *inTree = (TTree*)inFile->Get("ntuple/tree");
  TH1F  *counter = (TH1F*)inFile->Get("counter/c_nEvents");
  Float_t tree_weight = inTree->GetWeight();
  std::cout << "Tree weight: " << tree_weight << std::endl;

  TFile *mcFile = TFile::Open(mcFilename.data(),"READ"); if (!mcFile) return;
  TFile *dataFile = TFile::Open(dataFilename.data(),"READ"); if (!dataFile) return;
  TFile *dataFileUp = TFile::Open(dataFilenameUp.data(),"READ"); if (!dataFileUp) return;
  TFile *dataFileDown = TFile::Open(dataFilenameDown.data(),"READ"); if (!dataFileDown) return;

  TH1F  *pileup = (TH1F*)mcFile->Get("pileup");
  TH1F  *pileup_mc = (TH1F*)pileup->Clone("pileup_mc");

  TH1F  *p_d = (TH1F*)dataFile->Get("pileup");
  TH1F  *pileup_data = (TH1F*)p_d->Clone("pileup_data");

  TH1F  *p_d_u = (TH1F*)dataFileUp->Get("pileup");
  TH1F  *pileup_data_up = (TH1F*)p_d_u->Clone("pileup_data_up");

  TH1F  *p_d_d = (TH1F*)dataFileDown->Get("pileup");
  TH1F  *pileup_data_down = (TH1F*)p_d_d->Clone("pileup_data_down");

  //Hist normalization
  NormalizeHist(pileup_mc);
  NormalizeHist(pileup_data);
  NormalizeHist(pileup_data_up);
  NormalizeHist(pileup_data_down);

  // Input variables
  Long64_t EventNumber;
  Long64_t RunNumber;
  Long64_t LumiNumber;
  Float_t  EventWeight;
  Bool_t   isMC;
  Int_t    MeanNumInteractions;

  // Input branches
  TBranch        *b_EventNumber;
  TBranch        *b_RunNumber;
  TBranch        *b_LumiNumber;
  TBranch        *b_EventWeight;
  TBranch        *b_isMC;
  TBranch        *b_MeanNumInteractions;

  inTree->SetBranchAddress("EventNumber",       &EventNumber,       &b_EventNumber);
  inTree->SetBranchAddress("RunNumber",         &RunNumber,         &b_RunNumber);
  inTree->SetBranchAddress("LumiNumber",        &LumiNumber,        &b_LumiNumber);
  inTree->SetBranchAddress("EventWeight",       &EventWeight,       &b_EventWeight);
  inTree->SetBranchAddress("isMC",              &isMC,              &b_isMC);
  inTree->SetBranchAddress("MeanNumInteractions",              &MeanNumInteractions,              &b_MeanNumInteractions);
    
  // =================
  // Output
  // =================

  TFile *outFile = TFile::Open(outFilename.data(),"RECREATE", "", 207);
  TTree *outTree = new TTree("tree", "tree");

  TH1F *PileupWeight = (TH1F*)pileup_mc->Clone("PUWeight");
  PileupWeight->SetLineColor(1);
  PileupWeight->SetLineWidth(3);
  DivideHist( PileupWeight , pileup_data, pileup_mc);

  TH1F *PileupWeightUp = (TH1F*)pileup_mc->Clone("PUWeightUp");
  PileupWeightUp->SetLineColor(2);
  PileupWeightUp->SetLineWidth(3);
  PileupWeightUp->SetLineStyle(2);
  DivideHist( PileupWeightUp , pileup_data_up, pileup_mc);

  TH1F *PileupWeightDown = (TH1F*)pileup_mc->Clone("PUWeightDown");
  PileupWeightDown->SetLineColor(4);
  PileupWeightDown->SetLineWidth(3);
  PileupWeightDown->SetLineStyle(2);
  DivideHist( PileupWeightDown , pileup_data_down, pileup_mc);

  TH1F *pileup_scaled = (TH1F*)pileup_mc->Clone("pileup_scaled");
  pileup_scaled->SetLineColor(8);
  pileup_scaled->SetLineWidth(3);
  pileup_scaled->SetLineStyle(1);
  MultiplyHist( pileup_scaled, PileupWeight, pileup_mc);

  pileup_mc->SetLineColor(8);
  pileup_mc->SetLineWidth(3);
  pileup_mc->SetLineStyle(2);
  pileup_data->SetLineColor(1);
  pileup_data->SetLineWidth(3);
  pileup_data_up->SetLineColor(2);
  pileup_data_up->SetLineWidth(3);
  pileup_data_up->SetLineStyle(2);
  pileup_data_down->SetLineColor(4);
  pileup_data_down->SetLineWidth(3);
  pileup_data_down->SetLineStyle(2);
    
  // Output branches    
  outTree->Branch("EventNumber",       &EventNumber,       "EventNumber/I");
  outTree->Branch("RunNumber",         &RunNumber,         "RunNumber/I");
  outTree->Branch("LumiNumber",        &LumiNumber,        "LumiNumber/I");
  outTree->Branch("EventWeight",       &EventWeight,       "EventWeight/F");
  outTree->Branch("isMC",              &isMC,              "isMC/O");
  outTree->Branch("MeanNumInteractions", &MeanNumInteractions, "MeanNumInteractions/I");

  // =================
  // Event loop
  // =================


  Long64_t start_loop;
  Long64_t stop_loop;
  start_loop = first_event>-1 ? first_event : 0;
  stop_loop = last_event>-1  ? last_event : inTree->GetEntriesFast();
  //for (Long64_t entry=0; entry<inTree->GetEntriesFast(); entry++) {
  std::cout << "Events in tree: " << inTree->GetEntriesFast() << std::endl;
  for (Long64_t entry=start_loop; entry<inTree->GetEntriesFast() && entry<stop_loop; entry++) {
    inTree->GetEntry(entry);
    std::cout << "mean n.: " << MeanNumInteractions << std::endl;

    //std::cout << "======== " << std::endl;
    //std::cout << "Event " << entry << std::endl;
    //std::cout << "======== " << std::endl;
    //pileup->Fill(MeanNumInteractions);
    if(entry % 1000 == 0) 
      {
	std::cout << "======== " << std::endl;
	std::cout << "Entry n. " << entry << " "  << std::endl;
	std::cout << "======== " << std::endl;
      }
    outTree->Fill();

        
  } // End of event loop




  outTree->SetWeight(tree_weight);
  counter->Write();
  pileup->Write();
  pileup_mc->Write();
  pileup_scaled->Write();
  pileup_data->Write();
  pileup_data_up->Write();
  pileup_data_down->Write();
  PileupWeight->Write();
  PileupWeightUp->Write();
  PileupWeightDown->Write();
  outFile->Write();
  auto end = std::chrono::system_clock::now();//time!
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "**************************************************" << std::endl;
  std::cout << "finished  computations at " << std::ctime(&end_time)
	    << "elapsed time: " << elapsed_seconds.count() << "s\n";
  std::cout << "**************************************************" << std::endl;

}


