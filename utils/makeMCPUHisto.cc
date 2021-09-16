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


void makeMCPUHisto(
		   std::string inFilename=
		   "/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_2018_31December2020/ZZ_TuneCP5_13TeV-pythia8/crab_ZZ_TuneCP5_13TeV-pythia8-v2/210103_142219/0000/output_1.root",
		   std::string outFilename=
		   "PU.root",
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
  Int_t    MeanNumInteractions;

  // Input branches
  TBranch        *b_MeanNumInteractions;

  inTree->SetBranchAddress("MeanNumInteractions",              &MeanNumInteractions,              &b_MeanNumInteractions);
    
  // =================
  // Output
  // =================

  TFile *outFile = TFile::Open(outFilename.data(),"RECREATE", "", 207);
  TH1F* pileup =  new TH1F("pileup",";nPUMean;Number of Events", 200, 0, 200);

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
    //std::cout << "mean n.: " << MeanNumInteractions << std::endl;

    //std::cout << "======== " << std::endl;
    //std::cout << "Event " << entry << std::endl;
    //std::cout << "======== " << std::endl;
    pileup->Fill(MeanNumInteractions);
    if(entry % 1000 == 0) 
      {
	std::cout << "======== " << std::endl;
	std::cout << "Entry n. " << entry << " "  << std::endl;
	std::cout << "======== " << std::endl;
      }
        
  } // End of event loop

  pileup->SetLineColor(8);
  pileup->SetLineWidth(3);
  pileup->SetLineStyle(2);

  pileup->Write();
  outFile->Write();
  auto end = std::chrono::system_clock::now();//time!
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "**************************************************" << std::endl;
  std::cout << "finished  computations at " << std::ctime(&end_time)
	    << "elapsed time: " << elapsed_seconds.count() << "s\n";
  std::cout << "**************************************************" << std::endl;

}
