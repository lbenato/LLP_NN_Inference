#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include "TFile.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TMath.h"
#include "TGraph2D.h"
#include "TRandom2.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TF2.h"
#include "TF3.h"
#include "TF1.h"
#include "Math/Functor.h"
#include "TPolyLine3D.h"
#include "Math/Vector3D.h"
#include "Fit/Fitter.h"

#include <cassert>
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v6.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/CaloObjects_v6.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/dbscan.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <chrono>
#include <ctime>

using namespace std;

int main(int argc, char **argv) {
  
  if(argc<4)
    {
      std::cout<<"Invalid arguments, exit!" << std::endl;
      return 0;
    }

  std::cout << "Input file: " << argv[1] << std::endl;
  std::cout << "Output file: " << argv[2] << std::endl;

  std::string inputPath = argv[1];
  std::string outputPath = argv[2];
  std::string ratioHTFilename = argv[3];
  std::string ratioEleFilename = argv[4];
  std::string inputTreeName = "tree";
  std::string outputTreeName = "tree";
  TFile* inputFile = new TFile(inputPath.c_str(), "READ");
  TTree* inputTree = (TTree*)inputFile->Get(inputTreeName.c_str());
  TH1F   *counter = (TH1F*)inputFile->Get("c_nEvents");
  TH1F   *n_pass = (TH1F*)inputFile->Get("n_pass");
  TH1F   *n_odd = (TH1F*)inputFile->Get("n_odd");
  TH1F   *n_even = (TH1F*)inputFile->Get("n_even");
  TH1F   *b_skipTrain = (TH1F*)inputFile->Get("b_skipTrain");
  float  tree_weight = inputTree->GetWeight();
  
  TFile *ratioHTFile = TFile::Open(ratioHTFilename.data(),"READ"); if (!ratioHTFile) return 0;
  TH1D  *ratioHT = (TH1D*)ratioHTFile->Get("ratio");

  TFile *ratioEleFile = TFile::Open(ratioEleFilename.data(),"READ"); if (!ratioEleFile) return 0;
  if(ratioEleFile) TH1D  *ratioEle = (TH1D*)ratioEleFile->Get("ratio");

  TFile* outputFile = new TFile(outputPath.c_str(), "RECREATE");
  outputFile->cd();
  TTree *outputTree = inputTree->CloneTree(0);
  //outputTree->CopyEntries(inputTree);

  float   HT;
  float   oldEventWeight;
  std::vector<LeptonType> *Electrons=0;
  TBranch *b_HT;
  TBranch *b_EventWeight_old;
  TBranch        *b_Electrons = 0;
  inputTree->SetBranchAddress("HT", &HT, &b_HT);
  inputTree->SetBranchAddress("EventWeight", &oldEventWeight, &b_EventWeight_old);
  inputTree->SetBranchAddress("Electrons",         &Electrons,         &b_Electrons);

  //Branch to be cleared
  float   EventWeight;
  TBranch *b_EventWeight;
  outputTree->SetBranchAddress("EventWeight", &EventWeight, &b_EventWeight);

  //Output tree
  //int nEvents(0);
  //auto my_new_branch = outputTree->Branch("NewEventWeight", &EventWeight, "EventWeight/F");
  for (Long64_t entry = 0 ; entry < inputTree->GetEntriesFast() ; ++entry )
  //for (Long64_t entry = 0 ; entry < 10 ; ++entry )
    {
      inputTree->GetEntry(entry);
      EventWeight = oldEventWeight*ratioHT->GetBinContent(ratioHT->GetXaxis()->FindBin(HT));;
      //EventWeight *= ratioEle->GetBinContent(ratioEle->GetXaxis()->FindBin(Electrons->at(0).pt));;
      b_EventWeight->Clear();
      outputTree->Fill();
    }

  outputTree->SetWeight(tree_weight);
  counter->Write();
  n_pass->Write();
  n_odd->Write();
  n_even->Write();
  b_skipTrain->Write();
  outputFile->Write();

  ratioHTFile->Close();
  if(ratioEleFile) ratioEleFile->Close();
  outputFile->Close();
  inputFile->Close();

  std::cout << "**************************************************" << std::endl;
  return 0;
}
