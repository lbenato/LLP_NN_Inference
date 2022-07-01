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
  
  if(argc<3)
    {
      std::cout<<"Invalid arguments, exit!" << std::endl;
      return 0;
    }

  std::cout << "Input file: " << argv[1] << std::endl;
  std::cout << "Output file: " << argv[2] << std::endl;

  std::string inputPath = argv[1];
  std::string outputPath = argv[2];
  std::string ratioFilename = argv[3];
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
  
  TFile *ratioFile = TFile::Open(ratioFilename.data(),"READ"); if (!ratioFile) return 0;
  TH1D  *ratio = (TH1D*)ratioFile->Get("r");

  TFile* outputFile = new TFile(outputPath.c_str(), "RECREATE");
  outputFile->cd();
  TTree *outputTree = inputTree->CloneTree(0);
  //outputTree->CopyEntries(inputTree);

  float   HT;
  float   EventWeight;
  std::vector<JetType> *Jets=0;
  TBranch *b_HT;
  TBranch *b_EventWeight_old;
  TBranch *b_Jets = 0;
  inputTree->SetBranchAddress("HT", &HT, &b_HT);
  inputTree->SetBranchAddress("EventWeight", &EventWeight, &b_EventWeight_old);
  inputTree->SetBranchAddress("Jets",         &Jets,         &b_Jets);

  //Branch to be cleared
  float   NewEventWeight;
  TBranch *b_EventWeight;
  outputTree->SetBranchAddress("EventWeight", &NewEventWeight, &b_EventWeight);

  //Output tree
  //int nEvents(0);
  //auto my_new_branch = outputTree->Branch("NewEventWeight", &NewEventWeight, "NewEventWeight/F");
  for (Long64_t entry = 0 ; entry < inputTree->GetEntriesFast() ; ++entry )
  //for (Long64_t entry = 0 ; entry < 10 ; ++entry )
    {
      inputTree->GetEntry(entry);
      NewEventWeight = EventWeight;
      for(unsigned int j=0;j<Jets->size();j++)
	{
	  if(j==0) NewEventWeight*=ratio->GetBinContent(ratio->GetXaxis()->FindBin(Jets->at(j).photonEFrac));;
	}
      b_EventWeight->Clear();
      outputTree->Fill();
      //my_new_branch->Fill();
    }

  outputTree->SetWeight(tree_weight);
  counter->Write();
  n_pass->Write();
  n_odd->Write();
  n_even->Write();
  b_skipTrain->Write();
  outputFile->Write();

  ratioFile->Close();
  outputFile->Close();
  inputFile->Close();

  std::cout << "**************************************************" << std::endl;
  return 0;
}
