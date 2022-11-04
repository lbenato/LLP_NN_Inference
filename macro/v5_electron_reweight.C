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

//#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v6.h"
//#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/CaloObjects_v6.h"
#include <chrono>
#include <ctime>

using namespace std;

void v5_electron_reweight() {
  
  TFile *infile = new TFile("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_E_v5_ntuples/TTJets_DiLept_genMET-150_TuneCP5_13TeV-madgraphMLM-pythia8-v1.root", "READ");
  TTree* inputTree = (TTree*)infile->Get("tree");

  TFile *newfile = new TFile("small.root","recreate");
  TTree *newtree = inputTree->CloneTree(0);
  newtree->CopyEntries(inputTree);

  newtree->Print();
  
  float   HT;
  TBranch *b_HT;
  newtree->SetBranchAddress("HT",                &HT,                &b_HT);
  float   EventWeight;
  TBranch *b_EventWeight;
  newtree->SetBranchAddress("EventWeight",                &EventWeight,                &b_EventWeight);

  int nEvents(0);
  auto my_new_branch = newtree->Branch("new_v", &nEvents, "new_v/I");
  //loop
  for (Long64_t entry = 0 ; entry < newtree->GetEntriesFast() ; ++entry ) 
    {
      std::cout<<HT<<std::endl;
      std::cout<<EventWeight<<std::endl;
      EventWeight = -2.;

      newtree->GetEntry(entry);
      nEvents++;
      my_new_branch->Fill();
      newtree->Fill();
    }
  newfile->Write();

}
