/*
 * Minimal example showing how to evaluate data contained in a flat tree structure using TensorFlow.
 * By default, the inference code uses single threading and no batching. The thread model is
 * configurable, however, note that this is done differently depending on the version of TensorFlow,
 * which changed significantly as of version 2.
 *
 * Author: Marcel Rieger
 */

// Adapted to LLP displaced jets in calorimeter by Lisa Benato


#include <iostream>
#include <fstream>
#include <vector>
#include "TFile.h"
#include "TLorentzVector.h"
#include "TTree.h"
#include "TH1.h"
#include "TMath.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v5.h"
//#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/MyAnalysis.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <chrono>//compute time
#include <ctime>//compute time 

//JJ:
struct Particle {
  TLorentzVector vec;
  int charge;
  int pdgId;
};

double avg ( std::vector<double> & v )
{
  double return_value = 0.0;
  int n = v.size();

  for ( int i=0; i < n; i++)
    {
      return_value += v.at(i);
    }

  return ( return_value / n);
}


int main(int argc, char **argv) {

    std::string inputPath = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2017_WtoEN/SingleElectronRun2017F-17Nov2017-v1.root";
    std::string outputPath = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2017_WtoEN/SingleElectronRun2017F-17Nov2017-v1_skim.root";

    std::string inputTreeName = "tree";
    std::string outputTreeName = "tree";

    std::cout << "Input file: " << inputPath << std::endl;

    // ================= 
    // Input
    // ================= 

    // open input file, read the tree and register input branches
    TFile* inputFile = new TFile(inputPath.c_str(), "READ");
    TTree* inputTree = (TTree*)inputFile->Get(inputTreeName.c_str());
    float  tree_weight = inputTree->GetWeight();

    // Input variables & branches
    Long64_t EventNumber;
    TBranch *b_EventNumber;
    inputTree->SetBranchAddress("EventNumber",&EventNumber,&b_EventNumber);

    Long64_t RunNumber;
    TBranch *b_RunNumber;
    inputTree->SetBranchAddress("RunNumber",&RunNumber,&b_RunNumber);

    Long64_t LumiNumber;
    TBranch *b_LumiNumber;
    inputTree->SetBranchAddress("LumiNumber",&LumiNumber,&b_LumiNumber);

    float EventWeight;
    TBranch *b_EventWeight;
    inputTree->SetBranchAddress("EventWeight",&EventWeight,&b_EventWeight);

    float PUWeight;
    TBranch *b_PUWeight;
    inputTree->SetBranchAddress("PUWeight",&PUWeight,&b_PUWeight);

    float PUReWeight;
    TBranch *b_PUReWeight;
    inputTree->SetBranchAddress("PUReWeight",&PUReWeight,&b_PUReWeight);

    bool isMC;
    TBranch *b_isMC;
    inputTree->SetBranchAddress("isMC",&isMC,&b_isMC);
    bool isSR;
    TBranch *b_isSR;
    inputTree->SetBranchAddress("isSR",&isSR,&b_isSR);
    //bool isMR;
    //TBranch *b_isMR;
    //inputTree->SetBranchAddress("isMR",&isMR,&b_isMR);
    bool isZtoMM;
    TBranch *b_isZtoMM;
    inputTree->SetBranchAddress("isZtoMM",&isZtoMM,&b_isZtoMM);
    bool isZtoEE;
    TBranch *b_isZtoEE;
    inputTree->SetBranchAddress("isZtoEE",&isZtoEE,&b_isZtoEE);
    bool isWtoMN;
    TBranch *b_isWtoMN;
    inputTree->SetBranchAddress("isWtoMN",&isWtoMN,&b_isWtoMN);
    bool isWtoEN;
    TBranch *b_isWtoEN;
    inputTree->SetBranchAddress("isWtoEN",&isWtoEN,&b_isWtoEN);

    float HT;
    TBranch  *b_HT;
    inputTree->SetBranchAddress("HT",&HT,&b_HT);
    float MT;
    TBranch  *b_MT;
    inputTree->SetBranchAddress("MT",&MT,&b_MT);

    float MinJetMetDPhi;
    TBranch *b_MinJetMetDPhi;
    inputTree->SetBranchAddress("MinJetMetDPhi",&MinJetMetDPhi,&b_MinJetMetDPhi);
    float MinJetMetDPhiBarrel;
    TBranch *b_MinJetMetDPhiBarrel;
    inputTree->SetBranchAddress("MinJetMetDPhiBarrel",&MinJetMetDPhiBarrel,&b_MinJetMetDPhiBarrel);
    /*
    float MinLeadingJetMetDPhi;
    TBranch *b_MinLeadingJetMetDPhi;
    inputTree->SetBranchAddress("MinLeadingJetMetDPhi",&MinLeadingJetMetDPhi,&b_MinLeadingJetMetDPhi);
    float MinSubLeadingJetMetDPhi;
    TBranch *b_MinSubLeadingJetMetDPhi;
    inputTree->SetBranchAddress("MinSubLeadingJetMetDPhi",&MinSubLeadingJetMetDPhi,&b_MinSubLeadingJetMetDPhi);
    float MinSubSubLeadingJetMetDPhi;
    TBranch *b_MinSubSubLeadingJetMetDPhi;
    inputTree->SetBranchAddress("MinSubSubLeadingJetMetDPhi",&MinSubSubLeadingJetMetDPhi,&b_MinSubSubLeadingJetMetDPhi);
    */

    //int nPV;
    //TBranch *b_nPV;    
    //inputTree->SetBranchAddress("nPV",&nPV,&b_nPV);

    int nCHSJets;
    TBranch *b_nCHSJets;
    inputTree->SetBranchAddress("nCHSJets",&nCHSJets,&b_nCHSJets);
    int nCHSJets_in_HEM;
    TBranch *b_nCHSJets_in_HEM;
    inputTree->SetBranchAddress("nCHSJets_in_HEM",&nCHSJets_in_HEM,&b_nCHSJets_in_HEM);
    int nCHSJetsAcceptanceCalo;
    TBranch *b_nCHSJetsAcceptanceCalo;
    inputTree->SetBranchAddress("nCHSJetsAcceptanceCalo",&nCHSJetsAcceptanceCalo,&b_nCHSJetsAcceptanceCalo);

    std::vector<JetType> *Jets = 0;
    TBranch *b_Jets = 0;
    inputTree->SetBranchAddress("Jets",&Jets,&b_Jets);

    MEtType *MEt = 0;
    TBranch *b_MEt = 0;
    inputTree->SetBranchAddress("MEt",&MEt,&b_MEt);
    
    int nTagJets_cutbased;
    TBranch *b_nTagJets_cutbased;
    inputTree->SetBranchAddress("nTagJets_cutbased",&nTagJets_cutbased,&b_nTagJets_cutbased);

    int nTagJets_cutbased_JJ;
    TBranch *b_nTagJets_cutbased_JJ;
    inputTree->SetBranchAddress("nTagJets_cutbased_JJ",&nTagJets_cutbased_JJ,&b_nTagJets_cutbased_JJ);

    int nTagJets_0p996_JJ;
    TBranch *b_nTagJets_0p996_JJ;
    inputTree->SetBranchAddress("nTagJets_0p996_JJ",&nTagJets_0p996_JJ,&b_nTagJets_0p996_JJ);
    
    // ================= 
    // Output
    // ================= 

    TFile* outputFile = new TFile(outputPath.c_str(), "RECREATE");
    outputFile->cd();
    TTree *outputTree = new TTree(outputTreeName.c_str(), "");

    // Output branches 
    outputTree->Branch("EventNumber",       &EventNumber,       "EventNumber/L");
    outputTree->Branch("RunNumber",         &RunNumber,         "RunNumber/L");
    outputTree->Branch("LumiNumber",        &LumiNumber,        "LumiNumber/L");
    outputTree->Branch("EventWeight",       &EventWeight,       "EventWeight/F");
    outputTree->Branch("PUWeight",          &PUWeight,          "PUWeight/F");
    outputTree->Branch("PUReWeight",        &PUReWeight,        "PUReWeight/F");
    outputTree->Branch("isMC",              &isMC,              "isMC/O");
    outputTree->Branch("isSR",              &isSR,              "isSR/O");
    //outputTree->Branch("isMR",              &isMR,              "isMR/O");
    outputTree->Branch("isZtoMM",           &isZtoMM,           "isZtoMM/O");
    outputTree->Branch("isZtoEE",           &isZtoEE,           "isZtoEE/O");
    outputTree->Branch("isWtoMN",           &isWtoMN,           "isWtoMN/O");
    outputTree->Branch("isWtoEN",           &isWtoEN,           "isWtoEN/O");
    outputTree->Branch("HT",                &HT,                "HT/F");
    outputTree->Branch("MT",                &MT,                "MT/F");
    outputTree->Branch("MinJetMetDPhi",     &MinJetMetDPhi,     "MinJetMetDPhi/F");
    outputTree->Branch("MinJetMetDPhiBarrel",  &MinJetMetDPhiBarrel,  "MinJetMetDPhiBarrel/F");
    //outputTree->Branch("MinLeadingJetMetDPhi", &MinLeadingJetMetDPhi, "MinLeadingJetMetDPhi/F");
    //outputTree->Branch("MinSubLeadingJetMetDPhi", &MinSubLeadingJetMetDPhi, "MinSubLeadingJetMetDPhi/F");
    //outputTree->Branch("MinSubSubLeadingJetMetDPhi", &MinSubSubLeadingJetMetDPhi, "MinSubSubLeadingJetMetDPhi/F");
    //outputTree->Branch("nPV",          &nPV,          "nPV/I");
    outputTree->Branch("nCHSJets",          &nCHSJets,          "nCHSJets/I");
    outputTree->Branch("nCHSJetsAcceptanceCalo",          &nCHSJetsAcceptanceCalo,          "nCHSJetsAcceptanceCalo/I");
    outputTree->Branch("nCHSJets_in_HEM" , &nCHSJets_in_HEM, "nCHSJets_in_HEM/I");
    outputTree->Branch("Jets", &Jets);
    outputTree->Branch("MEt", &MEt);
    outputTree->Branch("nTagJets_cutbased", &nTagJets_cutbased,  "nTagJets_cutbased/I");
    outputTree->Branch("nTagJets_cutbased_JJ", &nTagJets_cutbased_JJ,  "nTagJets_cutbased_JJ/I");
    outputTree->Branch("nTagJets_0p996_JJ",     &nTagJets_0p996_JJ,      "nTagJets_0p996_JJ/I");

    // Event loop

    //for(int i = 0; i < 10; i++) {
    for(int i = 0; i < inputTree->GetEntriesFast(); i++) {

        inputTree->GetEntry(i);
	if(i%1000000 == 0) std::cout << "Entry n. " << i << std::endl;


	if(nTagJets_0p996_JJ<2) continue;
	//if(RunNumber>=319077)
	//  {
	//    RunNumber_in_HEM = true;
	//  }
	outputTree->Fill();

	//if(i>=2000) break;

    }


    // finalize files
    outputTree->SetWeight(tree_weight);

    outputFile->Write();
    outputFile->Close();
    inputFile->Close();


    std::cout << "**************************************************" << std::endl;
    std::cout << "Output written: " << outputPath << std::endl;
    std::cout << "\n" << std::endl;

    return 0;
}
