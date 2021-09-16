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

#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects.h"
//#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/MyAnalysis.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <chrono>//compute time
#include <ctime>//compute time 

//bool pt_sorter(const PFCandidateType& x, const PFCandidateType& y) { return x.pt > y.pt; }
//bool energy_sorter(const ecalRecHitType& x, const ecalRecHitType& y) { return x.energy > y.energy; }
////bool h_energy_sorter(const hcalRecHitType& x, const hcalRecHitType& y) { return x.energy > y.energy; }

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

    if(argc<3)
    //if(argc<2)
      {
	std::cout<<"Invalid arguments, exit!" << std::endl;
	return 0;
      }

    bool skipTrain(false);
    if(strcmp(argv[3], "y")==1 || strcmp(argv[3], "yes")==1) skipTrain=true;

    std::cout << "Input file: " << argv[1] << std::endl;
    std::cout << "Output file: " << argv[2] << std::endl;
    std::cout << "Skip even EventNumber: " << skipTrain << std::endl;

    auto start = std::chrono::system_clock::now();//time!     

    std::string basePath = std::string(std::getenv("CMSSW_BASE")) + "/src/NNInferenceCMSSW/LLP_NN_Inference/nn_inference";
    // input and output file settings
    //unskimmed crab output
    //std::string inputPath = "/pnfs/desy.de/cms/tier2/store/user/lbenato/v4_calo_AOD_2018_18October2020/GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/201017_234633/0000/output_1.root";
    std::string inputPath = argv[1];

    std::string outputPath = argv[2];//!!!//"/test_on_real_ntuple.root";
    //std::string inputTreeName = "skim";
    std::string inputTreeName = "ntuple/tree";
    std::string outputTreeName = "tree";//inputTreeName;

    // ================= 
    // Input
    // ================= 

    // open input file, read the tree and register input branches
    TFile* inputFile = new TFile(inputPath.c_str(), "READ");
    TTree* inputTree = (TTree*)inputFile->Get(inputTreeName.c_str());
    TH1F   *counter = (TH1F*)inputFile->Get("counter/c_nEvents");
    float  tree_weight = inputTree->GetWeight();
    std::cout << "Tree weight: " << tree_weight << std::endl;


    // Input variables
    Long64_t EventNumber;
    Long64_t RunNumber;
    Long64_t LumiNumber;
    float    EventWeight;
    bool   HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v;
    bool   HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v;
    bool   HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v;
    bool   HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v;
    //float  HT;
    //float  MinJetMetDPhi;
    //Long64_t nCHSJets;
    //Long64_t nCHSFatJets;
    //int    nElectrons;
    //int    nMuons;
    //int    nPhotons;
    //int    nTaus;
    //int    nPFCandidates;
    //int    nPFCandidatesTrack;
    MEtType                      *MEt = 0;
    std::vector<GenPType>        *GenHiggs = 0;
    std::vector<GenPType>        *GenLLPs = 0;
    std::vector<GenPType>        *GenBquarks = 0;
    float dPhi_met_LLP0;
    float dPhi_met_LLP1;
    float RecHigMass;
    float RecLLP0Mass;
    float RecLLP1Mass;

    // Input branches
    TBranch        *b_GenHiggs = 0;
    TBranch        *b_GenLLPs = 0;
    TBranch        *b_GenBquarks = 0;
    TBranch        *b_MEt = 0;
    TBranch        *b_EventNumber;
    TBranch        *b_RunNumber;
    TBranch        *b_LumiNumber;
    TBranch        *b_EventWeight;
    TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v;
    TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v;
    TBranch        *b_HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v;
    TBranch        *b_HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v;

    inputTree->SetBranchAddress("GenHiggs",          &GenHiggs,          &b_GenHiggs); 
    inputTree->SetBranchAddress("GenLLPs",           &GenLLPs,           &b_GenLLPs); 
    inputTree->SetBranchAddress("GenBquarks",        &GenBquarks,        &b_GenBquarks); 
    inputTree->SetBranchAddress("MEt",               &MEt,               &b_MEt);
    inputTree->SetBranchAddress("EventNumber",       &EventNumber,       &b_EventNumber);
    inputTree->SetBranchAddress("RunNumber",         &RunNumber,         &b_RunNumber);
    inputTree->SetBranchAddress("LumiNumber",        &LumiNumber,        &b_LumiNumber);
    inputTree->SetBranchAddress("EventWeight",       &EventWeight,       &b_EventWeight);
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, &b_HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v, &b_HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v);



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
    outputTree->Branch("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v/O");
    outputTree->Branch("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v/O");
    outputTree->Branch("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v/O");
    outputTree->Branch("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v, "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v/O");
    outputTree->Branch("GenHiggs", &GenHiggs);
    outputTree->Branch("GenLLPs", &GenLLPs);
    outputTree->Branch("GenBquarks", &GenBquarks);
    outputTree->Branch("MEt", &MEt);
    outputTree->Branch("dPhi_met_LLP0", &dPhi_met_LLP0, "dPhi_met_LLP0/F");
    outputTree->Branch("dPhi_met_LLP1", &dPhi_met_LLP1, "dPhi_met_LLP1/F");
    outputTree->Branch("RecHigMass", &RecHigMass, "RecHigMass/F");
    outputTree->Branch("RecLLP0Mass", &RecLLP0Mass, "RecLLP0Mass/F");
    outputTree->Branch("RecLLP1Mass", &RecLLP1Mass, "RecLLP1Mass/F");

    // Event loop

    //for(int i = 0; i < 1; i++) {
    for(int i = 0; i < inputTree->GetEntriesFast(); i++) {

	//Initialize nTagJets at every event
        RecHigMass = -1;
        RecLLP0Mass = -1;
        RecLLP1Mass = -1;
	dPhi_met_LLP0 = -10.;
	dPhi_met_LLP1 = -10.;

	//Clear all the vectors

        if (i % 1000 == 0) {
            std::cout << "evaluating entry " << i << std::endl;
        }
        inputTree->GetEntry(i);

        //std::cout << "======== " << std::endl;
        //std::cout << "Event " << entry << std::endl;
	//std::cout << "======== " << std::endl;

        //if(!HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v) continue;
        ////if(HT<100) continue;
        //if(MEt->pt<200) continue;

	if(GenLLPs->size()==2)
	  {
	    dPhi_met_LLP0 =  fabs(reco::deltaPhi(GenLLPs->at(0).phi, MEt->phi));
	    dPhi_met_LLP1 =  fabs(reco::deltaPhi(GenLLPs->at(1).phi, MEt->phi));
	    TLorentzVector LLP0;
	    TLorentzVector LLP1;
	    TLorentzVector RecHig;
	    LLP0.SetPtEtaPhiE(GenLLPs->at(0).pt,GenLLPs->at(0).eta,GenLLPs->at(0).phi,GenLLPs->at(0).energy);
	    LLP1.SetPtEtaPhiE(GenLLPs->at(1).pt,GenLLPs->at(1).eta,GenLLPs->at(1).phi,GenLLPs->at(1).energy);
	    RecHig = LLP0 + LLP1;
	    RecHigMass = RecHig.M();
	  }

	if(GenBquarks->size()==4)
	  {
	    TLorentzVector b0;
	    TLorentzVector b1;
	    TLorentzVector b2;
	    TLorentzVector b3;
	    TLorentzVector LLP0;
	    TLorentzVector LLP1;
	    b0.SetPtEtaPhiE(GenBquarks->at(0).pt,GenBquarks->at(0).eta,GenBquarks->at(0).phi,GenBquarks->at(0).energy);
	    b1.SetPtEtaPhiE(GenBquarks->at(1).pt,GenBquarks->at(1).eta,GenBquarks->at(1).phi,GenBquarks->at(1).energy);
	    LLP0 = b0 + b1;
	    RecLLP0Mass = LLP0.M();

	    b2.SetPtEtaPhiE(GenBquarks->at(2).pt,GenBquarks->at(2).eta,GenBquarks->at(2).phi,GenBquarks->at(2).energy);
	    b3.SetPtEtaPhiE(GenBquarks->at(3).pt,GenBquarks->at(3).eta,GenBquarks->at(3).phi,GenBquarks->at(3).energy);
	    LLP1 = b2 + b3;
	    RecLLP1Mass = LLP1.M();
	  }

	outputTree->Fill();

    }

    // finalize files
    outputTree->SetWeight(tree_weight);
    counter->Write();
    //n_pass->Write();
    //n_odd->Write();
    //n_even->Write();
    //b_skipTrain->Write();
    outputFile->Write();
    outputFile->Close();
    inputFile->Close();

    auto end = std::chrono::system_clock::now();//time!
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "**************************************************" << std::endl;
    std::cout << "finished  computations at " << std::ctime(&end_time)
	      << "elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "**************************************************" << std::endl;

    std::cout << "Output written: " << outputPath << std::endl;
    std::cout << "\n" << std::endl;

    return 0;
}
