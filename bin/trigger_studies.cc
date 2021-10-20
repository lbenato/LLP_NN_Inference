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
    bool   HLT_AK8PFHT800_TrimMass50_v;
    bool   HLT_AK8PFHT850_TrimMass50_v;
    bool   HLT_AK8PFHT900_TrimMass50_v;
    bool   HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17_v;
    bool   HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p1_v;
    bool   HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2_v;
    bool   HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4_v;
    bool   HLT_AK8PFJet360_TrimMass30_v;
    bool   HLT_AK8PFJet380_TrimMass30_v;
    bool   HLT_AK8PFJet400_TrimMass30_v;
    bool   HLT_AK8PFJet420_TrimMass30_v;
    bool   HLT_AK8PFJet500_v;  
    bool   HLT_HT430_DisplacedDijet40_DisplacedTrack_v;
    bool   HLT_HT430_DisplacedDijet60_DisplacedTrack_v;
    bool   HLT_HT500_DisplacedDijet40_DisplacedTrack_v;
    bool   HLT_HT650_DisplacedDijet60_Inclusive_v;
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
    std::vector<GenPType>        *GenHiggs = 0;
    std::vector<GenPType>        *GenLLPs = 0;
    std::vector<GenPType>        *GenBquarks = 0;

    // Input branches
    TBranch        *b_GenHiggs = 0;
    TBranch        *b_GenLLPs = 0;
    TBranch        *b_GenBquarks = 0;
    TBranch        *b_EventNumber;
    TBranch        *b_RunNumber;
    TBranch        *b_LumiNumber;
    TBranch        *b_EventWeight;
    TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v;
    TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v;
    TBranch        *b_HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v;
    TBranch        *b_HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v;
    TBranch        *b_HLT_AK8PFHT800_TrimMass50_v;
    TBranch        *b_HLT_AK8PFHT850_TrimMass50_v;
    TBranch        *b_HLT_AK8PFHT900_TrimMass50_v;
    TBranch        *b_HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17_v;
    TBranch        *b_HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p1_v;
    TBranch        *b_HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2_v;
    TBranch        *b_HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4_v;
    TBranch        *b_HLT_AK8PFJet360_TrimMass30_v;
    TBranch        *b_HLT_AK8PFJet380_TrimMass30_v;
    TBranch        *b_HLT_AK8PFJet400_TrimMass30_v;
    TBranch        *b_HLT_AK8PFJet420_TrimMass30_v;
    TBranch        *b_HLT_AK8PFJet500_v;  
    TBranch        *b_HLT_HT430_DisplacedDijet40_DisplacedTrack_v;
    TBranch        *b_HLT_HT430_DisplacedDijet60_DisplacedTrack_v;
    TBranch        *b_HLT_HT500_DisplacedDijet40_DisplacedTrack_v;
    TBranch        *b_HLT_HT650_DisplacedDijet60_Inclusive_v;

    inputTree->SetBranchAddress("GenHiggs",          &GenHiggs,          &b_GenHiggs); 
    inputTree->SetBranchAddress("GenLLPs",           &GenLLPs,           &b_GenLLPs); 
    inputTree->SetBranchAddress("GenBquarks",        &GenBquarks,        &b_GenBquarks); 
    inputTree->SetBranchAddress("EventNumber",       &EventNumber,       &b_EventNumber);
    inputTree->SetBranchAddress("RunNumber",         &RunNumber,         &b_RunNumber);
    inputTree->SetBranchAddress("LumiNumber",        &LumiNumber,        &b_LumiNumber);
    inputTree->SetBranchAddress("EventWeight",       &EventWeight,       &b_EventWeight);
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, &b_HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v, &b_HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v);
    inputTree->SetBranchAddress("HLT_AK8PFHT800_TrimMass50_v", &HLT_AK8PFHT800_TrimMass50_v , &b_HLT_AK8PFHT800_TrimMass50_v);
    inputTree->SetBranchAddress("HLT_AK8PFHT850_TrimMass50_v", &HLT_AK8PFHT850_TrimMass50_v , &b_HLT_AK8PFHT850_TrimMass50_v);
    inputTree->SetBranchAddress("HLT_AK8PFHT900_TrimMass50_v", &HLT_AK8PFHT900_TrimMass50_v , &b_HLT_AK8PFHT900_TrimMass50_v);
    inputTree->SetBranchAddress("HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17_v", &HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17_v , &b_HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17_v);
    inputTree->SetBranchAddress("HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p1_v", &HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p1_v , &b_HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p1_v);
    inputTree->SetBranchAddress("HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2_v", &HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2_v , &b_HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2_v);
    inputTree->SetBranchAddress("HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4_v", &HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4_v , &b_HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4_v);
    inputTree->SetBranchAddress("HLT_AK8PFJet360_TrimMass30_v", &HLT_AK8PFJet360_TrimMass30_v , &b_HLT_AK8PFJet360_TrimMass30_v);
    inputTree->SetBranchAddress("HLT_AK8PFJet380_TrimMass30_v", &HLT_AK8PFJet380_TrimMass30_v , &b_HLT_AK8PFJet380_TrimMass30_v);
    inputTree->SetBranchAddress("HLT_AK8PFJet400_TrimMass30_v", &HLT_AK8PFJet400_TrimMass30_v , &b_HLT_AK8PFJet400_TrimMass30_v);
    inputTree->SetBranchAddress("HLT_AK8PFJet420_TrimMass30_v", &HLT_AK8PFJet420_TrimMass30_v , &b_HLT_AK8PFJet420_TrimMass30_v);
    inputTree->SetBranchAddress("HLT_AK8PFJet500_v", &HLT_AK8PFJet500_v , &b_HLT_AK8PFJet500_v);  
    inputTree->SetBranchAddress("HLT_HT430_DisplacedDijet40_DisplacedTrack_v", &HLT_HT430_DisplacedDijet40_DisplacedTrack_v , &b_HLT_HT430_DisplacedDijet40_DisplacedTrack_v);
    inputTree->SetBranchAddress("HLT_HT430_DisplacedDijet60_DisplacedTrack_v", &HLT_HT430_DisplacedDijet60_DisplacedTrack_v , &b_HLT_HT430_DisplacedDijet60_DisplacedTrack_v);
    inputTree->SetBranchAddress("HLT_HT500_DisplacedDijet40_DisplacedTrack_v", &HLT_HT500_DisplacedDijet40_DisplacedTrack_v , &b_HLT_HT500_DisplacedDijet40_DisplacedTrack_v);
    inputTree->SetBranchAddress("HLT_HT650_DisplacedDijet60_Inclusive_v", &HLT_HT650_DisplacedDijet60_Inclusive_v , &b_HLT_HT650_DisplacedDijet60_Inclusive_v);


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

    outputTree->Branch("HLT_AK8PFHT800_TrimMass50_v", &HLT_AK8PFHT800_TrimMass50_v , "HLT_AK8PFHT800_TrimMass50_v/O");
    outputTree->Branch("HLT_AK8PFHT850_TrimMass50_v", &HLT_AK8PFHT850_TrimMass50_v , "HLT_AK8PFHT850_TrimMass50_v/O");
    outputTree->Branch("HLT_AK8PFHT900_TrimMass50_v", &HLT_AK8PFHT900_TrimMass50_v , "HLT_AK8PFHT900_TrimMass50_v/O");
    outputTree->Branch("HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17_v", &HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17_v , "HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17_v/O");
    outputTree->Branch("HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p1_v", &HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p1_v , "HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p1_v/O");
    outputTree->Branch("HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2_v", &HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2_v , "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2_v/O");
    outputTree->Branch("HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4_v", &HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4_v , "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4_v/O");
    outputTree->Branch("HLT_AK8PFJet360_TrimMass30_v", &HLT_AK8PFJet360_TrimMass30_v , "HLT_AK8PFJet360_TrimMass30_v/O");
    outputTree->Branch("HLT_AK8PFJet380_TrimMass30_v", &HLT_AK8PFJet380_TrimMass30_v , "HLT_AK8PFJet380_TrimMass30_v/O");
    outputTree->Branch("HLT_AK8PFJet400_TrimMass30_v", &HLT_AK8PFJet400_TrimMass30_v , "HLT_AK8PFJet400_TrimMass30_v/O");
    outputTree->Branch("HLT_AK8PFJet420_TrimMass30_v", &HLT_AK8PFJet420_TrimMass30_v , "HLT_AK8PFJet420_TrimMass30_v/O");
    outputTree->Branch("HLT_AK8PFJet500_v", &HLT_AK8PFJet500_v , "HLT_AK8PFJet500_v/O");  
    outputTree->Branch("HLT_HT430_DisplacedDijet40_DisplacedTrack_v", &HLT_HT430_DisplacedDijet40_DisplacedTrack_v , "HLT_HT430_DisplacedDijet40_DisplacedTrack_v/O");
    outputTree->Branch("HLT_HT430_DisplacedDijet60_DisplacedTrack_v", &HLT_HT430_DisplacedDijet60_DisplacedTrack_v , "HLT_HT430_DisplacedDijet60_DisplacedTrack_v/O");
    outputTree->Branch("HLT_HT500_DisplacedDijet40_DisplacedTrack_v", &HLT_HT500_DisplacedDijet40_DisplacedTrack_v , "HLT_HT500_DisplacedDijet40_DisplacedTrack_v/O");
    outputTree->Branch("HLT_HT650_DisplacedDijet60_Inclusive_v", &HLT_HT650_DisplacedDijet60_Inclusive_v , "HLT_HT650_DisplacedDijet60_Inclusive_v/O");


    outputTree->Branch("GenHiggs", &GenHiggs);
    outputTree->Branch("GenLLPs", &GenLLPs);
    outputTree->Branch("GenBquarks", &GenBquarks);

    // Event loop

    //for(int i = 0; i < 1; i++) {
    for(int i = 0; i < inputTree->GetEntriesFast(); i++) {

	//Initialize nTagJets at every event
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

	outputTree->Fill();

    }

    // finalize files
    outputTree->SetWeight(tree_weight);
    counter->Write();
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
