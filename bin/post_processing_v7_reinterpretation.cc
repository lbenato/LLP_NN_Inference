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
//#include "TLinearFitter.h"

#include <cassert>
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v6.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/CaloObjects_v6.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/dbscan.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <chrono>//compute time
#include <ctime>//compute time 

using namespace ROOT::Math;
using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {

    if(argc<4)
    //if(argc<2)
      {
	std::cout<<"Invalid arguments, exit!" << std::endl;
	return 0;
      }

    //needed argv: input, output, a sting to tell the year for phi selections, a string for time smearing

    std::cout << "Input file: " << argv[1] << std::endl;
    std::cout << "Output file: " << argv[2] << std::endl;
    std::cout << "ERA string: " << argv[3] << std::endl;
    std::cout << "Time smearing string: " << argv[4] << std::endl;

    auto start = std::chrono::system_clock::now();//time!     
    std::string basePath = std::string(std::getenv("CMSSW_BASE")) + "/src/NNInferenceCMSSW/LLP_NN_Inference/nn_inference";

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    std::string eraString = argv[3];

    float minphi(-9999.);
    float maxphi( 9999.);
    std::string tag_2017 = "2017";
    std::string tag_2018 = "2018";
    if(eraString.find(tag_2017) != std::string::npos)
      {
	minphi = 3.5;
	maxphi = 2.7;
      }
    if(eraString.find(tag_2018) != std::string::npos)
      {
	minphi = 0.9;
	maxphi = 0.4;
      }

    std::cout << "minphi: " << minphi << std::endl;
    std::cout << "maxphi: " << maxphi << std::endl;

    std::string timeCBFilename = argv[4];

    std::string inputTreeName = "tree";
    std::string outputTreeName = "tree";//inputTreeName;

    // model and inference settings
    std::string graphPathAK4 = basePath + "/tagger_AK4_v3/graph.pb";
    std::string MetaDataFileAK4 = basePath + "/tagger_AK4_v3/metadata.dat";
    std::string inputTensorNameAK4 = "input_input";
    std::string outputTensorNameAK4 = "FCN/output/Softmax";//"FCN/dense_4/Softmax";//or Softmax?
    //int nInputs = 10;

    // threading setup
    // to enable tensorflow-native multi-threading, change to "tensorflow" and increase nThreads
    std::string threadPool = "no_threads";
    int nThreads = 1;

    // ================= 
    // Input
    // ================= 

    // open input file, read the tree and register input branches
    TFile* inputFile = new TFile(inputPath.c_str(), "READ");
    TTree* inputTree = (TTree*)inputFile->Get(inputTreeName.c_str());
    TH1F   *counter = (TH1F*)inputFile->Get("c_nEvents");
    TH1F   *n_pass = (TH1F*)inputFile->Get("n_pass");
    TH1F   *n_odd = (TH1F*)inputFile->Get("n_odd");
    TH1F   *n_even = (TH1F*)inputFile->Get("n_even");

    float  tree_weight = inputTree->GetWeight();

    TFile *timeCBFile = TFile::Open(timeCBFilename.data(),"READ"); if (!timeCBFile) return 0;
    TF1  *dataCB = (TF1*)timeCBFile->Get("data_CB");
    TF1  *mcCB = (TF1*)timeCBFile->Get("back_CB");

    TF1 *smearCB = (TF1*)dataCB->Clone("smear_cb");
    smearCB->SetParameter(0,dataCB->GetParameter(0));
    smearCB->SetParameter(1,dataCB->GetParameter(1) - mcCB->GetParameter(1));
    smearCB->SetParameter(2, sqrt( abs( pow(dataCB->GetParameter(2),2) - pow(mcCB->GetParameter(2),2) )) );
    smearCB->SetParameter(3,dataCB->GetParameter(3));
    smearCB->SetParameter(4,dataCB->GetParameter(4));


    // Read jet input features from metadata.dat file
    //AK4
    std::ifstream finAK4;
    std::string featAK4;
    finAK4.open(MetaDataFileAK4);
    std::vector<std::string> featuresAK4;
    std::string toEraseAK4 = "Jet_";
    //std::cout << "   -- > Features AK4: " << std::endl;
    while (finAK4 >> featAK4)
      {
	size_t pos = featAK4.find(toEraseAK4);
	if (pos != std::string::npos)
	  {
	    // If found then erase it from string
	    featAK4.erase(pos, toEraseAK4.length());
	  }
	//std::string new_feat = featAK4.substr(position);
	//std::cout << featAK4 << std::endl;
	featuresAK4.push_back(featAK4);
      }
    finAK4.close();

    //Deactivate branches you don't want to copy
    //inputTree->SetBranchStatus("*",1);
    inputTree->SetBranchStatus("Jets.*",0);
    inputTree->SetBranchStatus("JetsNegative.*",0);
    inputTree->SetBranchStatus("JetsNum.*",0);
    inputTree->SetBranchStatus("JetsDen.*",0);
    inputTree->SetBranchStatus("JetsFullEta.*",0);
    inputTree->SetBranchStatus("JetsMatch.*",0);
    inputTree->SetBranchStatus("JetsAcceptanceMatch.*",0);
    inputTree->SetBranchStatus("Muons.*",0);
    inputTree->SetBranchStatus("Electrons.*",0);
    inputTree->SetBranchStatus("Taus.*",0);
    inputTree->SetBranchStatus("CSCSegments.*",0);
    inputTree->SetBranchStatus("DTSegments.*",0);
    inputTree->SetBranchStatus("EcalRecHitsAK4.*",0);
    inputTree->SetBranchStatus("skimmedEcalRecHitsAK4.*",0);
    inputTree->SetBranchStatus("skimmedAcceptanceEcalRecHitsAK4.*",0);
    inputTree->SetBranchStatus("taggedEcalRecHitsAK4.*",0);
    inputTree->SetBranchStatus("taggedAcceptanceEcalRecHitsAK4.*",0);
    inputTree->SetBranchStatus("DT_fit_*",0);

    // ================= 
    // Output
    // ================= 

    //inputTree->Print();
    std::cout << "Copying old tree..." << std::endl;


    TFile* outputFile = new TFile(outputPath.c_str(), "RECREATE");
    outputFile->cd();
    //TTree *outputTree = new TTree(outputTreeName.c_str(), "");
    //inputTree->SetBranchStatus("*Jets*",0);
    //JetsFullEta, Jets, JetsNegative, JetsMatch, JetsAcceptanceMatch, JetsNum, JetsDen
    //Muons, Electrons, Taus, CSCSegments, DTSegments

    TTree *outputTree = inputTree->CloneTree(0);
    //outputTree->CopyEntries(inputTree);

    //Activate input branches you have not copied but that you need
    inputTree->SetBranchStatus("JetsNegative.*",1);


    // setup TensorFlow objects
    tensorflow::setLogging();
    tensorflow::GraphDef* graphDefAK4 = tensorflow::loadGraphDef(graphPathAK4);
    // TF < 2
    //tensorflow::SessionOptions sessionOptions;
    //tensorflow::setThreading(sessionOptions, nThreads, threadPool);
    //tensorflow::Session* session = tensorflow::createSession(graphDef, sessionOptions);
    // TF >= 2
    tensorflow::Session* sessionAK4 = tensorflow::createSession(graphDefAK4, nThreads);
    tensorflow::Tensor inputTensorAK4(tensorflow::DT_FLOAT, {1, int(featuresAK4.size()) });
    float outputValueAK4;

    //Branch I want to change
    bool   old_isPreselected;
    TBranch *b_old_isPreselected;
    inputTree->SetBranchAddress("isPreselected", &old_isPreselected, &b_old_isPreselected);

    float   oldHT;
    TBranch *b_oldHT;
    inputTree->SetBranchAddress("HT", &oldHT, &b_oldHT);

    int   old_nJets_to_LLP0;
    TBranch *b_old_nJets_to_LLP0;
    inputTree->SetBranchAddress("nJets_to_LLP0", &old_nJets_to_LLP0, &b_old_nJets_to_LLP0);

    int   old_nJets_to_LLP1;
    TBranch *b_old_nJets_to_LLP1;
    inputTree->SetBranchAddress("nJets_to_LLP1", &old_nJets_to_LLP1, &b_old_nJets_to_LLP1);

    int   old_nTagJets_to_LLP0;
    TBranch *b_old_nTagJets_to_LLP0;
    inputTree->SetBranchAddress("nTagJets_to_LLP0", &old_nTagJets_to_LLP0, &b_old_nTagJets_to_LLP0);

    int   old_nTagJets_to_LLP1;
    TBranch *b_old_nTagJets_to_LLP1;
    inputTree->SetBranchAddress("nTagJets_to_LLP1", &old_nTagJets_to_LLP1, &b_old_nTagJets_to_LLP1);

    int   oldTag;
    TBranch *b_oldTag;
    inputTree->SetBranchAddress("nTagJets_0p996", &oldTag, &b_oldTag);

    std::vector<JetType>  *oldJets = 0;
    TBranch               *b_oldJets = 0;
    inputTree->SetBranchAddress("JetsNegative", &oldJets, &b_oldJets);

    std::vector<GenPType>  *GenLLPs = 0;
    TBranch               *b_GenLLPs = 0;
    inputTree->SetBranchAddress("GenLLPs", &GenLLPs, &b_GenLLPs);

    //Input branches needed to redo preselections
    bool   HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v(false);
    bool   HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v(false);
    bool   HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v(false);
    bool   HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v(false);
    bool   Flag2_globalSuperTightHalo2016Filter;
    bool   Flag2_goodVertices;
    bool   Flag2_EcalDeadCellTriggerPrimitiveFilter;
    bool   Flag2_HBHENoiseFilter;
    bool   Flag2_HBHEIsoNoiseFilter;
    bool   Flag2_ecalBadCalibFilter;
    bool   Flag2_eeBadScFilter;
    bool   Flag2_BadPFMuonFilter;
    MEtType *MEt = 0;
    int nMuonsPassing;
    int nElectronsPassing;
    int nPhotonsPassing;
    int nTausPassing;
    float dt_ecal_dist;
    float MinJetMetDPhi;
    float min_dPhi_jets_eta_1p0_0p996;

    TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v;
    TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v;
    TBranch        *b_HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v;
    TBranch        *b_HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v;
    TBranch        *b_Flag2_globalSuperTightHalo2016Filter;
    TBranch        *b_Flag2_goodVertices;
    TBranch        *b_Flag2_EcalDeadCellTriggerPrimitiveFilter;
    TBranch        *b_Flag2_HBHENoiseFilter;
    TBranch        *b_Flag2_HBHEIsoNoiseFilter;
    TBranch        *b_Flag2_ecalBadCalibFilter;
    TBranch        *b_Flag2_eeBadScFilter;
    TBranch        *b_Flag2_BadPFMuonFilter;
    TBranch        *b_MEt = 0;
    TBranch        *b_nMuonsPassing;
    TBranch        *b_nElectronsPassing;
    TBranch        *b_nPhotonsPassing;
    TBranch        *b_nTausPassing;
    TBranch        *b_dt_ecal_dist;
    TBranch        *b_MinJetMetDPhi;
    TBranch        *b_min_dPhi_jets_eta_1p0_0p996;

    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v,&b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, &b_HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v, &b_HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v);
    inputTree->SetBranchAddress("Flag2_globalSuperTightHalo2016Filter", &Flag2_globalSuperTightHalo2016Filter, &b_Flag2_globalSuperTightHalo2016Filter);
    inputTree->SetBranchAddress("Flag2_goodVertices", &Flag2_goodVertices, &b_Flag2_goodVertices);
    inputTree->SetBranchAddress("Flag2_EcalDeadCellTriggerPrimitiveFilter", &Flag2_EcalDeadCellTriggerPrimitiveFilter, &b_Flag2_EcalDeadCellTriggerPrimitiveFilter);
    inputTree->SetBranchAddress("Flag2_HBHENoiseFilter", &Flag2_HBHENoiseFilter, &b_Flag2_HBHENoiseFilter);
    inputTree->SetBranchAddress("Flag2_HBHEIsoNoiseFilter", &Flag2_HBHEIsoNoiseFilter, &b_Flag2_HBHEIsoNoiseFilter);
    inputTree->SetBranchAddress("Flag2_ecalBadCalibFilter", &Flag2_ecalBadCalibFilter, &b_Flag2_ecalBadCalibFilter);
    inputTree->SetBranchAddress("Flag2_eeBadScFilter", &Flag2_eeBadScFilter, &b_Flag2_eeBadScFilter);
    inputTree->SetBranchAddress("Flag2_BadPFMuonFilter", &Flag2_BadPFMuonFilter, &b_Flag2_BadPFMuonFilter);

    inputTree->SetBranchAddress("MEt",               &MEt,               &b_MEt);
    inputTree->SetBranchAddress("nMuonsPassing", &nMuonsPassing, &b_nMuonsPassing);
    inputTree->SetBranchAddress("nElectronsPassing", &nElectronsPassing, &b_nElectronsPassing);
    inputTree->SetBranchAddress("nPhotonsPassing", &nPhotonsPassing, &b_nPhotonsPassing);
    inputTree->SetBranchAddress("nTausPassing", &nTausPassing, &b_nTausPassing);
    inputTree->SetBranchAddress("dt_ecal_dist", &dt_ecal_dist, &b_dt_ecal_dist);
    inputTree->SetBranchAddress("MinJetMetDPhi", &MinJetMetDPhi, &b_MinJetMetDPhi);
    inputTree->SetBranchAddress("min_dPhi_jets_eta_1p0_0p996", &min_dPhi_jets_eta_1p0_0p996, &b_min_dPhi_jets_eta_1p0_0p996);

    //Output branches
    bool   isPreselected;
    TBranch *b_isPreselected;
    outputTree->SetBranchAddress("isPreselected", &isPreselected, &b_isPreselected);

    float   HT;
    TBranch *b_HT;
    outputTree->SetBranchAddress("HT", &HT, &b_HT);

    int   nJets_to_LLP0;
    TBranch *b_nJets_to_LLP0;
    outputTree->SetBranchAddress("nJets_to_LLP0", &nJets_to_LLP0, &b_nJets_to_LLP0);

    int   nJets_to_LLP1;
    TBranch *b_nJets_to_LLP1;
    outputTree->SetBranchAddress("nJets_to_LLP1", &nJets_to_LLP1, &b_nJets_to_LLP1);

    int   nTagJets_to_LLP0;
    TBranch *b_nTagJets_to_LLP0;
    outputTree->SetBranchAddress("nTagJets_to_LLP0", &nTagJets_to_LLP0, &b_nTagJets_to_LLP0);

    int   nTagJets_to_LLP1;
    TBranch *b_nTagJets_to_LLP1;
    outputTree->SetBranchAddress("nTagJets_to_LLP1", &nTagJets_to_LLP1, &b_nTagJets_to_LLP1);


    int   nTagJets_0p996;
    TBranch *b_nTagJets_0p996;
    outputTree->SetBranchAddress("nTagJets_0p996",    &nTagJets_0p996,                &b_nTagJets_0p996);

    //New branches
    outputTree->Branch("HT_old", &oldHT, "HT_old/F");
    outputTree->Branch("nTagJets_0p996_old", &oldTag, "nTagJets_0p996_old/I");

    std::vector<float> smearer_vector;
    outputTree->Branch("smearer_vector", &smearer_vector);

    std::vector<JetType>    skimmedJets;
    outputTree->Branch("Jets", &skimmedJets);

    // Event loop
    
    //for(int i = 0; i < 10; i++) {
    for(int i = 0; i < inputTree->GetEntriesFast(); i++) 

      {

	//clear collections
        skimmedJets.clear();
	smearer_vector.clear();

        inputTree->GetEntry(i);
	HT = -2.;	
	isPreselected = false;
	nTagJets_0p996 = 0;
	nTagJets_to_LLP0 = 0;
	nTagJets_to_LLP1 = 0;
	nJets_to_LLP0 = 0;
	nJets_to_LLP1 = 0;

	int n_jets_tagged_close_phi_low_multi (0);

	//Redo preselections as they seem messed up....

	// 
	for (unsigned int j=0; j<oldJets->size(); j++)
	  {

	    //Time smearing!
	    //We need:
	    //1. data file --> can pass from python
	    //2. signal file fit --> can pass from python
	    //3. name of the CB --> can pass from python

	    //Don't remove jets with eta>1 now, it messes up with the cosmic veto!
	    if( oldJets->at(j).pt>30 and fabs(oldJets->at(j).eta)<1.48 and oldJets->at(j).timeRecHitsEB>-100. and oldJets->at(j).muEFrac<0.6 and oldJets->at(j).eleEFrac<0.6 and oldJets->at(j).photonEFrac<0.8)//cleaned jets!

	      {

		//Time smearing here
		// ** commented as suboptimal **
		float pre_time = oldJets->at(j).timeRecHitsEB;
		float smearer = smearCB->GetRandom();
		smearer_vector.push_back(smearer);
		//std::cout << "Event n. " << i << " ; jet n. " << j << std::endl;
		//std::cout << smearer << std::endl;
		////Keep also the original time if needed
		oldJets->at(j).timeRecHitsHB = pre_time;
		oldJets->at(j).timeRecHitsEB = pre_time + smearer;

		//here build the inputVector for each jet
		std::vector<float> inputValues(featuresAK4.size());
		//std::vector<float> inputValuesUnsmeared(featuresAK4.size());

		//tagger_AK4_v3
		inputValues.at(0) = oldJets->at(j).nTrackConstituents;
		inputValues.at(1) = oldJets->at(j).nSelectedTracks;
		inputValues.at(2) = oldJets->at(j).timeRecHitsEB;
		inputValues.at(3) = oldJets->at(j).eFracRecHitsEB;
		inputValues.at(4) = oldJets->at(j).nRecHitsEB;
		inputValues.at(5) = oldJets->at(j).sig1EB;
		inputValues.at(6) = oldJets->at(j).sig2EB;
		inputValues.at(7) = oldJets->at(j).ptDEB;
		inputValues.at(8) = oldJets->at(j).cHadEFrac;
		inputValues.at(9) = oldJets->at(j).nHadEFrac;
		inputValues.at(10) = oldJets->at(j).eleEFrac;
		inputValues.at(11) = oldJets->at(j).photonEFrac;
		inputValues.at(12) = oldJets->at(j).ptAllTracks;
		inputValues.at(13) = oldJets->at(j).ptAllPVTracks;
		inputValues.at(14) = oldJets->at(j).alphaMax;
	        inputValues.at(15) = oldJets->at(j).betaMax;
		inputValues.at(16) = oldJets->at(j).gammaMax;
		inputValues.at(17) = oldJets->at(j).gammaMaxEM;
		inputValues.at(18) = oldJets->at(j).gammaMaxHadronic;
		inputValues.at(19) = oldJets->at(j).gammaMaxET;
		inputValues.at(20) = oldJets->at(j).minDeltaRAllTracks;
		inputValues.at(21) = oldJets->at(j).minDeltaRPVTracks;


		float* d = inputTensorAK4.flat<float>().data();
		for (float v : inputValues) {
		  *d = v;
		  d++;
		}

		// run the inference
		std::vector<tensorflow::Tensor> outputsAK4;
		tensorflow::run(sessionAK4, {{inputTensorNameAK4, inputTensorAK4}}, {outputTensorNameAK4}, &outputsAK4, threadPool);

		// store the result
		outputValueAK4 = outputsAK4[0].matrix<float>()(0, 1);
		// keras cannot predict the output for invalid jets
		// fix it manually
		if(oldJets->at(j).pt<0) outputValueAK4 = -1;
		oldJets->at(j).sigprob = outputValueAK4;

		//eta 1
		if( abs(oldJets->at(j).eta) < 1. )
		  {
		    //reject phi pathological regions
		    if( oldJets->at(j).phi>minphi or oldJets->at(j).phi<maxphi )
		      {
			//time cut after smearing
			if (oldJets->at(j).timeRecHitsEB>-1.)
			  {
			    skimmedJets.push_back(oldJets->at(j));

			    if(GenLLPs->at(0).travelRadius==oldJets->at(j).radiusLLP or GenLLPs->at(0).travelRadius==oldJets->at(j).radiusLLPCaloCorr) nJets_to_LLP0++;
			    if(GenLLPs->at(1).travelRadius==oldJets->at(j).radiusLLP or GenLLPs->at(1).travelRadius==oldJets->at(j).radiusLLPCaloCorr) nJets_to_LLP1++;

			    if(outputValueAK4>0.996 and oldJets->at(j).muEFrac<0.6 and oldJets->at(j).eleEFrac<0.6 and oldJets->at(j).photonEFrac<0.8 and oldJets->at(j).timeRecHitsEB>-1 and abs(oldJets->at(j).eta)<1.) 
			      {
				nTagJets_0p996++;
				if(GenLLPs->at(0).travelRadius==oldJets->at(j).radiusLLP or GenLLPs->at(0).travelRadius==oldJets->at(j).radiusLLPCaloCorr) nTagJets_to_LLP0++;
				if(GenLLPs->at(1).travelRadius==oldJets->at(j).radiusLLP or GenLLPs->at(1).travelRadius==oldJets->at(j).radiusLLPCaloCorr) nTagJets_to_LLP1++;

				if(oldJets->at(j).nRecHitsEB<=10 and min_dPhi_jets_eta_1p0_0p996<0.05) n_jets_tagged_close_phi_low_multi++;

			      }//tag requirement
			  }//time

		      }//phi
		  }//eta


	      }//jet cleaning

	  }//jet loop
	//

	//isPreselected fix
	if( (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v or HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v or HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v) and MEt->pt>200 and nMuonsPassing==0 and nElectronsPassing==0 and nTausPassing==0 and nPhotonsPassing==0 and dt_ecal_dist>0.5 and MinJetMetDPhi>0.5 and n_jets_tagged_close_phi_low_multi==0) isPreselected = true;

	//Branches I don't want to store:
	//b_HT->Clear();
	outputTree->Fill();

      }//event loop
    
    std::cout<<"Filled output:"<<std::endl;
    //outputTree->Print("nTagJets_0p996*");

    // finalize files
    outputTree->SetWeight(tree_weight);
    counter->Write();
    n_pass->Write();
    n_odd->Write();
    n_even->Write();

    outputFile->Write();
    outputFile->Close();
    inputFile->Close();
    

    auto end = std::chrono::system_clock::now();//time!
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "**************************************************" << std::endl;
    std::cout << "finished  computations at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << " s\n";
    std::cout << "**************************************************" << std::endl;
    std::cout << " " << std::endl;



    std::cout << "**************************************************" << std::endl;
    std::cout << "Output written: " << outputPath << std::endl;
    std::cout << "\n" << std::endl;

    return 0;
}
