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
#include "TTree.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects.h"
//#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/MyAnalysis.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <chrono>//compute time
#include <ctime>//compute time 


int main(int argc, char **argv) {

    if(argc<3)
      {
	std::cout<<"Invalid arguments, exit!" << std::endl;
	return 0;
      }
    std::cout << "Input file: " << argv[1] << std::endl;
    std::cout << "Output file: " << argv[2] << std::endl;
    std::cout << "Skip even EventNumber: " << argv[3] << std::endl;
    auto start = std::chrono::system_clock::now();//time!     

    std::string basePath = std::string(std::getenv("CMSSW_BASE")) + "/src/NNInferenceCMSSW/LLP_NN_Inference/nn_inference";

    // input and output file settings
    //unskimmed crab output
    //std::string inputPath = "/pnfs/desy.de/cms/tier2/store/user/lbenato/v4_calo_AOD_2018_18October2020/GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/201017_234633/0000/output_1.root";

    //smaller sample, not trained on this
    //std::string inputPath = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_skimAccept_unmerged/GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/output_1.root";
    //full sample to check time needed
    std::string inputPath = argv[1];
    //!!!//"/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_skimAccept_unmerged/GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche3_v2.root";

    std::string outputPath = basePath + argv[2];//!!!//"/test_on_real_ntuple.root";
    std::string inputTreeName = "skim";
    //std::string inputTreeName = "ntuple/tree";
    std::string outputTreeName = "tree";//inputTreeName;

    // model and inference settings
    std::string graphPathAK4 = basePath + "/tagger_v1/graph.pb";
    std::string MetaDataFileAK4 = basePath + "/tagger_v1/metadata.dat";
    std::string inputTensorNameAK4 = "input_input";
    std::string outputTensorNameAK4 = "FCN/output/Softmax";//"FCN/dense_4/Softmax";//or Softmax?
    //int nInputs = 10;

    // threading setup
    // to enable tensorflow-native multi-threading, change to "tensorflow" and increase nThreads
    std::string threadPool = "no_threads";
    int nThreads = 1;

    // open input file, read the tree and register input branches
    TFile* inputFile = new TFile(inputPath.c_str(), "READ");
    TTree* inputTree = (TTree*)inputFile->Get(inputTreeName.c_str());


    // Input branches
    int EventNumber;
    inputTree->SetBranchAddress("EventNumber", &EventNumber );
    int RunNumber;
    inputTree->SetBranchAddress("RunNumber", &RunNumber );
    int LumiNumber;
    inputTree->SetBranchAddress("LumiNumber", &LumiNumber );
    std::vector<JetType> *Jets = 0;
    TBranch *b_Jets = 0;
    inputTree->SetBranchAddress("Jets", &Jets, &b_Jets);

    //inputTree->SetBranchStatus("*",0);

    // Read jet input features from metadata.dat file
    std::ifstream fin;
    std::string feat;
    fin.open(MetaDataFileAK4);
    std::vector<std::string> featuresAK4;
    std::string toErase = "Jet_";

    while (fin >> feat)
      {
	size_t pos = feat.find(toErase);
	if (pos != std::string::npos)
	  {
	    // If found then erase it from string
	    feat.erase(pos, toErase.length());
	  }
	//std::string new_feat = feat.substr(position);
	std::cout << feat << std::endl;
	featuresAK4.push_back(feat);
      }
    fin.close();

    //inputTree->SetBranchStatus("Jets_pt",1);//needed?
    

    //do per branch
    //float pt;
    //inputTree->SetBranchAddress("Jets.pt", &pt );


    // This allows to activate only the needed branches
    //for(unsigned int f; f<features.size(); f++)
      //{
	//std::cout<<features.at(f)<<std::endl;
	//std::string tmp_feat = "Jets.";
	//tmp_feat.append(features.at(f)); 
	//std::cout<<tmp_feat<<std::endl;
	//char * cstr = new char [tmp_feat.length()+1];
	//std::strcpy (cstr, tmp_feat.c_str());
	//inputTree->SetBranchStatus(cstr,1);//needed?
      //}


    //const char* L = "Jets.ptAllTracks";
    //inputTree->SetBranchStatus(L,1);//needed?




    // Intermediate file with zero padded stuff?
    TFile* outputFile = new TFile(outputPath.c_str(), "RECREATE");
    outputFile->cd();
    TTree *outputTree = new TTree(outputTreeName.c_str(), "");

    int nTagJets_0p9(0);
    int nTagJets_0p95(0);
    int nTagJets_0p99(0); 
    outputTree->Branch("Jets", &Jets);
    outputTree->Branch("nTagJets_0p9",&nTagJets_0p9,"nTagJets_0p9/I");
    outputTree->Branch("nTagJets_0p95",&nTagJets_0p95,"nTagJets_0p95/I");
    outputTree->Branch("nTagJets_0p99",&nTagJets_0p99,"nTagJets_0p99/I");

    //do it as a loop
    //std::vector<float> Jet_0_inputValues(features.size());
    //Here loop and to the branch thing



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


    // Event loop

    for(int i = 0; i < inputTree->GetEntriesFast(); i++) {

        if (i % 100 == 0) {
            std::cout << "evaluating entry " << i << std::endl;
        }
        inputTree->GetEntry(i);

	if(strcmp(argv[3], "y")==1 || strcmp(argv[3], "yes")==1)
	  {
	    if (EventNumber % 2 == 0)
	      {
		std::cout << "Skip even EventNumber! " << std::endl;
		continue;
	      }
	  }
	//Initialize nTagJets at every event
	nTagJets_0p9 = 0;
	nTagJets_0p95 = 0;
	nTagJets_0p99 = 0;


	for (unsigned int j=0; j<Jets->size(); j++)
	  {
	    //std::cout<< "Jet n. " << j << " pt: " << Jets->at(j).pt << " ; sigprob: " << Jets->at(j).sigprob  << std::endl;
	    //here biuld the inputVector for each jet
	    std::vector<float> inputValues(featuresAK4.size());
	    inputValues.at(0) = Jets->at(j).nTrackConstituents;
	    inputValues.at(1) = Jets->at(j).nSelectedTracks;
	    inputValues.at(2) = Jets->at(j).timeRecHitsEB;
	    inputValues.at(3) = Jets->at(j).eFracRecHitsEB;
	    inputValues.at(4) = Jets->at(j).nRecHitsEB;
	    inputValues.at(5) = Jets->at(j).cHadEFrac;
	    inputValues.at(6) = Jets->at(j).nHadEFrac;
	    inputValues.at(7) = Jets->at(j).eleEFrac;
	    inputValues.at(8) = Jets->at(j).photonEFrac;
	    inputValues.at(9) = Jets->at(j).ptAllTracks;
	    inputValues.at(10) = Jets->at(j).ptAllPVTracks;
	    inputValues.at(11) = Jets->at(j).alphaMax;
	    inputValues.at(12) = Jets->at(j).betaMax;
	    inputValues.at(13) = Jets->at(j).gammaMax;
	    inputValues.at(14) = Jets->at(j).gammaMaxEM;
	    inputValues.at(15) = Jets->at(j).gammaMaxHadronic;
	    inputValues.at(16) = Jets->at(j).gammaMaxET;
	    inputValues.at(17) = Jets->at(j).minDeltaRAllTracks;
	    inputValues.at(18) = Jets->at(j).minDeltaRPVTracks;



	    float* d = inputTensorAK4.flat<float>().data();
	    for (float v : inputValues) {
	      //std::cout<< " input value: " << v <<std::endl;
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
	    if(Jets->at(j).pt<0) outputValueAK4 = -1;
	    Jets->at(j).sigprob = outputValueAK4;
	    if(outputValueAK4>=0.9) nTagJets_0p9++;
	    if(outputValueAK4>=0.95) nTagJets_0p95++;
	    if(outputValueAK4>=0.99) nTagJets_0p99++;

	    //std::cout<< "Post inference: " << std::endl;
	    //std::cout<< "Jet n. " << j << " pt: " << Jets->at(j).pt << "; genmatched: " << Jets->at(j).isGenMatchedCaloCorrLLPAccept  << " ; sigprob: " << Jets->at(j).sigprob  << std::endl;

	  }//Jet loop

        outputTree->Fill();
    }


    // finalize files
   
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

    return 0;
}
