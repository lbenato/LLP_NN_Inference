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
#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

int main() {
    std::string basePath = std::string(std::getenv("CMSSW_BASE")) + "/src/NNInferenceCMSSW/LLP_NN_Inference/nn_inference";

    // input and output file settings
    std::string inputPath = "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/model_weights/v3_calo_AOD_2018_dnn__v4_2018_5Upsampling_0p25Background/root_files/SUSY/AK4jets/model_FCN_2_EventWeightNormalized_NoMedian_NoRecHitsHB_NoSi_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2/TChiHH_mass400_pl1000.root";
    std::string outputPath = basePath + "/output_ntuple.root";
    std::string inputTreeName = "tree";
    std::string outputTreeName = inputTreeName;

    // model and inference settings
    std::string graphPath = basePath + "/graph_NoHCAL_NoSi.pb";
    std::string inputTensorName = "input_input";
    std::string outputTensorName = "FCN/output/Softmax";//"FCN/dense_4/Softmax";//or Softmax?
    //int nInputs = 10;

    // threading setup
    // to enable tensorflow-native multi-threading, change to "tensorflow" and increase nThreads
    std::string threadPool = "no_threads";
    int nThreads = 1;

    // open input file, read the tree and register input branches
    TFile* inputFile = new TFile(inputPath.c_str(), "READ");
    TTree* inputTree = (TTree*)inputFile->Get(inputTreeName.c_str());

    float EventNumber;
    inputTree->SetBranchAddress("EventNumber", &EventNumber );
    float RunNumber;
    inputTree->SetBranchAddress("RunNumber", &RunNumber );
    float LumiNumber;
    inputTree->SetBranchAddress("LumiNumber", &LumiNumber );

    //per-jet inputs
    //std::vector<std::string> inputFeatures_0 = { "Jet_0_nTrackConstituents","Jet_0_nSelectedTracks", "Jet_0_timeRecHitsEB", "Jet_0_timeRecHitsHB", "Jet_0_energyRecHitsEB", "Jet_0_energyRecHitsHB", "Jet_0_nRecHitsEB", "Jet_0_nRecHitsHB", "Jet_0_cHadEFrac", "Jet_0_nHadEFrac", "Jet_0_eleEFrac", "Jet_0_photonEFrac", "Jet_0_ptAllTracks", "Jet_0_ptAllPVTracks", "Jet_0_ptPVTracksMax", "Jet_0_nTracksAll", "Jet_0_nTracksPVMax", "Jet_0_alphaMax", "Jet_0_betaMax", "Jet_0_gammaMax", "Jet_0_gammaMaxEM", "Jet_0_gammaMaxHadronic", "Jet_0_gammaMaxET","Jet_0_minDeltaRAllTracks","Jet_0_minDeltaRPVTracks",};
    //std::vector<std::string> inputFeatures_1 = { "Jet_1_nTrackConstituents","Jet_1_nSelectedTracks", "Jet_1_timeRecHitsEB", "Jet_1_timeRecHitsHB", "Jet_1_energyRecHitsEB", "Jet_1_energyRecHitsHB", "Jet_1_nRecHitsEB", "Jet_1_nRecHitsHB", "Jet_1_cHadEFrac", "Jet_1_nHadEFrac", "Jet_1_eleEFrac", "Jet_1_photonEFrac", "Jet_1_ptAllTracks", "Jet_1_ptAllPVTracks", "Jet_1_ptPVTracksMax", "Jet_1_nTracksAll", "Jet_1_nTracksPVMax", "Jet_1_alphaMax", "Jet_1_betaMax", "Jet_1_gammaMax", "Jet_1_gammaMaxEM", "Jet_1_gammaMaxHadronic", "Jet_1_gammaMaxET","Jet_1_minDeltaRAllTracks","Jet_1_minDeltaRPVTracks",};
    //std::vector<std::string> inputFeatures_2 = { "Jet_2_nTrackConstituents","Jet_2_nSelectedTracks", "Jet_2_timeRecHitsEB", "Jet_2_timeRecHitsHB", "Jet_2_energyRecHitsEB", "Jet_2_energyRecHitsHB", "Jet_2_nRecHitsEB", "Jet_2_nRecHitsHB", "Jet_2_cHadEFrac", "Jet_2_nHadEFrac", "Jet_2_eleEFrac", "Jet_2_photonEFrac", "Jet_2_ptAllTracks", "Jet_2_ptAllPVTracks", "Jet_2_ptPVTracksMax", "Jet_2_nTracksAll", "Jet_2_nTracksPVMax", "Jet_2_alphaMax", "Jet_2_betaMax", "Jet_2_gammaMax", "Jet_2_gammaMaxEM", "Jet_2_gammaMaxHadronic", "Jet_2_gammaMaxET","Jet_2_minDeltaRAllTracks","Jet_2_minDeltaRPVTracks",};
    std::vector<std::string> inputFeatures_0 = { "Jet_0_nTrackConstituents","Jet_0_nSelectedTracks", "Jet_0_timeRecHitsEB", "Jet_0_energyRecHitsEB", "Jet_0_nRecHitsEB", "Jet_0_cHadEFrac", "Jet_0_nHadEFrac", "Jet_0_eleEFrac", "Jet_0_photonEFrac", "Jet_0_ptAllTracks", "Jet_0_ptAllPVTracks", "Jet_0_alphaMax", "Jet_0_betaMax", "Jet_0_gammaMax", "Jet_0_gammaMaxEM", "Jet_0_gammaMaxHadronic", "Jet_0_gammaMaxET","Jet_0_minDeltaRAllTracks","Jet_0_minDeltaRPVTracks",};
    std::vector<std::string> inputFeatures_1 = { "Jet_1_nTrackConstituents","Jet_1_nSelectedTracks", "Jet_1_timeRecHitsEB", "Jet_1_energyRecHitsEB",  "Jet_1_nRecHitsEB", "Jet_1_cHadEFrac", "Jet_1_nHadEFrac", "Jet_1_eleEFrac", "Jet_1_photonEFrac", "Jet_1_ptAllTracks", "Jet_1_ptAllPVTracks", "Jet_1_alphaMax", "Jet_1_betaMax", "Jet_1_gammaMax", "Jet_1_gammaMaxEM", "Jet_1_gammaMaxHadronic", "Jet_1_gammaMaxET","Jet_1_minDeltaRAllTracks","Jet_1_minDeltaRPVTracks",};
    std::vector<std::string> inputFeatures_2 = { "Jet_2_nTrackConstituents","Jet_2_nSelectedTracks", "Jet_2_timeRecHitsEB", "Jet_2_energyRecHitsEB", "Jet_2_nRecHitsEB", "Jet_2_cHadEFrac", "Jet_2_nHadEFrac", "Jet_2_eleEFrac", "Jet_2_photonEFrac", "Jet_2_ptAllTracks", "Jet_2_ptAllPVTracks", "Jet_2_alphaMax", "Jet_2_betaMax", "Jet_2_gammaMax", "Jet_2_gammaMaxEM", "Jet_2_gammaMaxHadronic", "Jet_2_gammaMaxET","Jet_2_minDeltaRAllTracks","Jet_2_minDeltaRPVTracks",};

    // the input branches are named "f0" to "f9", so use a loop in this example
    int nInputs_0 = inputFeatures_0.size();
    std::vector<float> inputValues_0(nInputs_0);
    for (size_t i = 0; i < inputFeatures_0.size(); i++) {
        std::string branchName_0 = inputFeatures_0.at(i);
        inputTree->SetBranchAddress(branchName_0.c_str(), &inputValues_0[i] );
    }

    float prob_0;
    inputTree->SetBranchAddress("Jet_0_sigprob", &prob_0 );
    float pt_0;
    inputTree->SetBranchAddress("Jet_0_pt", &pt_0 );


    int nInputs_1 = inputFeatures_1.size();
    std::vector<float> inputValues_1(nInputs_1);
    for (size_t i = 0; i < inputFeatures_1.size(); i++) {
        std::string branchName_1 = inputFeatures_1.at(i);
        inputTree->SetBranchAddress(branchName_1.c_str(), &inputValues_1[i] );
    }

    float prob_1;
    inputTree->SetBranchAddress("Jet_1_sigprob", &prob_1 );
    float pt_1;
    inputTree->SetBranchAddress("Jet_1_pt", &pt_1 );

    int nInputs_2 = inputFeatures_2.size();
    std::vector<float> inputValues_2(nInputs_2);
    for (size_t i = 0; i < inputFeatures_2.size(); i++) {
        std::string branchName_2 = inputFeatures_2.at(i);
        inputTree->SetBranchAddress(branchName_2.c_str(), &inputValues_2[i] );
    }

    float prob_2;
    inputTree->SetBranchAddress("Jet_2_sigprob", &prob_2 );
    float pt_2;
    inputTree->SetBranchAddress("Jet_2_pt", &pt_2 );


    //Read probability    

    //here we need a cast back to float?
    // create the output file, tree and a new branch
    TFile* outputFile = new TFile(outputPath.c_str(), "RECREATE");
    outputFile->cd();
    TTree *outputTree = new TTree(outputTreeName.c_str(), "");
    float outputValue_0;
    outputTree->Branch("model_output_0", &outputValue_0, "model_output_0/F");
    float outputValue_1;
    outputTree->Branch("model_output_1", &outputValue_1, "model_output_1/F");
    float outputValue_2;
    outputTree->Branch("model_output_2", &outputValue_2, "model_output_2/F");


    // setup TensorFlow objects
    tensorflow::setLogging();
    tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(graphPath);
    // TF < 2
    //tensorflow::SessionOptions sessionOptions;
    //tensorflow::setThreading(sessionOptions, nThreads, threadPool);
    //tensorflow::Session* session = tensorflow::createSession(graphDef, sessionOptions);
    // TF >= 2
    tensorflow::Session* session_0 = tensorflow::createSession(graphDef, nThreads);
    tensorflow::Session* session_1 = tensorflow::createSession(graphDef, nThreads);
    tensorflow::Session* session_2 = tensorflow::createSession(graphDef, nThreads);


    // register an input tensor (1 x nInputs_0) that is filled during the event loop
    tensorflow::Tensor inputTensor_0(tensorflow::DT_FLOAT, {1, nInputs_0});
    tensorflow::Tensor inputTensor_1(tensorflow::DT_FLOAT, {1, nInputs_1});
    tensorflow::Tensor inputTensor_2(tensorflow::DT_FLOAT, {1, nInputs_2});
       
    // loop over input events
    //for(int i = 0; i < inputTree->GetEntries(); i++) {
    for(int i = 0; i < 20; i++) {
        if (i % 100 == 0) {
            std::cout << "evaluating entry " << i << std::endl;
        }
        inputTree->GetEntry(i);

        // fill the input tensor using a data pointer that is shifted consecutively
        float* d_0 = inputTensor_0.flat<float>().data();
        for (float v_0 : inputValues_0) {
	    //std::cout<< " input value: " << v <<std::endl;
            *d_0 = v_0;
            d_0++;
        }
        float* d_1 = inputTensor_1.flat<float>().data();
        for (float v_1 : inputValues_1) {
            *d_1 = v_1;
            d_1++;
        }
        float* d_2 = inputTensor_2.flat<float>().data();
        for (float v_2 : inputValues_2) {
            *d_2 = v_2;
            d_2++;
        }


        // run the inference
        std::vector<tensorflow::Tensor> outputs_0;
        std::vector<tensorflow::Tensor> outputs_1;
        std::vector<tensorflow::Tensor> outputs_2;
        // TF < 2
        //tensorflow::run(session, {{inputTensorName, inputTensor}}, {outputTensorName}, &outputs);
        // TF >= 2
        tensorflow::run(session_0, {{inputTensorName, inputTensor_0}}, {outputTensorName}, &outputs_0, threadPool);
        tensorflow::run(session_1, {{inputTensorName, inputTensor_1}}, {outputTensorName}, &outputs_1, threadPool);
        tensorflow::run(session_2, {{inputTensorName, inputTensor_2}}, {outputTensorName}, &outputs_2, threadPool);

        // store the result
        outputValue_0 = outputs_0[0].matrix<float>()(0, 1);
        outputValue_1 = outputs_1[0].matrix<float>()(0, 1);
        outputValue_2 = outputs_2[0].matrix<float>()(0, 1);
	// keras cannot predict the output for invalid jets
	// fix it manually
	if(pt_0<0) outputValue_0 = -1;
	if(pt_1<0) outputValue_1 = -1;
	if(pt_2<0) outputValue_2 = -1;
	std::cout << "EventNumber: " << int(EventNumber) << "; RunNumber: " << RunNumber << "; LumiNumber: " << LumiNumber << std::endl;
	std::cout << "original prob[0]: " << prob_0 << std::endl;
	std::cout << "original prob[1]: " << prob_1 << std::endl;
	std::cout << "original prob[2]: " << prob_2 << std::endl;
	std::cout << "Jet[0] pt: " << pt_0 << "; output DNN value: " << outputValue_0 << std::endl;
	std::cout << "Jet[1] pt: " << pt_1 << "; output DNN value: " << outputValue_1 << std::endl;
	std::cout << "Jet[2] pt: " << pt_2 << "; output DNN value: " << outputValue_2 << std::endl;
	std::cout << "\n" << std::endl;
        outputTree->Fill();
    }
    
    // finalize files
   
    outputFile->Write();
    outputFile->Close();
    inputFile->Close();

    return 0;
    
}
