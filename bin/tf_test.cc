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
    std::string inputPath = basePath + "/test_LLP_signal.root";
    std::string outputPath = basePath + "/output.root";
    std::string inputTreeName = "tree";
    std::string outputTreeName = inputTreeName;

    // model and inference settings
    std::string graphPath = basePath + "/graph.pb";
    std::string inputTensorName = "dense_input";
    std::string outputTensorName = "FCN/dense_4/Softmax";//"FCN/dense_4/Softmax";//or Softmax?
    //int nInputs = 10;

    // threading setup
    // to enable tensorflow-native multi-threading, change to "tensorflow" and increase nThreads
    std::string threadPool = "no_threads";
    int nThreads = 1;

    // open input file, read the tree and register input branches
    TFile* inputFile = new TFile(inputPath.c_str(), "READ");
    TTree* inputTree = (TTree*)inputFile->Get(inputTreeName.c_str());

    std::vector<std::string> inputFeatures = { "Jet_nTrackConstituents","Jet_nSelectedTracks", "Jet_timeRecHitsEB", "Jet_timeRecHitsHB", "Jet_energyRecHitsEB", "Jet_energyRecHitsHB", "Jet_nRecHitsEB", "Jet_nRecHitsHB", "Jet_cHadEFrac", "Jet_nHadEFrac", "Jet_eleEFrac", "Jet_photonEFrac", "Jet_ptAllTracks", "Jet_ptAllPVTracks", "Jet_ptPVTracksMax", "Jet_nTracksAll", "Jet_nTracksPVMax", "Jet_alphaMax", "Jet_betaMax", "Jet_gammaMax", "Jet_gammaMaxEM", "Jet_gammaMaxHadronic", "Jet_gammaMaxET","Jet_minDeltaRAllTracks","Jet_minDeltaRPVTracks",};

    // the input branches are named "f0" to "f9", so use a loop in this example
    int nInputs = inputFeatures.size();
    std::vector<float> inputValues(nInputs);
    for (size_t i = 0; i < inputFeatures.size(); i++) {
        std::string branchName = inputFeatures.at(i);
        inputTree->SetBranchAddress(branchName.c_str(), &inputValues[i] );
    }
    float prob;
    std::string probName = "Jet_sigprob";
    inputTree->SetBranchAddress(probName.c_str(), &prob );


    //Read probability    

    //here we need a cast back to float?
    // create the output file, tree and a new branch
    TFile* outputFile = new TFile(outputPath.c_str(), "RECREATE");
    outputFile->cd();
    TTree *outputTree = new TTree(outputTreeName.c_str(), "");
    float outputValue;
    outputTree->Branch("model_output", &outputValue, "model_output/F");


    // setup TensorFlow objects
    tensorflow::setLogging();
    tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(graphPath);
    // TF < 2
    //tensorflow::SessionOptions sessionOptions;
    //tensorflow::setThreading(sessionOptions, nThreads, threadPool);
    //tensorflow::Session* session = tensorflow::createSession(graphDef, sessionOptions);
    // TF >= 2
    tensorflow::Session* session = tensorflow::createSession(graphDef, nThreads);


    // register an input tensor (1 x nInputs) that is filled during the event loop
    tensorflow::Tensor inputTensor(tensorflow::DT_FLOAT, {1, nInputs});
       
    // loop over input events
    for(int i = 0; i < inputTree->GetEntries(); i++) {
        if (i % 100 == 0) {
            std::cout << "evaluating entry " << i << std::endl;
        }
        inputTree->GetEntry(i);

        // fill the input tensor using a data pointer that is shifted consecutively
        float* d = inputTensor.flat<float>().data();
        for (float v : inputValues) {
	    //std::cout<< " input value: " << v <<std::endl;
            *d = v;
            d++;
        }

        // run the inference
        std::vector<tensorflow::Tensor> outputs;
        // TF < 2
        //tensorflow::run(session, {{inputTensorName, inputTensor}}, {outputTensorName}, &outputs);
        // TF >= 2
        tensorflow::run(session, {{inputTensorName, inputTensor}}, {outputTensorName}, &outputs, threadPool);

        // store the result
        outputValue = outputs[0].matrix<float>()(0, 1);
	std::cout << "original prob: " << prob << std::endl;
	std::cout << "output value: " << outputValue << std::endl;
	std::cout << "\n" << std::endl;
        outputTree->Fill();
    }
    
    // finalize files
   
    outputFile->Write();
    outputFile->Close();
    inputFile->Close();

    return 0;
    
}
