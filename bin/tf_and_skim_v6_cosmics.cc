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

#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v6_cosmics.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/CaloObjects_v6.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/dbscan.h"
//#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/MyAnalysis.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <chrono>//compute time
#include <ctime>//compute time 

using namespace ROOT::Math;
using namespace std;
using namespace Eigen;

float avg ( std::vector<float> & v )
{
  float return_value = 0.0;
  int n = v.size();
  for ( int i=0; i < n; i++)
    {
      return_value += v.at(i);
    }
  return ( return_value / n);
}

float weighted_avg ( std::vector<float> & v, std::vector<float> & w )
{
  float return_value = 0.0;
  float w_sum = 0.;
  int n = v.size();
  for ( int i=0; i < n; i++)
    {
      return_value += v.at(i)*w.at(i);
      w_sum += w.at(i);
    }
  return ( return_value / w_sum);
}

float stdev ( std::vector<float> & v )
{
  float return_value = 0.0;
  int n = v.size();
  for ( int i=0; i < n; i++)
    {
      return_value += pow(v.at(i) - avg(v),2);
    }
  return sqrt( return_value / n);
}

float biased_weighted_stdev ( std::vector<float> & v , std::vector<float> & w )
//https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
{
  float return_value = 0.0;
  float w_sum = 0.;
  int n = v.size();
  for ( int i=0; i < n; i++)
    {
      return_value += w.at(i)*pow(v.at(i) - weighted_avg(v,w),2);
      w_sum += w.at(i);
    }
  return sqrt( return_value / w_sum);
}


//DBSCAN
#define MINIMUM_POINTS 3     // minimum number of cluster
#define EPSILON (1.4*1.4)//  // distance for clustering, metre^2

//#define MINIMUM_POINTS 4     // minimum number of cluster
//#define EPSILON (1.*1.)//(0.75*0.75)  // distance for clustering, metre^2

void readBenchmarkData(vector<Point>& points)
{
  // load point cloud
  FILE *stream;
  stream = fopen ("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/dbscan_example/benchmark_hepta.dat","ra");

  unsigned int minpts, num_points, cluster, i = 0;
  double epsilon;
  fscanf(stream, "%u\n", &num_points);

  Point *p = (Point *)calloc(num_points, sizeof(Point));

  while (i < num_points)
    {
      fscanf(stream, "%f,%f,%f,%d\n", &(p[i].x), &(p[i].y), &(p[i].z), &cluster);
      p[i].clusterID = UNCLASSIFIED;
      points.push_back(p[i]);
      ++i;
    }

  free(p);
  fclose(stream);
}

void printResults(vector<Point>& points, int num_points)
{
  int i = 0;
  /*
  printf("Number of points: %u\n"
        " x     y     z     cluster_id\n"
        "-----------------------------\n"
  	 , num_points);
  while (i < num_points)
    {
      printf("%5.2lf %5.2lf %5.2lf: %d\n",
  	     points[i].x,
  	     points[i].y, points[i].z,
  	     points[i].clusterID);
      ++i;
    }
  */
  
  printf("Number of points: %u\n"
        " x     y     z     time     eta  phi  station  nRecHits  cluster_id\n"
        "------------------------------------------------------------------------------\n"
	 , num_points);
  while (i < num_points)
    {
      printf("%5.2lf %5.2lf %5.2lf  %5.2lf   %5.2lf \t%5.2lf \t%d \t%d \t%d\n",
	     points[i].x,
	     points[i].y, points[i].z,
	     points[i].time, 
	     points[i].eta, points[i].phi, points[i].station, points[i].nRecHits, 
	     points[i].clusterID);
      ++i;
    }
  
}


//3D line
float get_coord_line(float z, VectorXf Sol) {
  float coord(-999999.);
  if(Sol.size()==2 and Sol[0]!=0)
    {
      coord = (z - Sol[1])/Sol[0];
    }
  return coord;
}




//Assigns x, y, z based on t and p (size 4)
void line(float t, float &x, float &y, float &z, VectorXf SolXZ, VectorXf SolYZ) {
  // a parametric line is define from 6 parameters but 4 are independent
  // x0,y0,z0,z1,y1,z1 which are the coordinates of two points on the line
  // can choose z0 = 0 if line not parallel to x-y plane and z1 = 1;
  x = get_coord_line(t,SolXZ);
  y = get_coord_line(t,SolYZ);
  z = t;
}

//calculate distance between a point and a parametric line
//it looks at two points with coordinates z=0 and z=1
float distance2(float x,float y,float z, VectorXf SolXZ, VectorXf SolYZ) {
    // distance line point is D= | (xp-x0) cross  ux |
    // where ux is direction of line and x0 is a point in the line (like t = 0)
    XYZVector p(x,y,z);
    float x0, y0, z0 = -9999.; 
    float x1, y1, z1 = -9999.;
    line(-1.,x0,y0,z0,SolXZ,SolYZ);
    line(1.,x1,y1,z1,SolXZ,SolYZ);
    //std::cout<< "x0, y0, z0 " << x0 << " " << y0 << " " << z0 << endl;
    //std::cout<< "x1, y1, z1 " << x1 << " " << y1 << " " << z1 << endl;
    XYZVector p0(x0,y0,z0);
    XYZVector p1(x1,y1,z1);
    XYZVector u = (p1-p0).Unit();
    double d2 = ((p-p0).Cross(u)).Mag2();
    return d2;
  }


////////

int main(int argc, char **argv) {

    if(argc<2)
      {
	std::cout<<"Invalid arguments, exit!" << std::endl;
	return 0;
      }

    bool isVerbose(false);
    bool printFit(false);

    std::cout << "Input file: " << argv[1] << std::endl;
    std::cout << "Output file: " << argv[2] << std::endl;
    auto start = std::chrono::system_clock::now();//time!     

    std::string basePath = std::string(std::getenv("CMSSW_BASE")) + "/src/NNInferenceCMSSW/LLP_NN_Inference/nn_inference";
    std::string inputPath = argv[1];
    std::string outputPath = argv[2];//!!!//"/test_on_real_ntuple.root";
    std::string inputTreeName = "ntuple/tree";
    std::string outputTreeName = "tree";//inputTreeName;
    
    /*
    // model and inference settings
    std::string graphPathAK4 = basePath + "/tagger_AK4_v3/graph.pb";
    std::string MetaDataFileAK4 = basePath + "/tagger_AK4_v3/metadata.dat";
    std::string inputTensorNameAK4 = "input_input";
    std::string outputTensorNameAK4 = "FCN/output/Softmax";//"FCN/dense_4/Softmax";//or Softmax?
    //int nInputs = 10;
    std::string graphPathAK8 = basePath + "/tagger_AK8_v2_double_match/graph.pb";
    std::string MetaDataFileAK8 = basePath + "/tagger_AK8_v2_double_match/metadata.dat";
    std::string inputTensorNameAK8 = "input_input";
    std::string outputTensorNameAK8 = "FCN/output/Softmax";//"FCN/dense_4/Softmax";//or Softmax?

    // threading setup
    // to enable tensorflow-native multi-threading, change to "tensorflow" and increase nThreads
    std::string threadPool = "no_threads";
    int nThreads = 1;
    */

    // ================= 
    // Input
    // ================= 

    // open input file, read the tree and register input branches
    TFile* inputFile = new TFile(inputPath.c_str(), "READ");
    TTree* inputTree = (TTree*)inputFile->Get(inputTreeName.c_str());

    float  tree_weight = inputTree->GetWeight();
    if(isVerbose) std::cout << "Tree weight: " << tree_weight << std::endl;

    // Input variables
    Long64_t EventNumber;
    Long64_t RunNumber;
    Long64_t LumiNumber;
    Long64_t nDTSegments;
    Long64_t nCSCSegments;
    Long64_t nCosmicMuons, nCosmicMuonsOneLeg;
    std::vector<CaloJetType>     *CaloJets = 0;
    std::vector<ecalRecHitType>  *EcalRecHitsAK4 = 0;
    CaloMEtType                  *CaloMEt = 0;
    std::vector<DT4DSegmentType> *DTSegments = 0;
    std::vector<CSCSegmentType>  *CSCSegments = 0;
    std::vector<TrackType> *CosmicMuons= 0;
    std::vector<TrackType> *CosmicMuonsOneLeg= 0;

    // Input branches
    TBranch        *b_RunNumber = 0;
    TBranch        *b_LumiNumber = 0;
    TBranch        *b_EventNumber = 0;
    TBranch        *b_CaloJets = 0;
    TBranch        *b_CaloMEt = 0;
    TBranch        *b_EcalRecHitsAK4 = 0;
    TBranch        *b_DTSegments = 0;
    TBranch        *b_CSCSegments = 0;
    TBranch        *b_nDTSegments = 0;
    TBranch        *b_nCSCSegments = 0;
    TBranch        *b_nCosmicMuons = 0;
    TBranch        *b_nCosmicMuonsOneLeg = 0;
    TBranch        *b_CosmicMuons = 0;
    TBranch        *b_CosmicMuonsOneLeg = 0;

    inputTree->SetBranchAddress("RunNumber",       &RunNumber,       &b_RunNumber);
    inputTree->SetBranchAddress("LumiNumber",       &LumiNumber,       &b_LumiNumber);
    inputTree->SetBranchAddress("EventNumber",       &EventNumber,       &b_EventNumber);
    inputTree->SetBranchAddress("CaloJets",              &CaloJets,              &b_CaloJets);
    inputTree->SetBranchAddress("EcalRecHitsAK4",    &EcalRecHitsAK4,    &b_EcalRecHitsAK4);
    inputTree->SetBranchAddress("CaloMEt",               &CaloMEt,               &b_CaloMEt); 
    inputTree->SetBranchAddress("DTSegments",        &DTSegments,        &b_DTSegments); 
    inputTree->SetBranchAddress("CSCSegments",       &CSCSegments,       &b_CSCSegments); 
    inputTree->SetBranchAddress("nDTSegments",       &nDTSegments,       &b_nDTSegments);
    inputTree->SetBranchAddress("nCSCSegments",      &nCSCSegments,      &b_nCSCSegments);
    inputTree->SetBranchAddress("nCosmicMuons",      &nCosmicMuons,      &b_nCosmicMuons);
    inputTree->SetBranchAddress("nCosmicMuonsOneLeg",      &nCosmicMuonsOneLeg,      &b_nCosmicMuonsOneLeg);
    inputTree->SetBranchAddress("CosmicMuons",      &CosmicMuons,      &b_CosmicMuons);
    inputTree->SetBranchAddress("CosmicMuonsOneLeg",      &CosmicMuonsOneLeg,      &b_CosmicMuonsOneLeg);

    // Read jet input features from metadata.dat file
    //AK4
    /*
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
    */

    // ================= 
    // Output
    // ================= 

    TFile* outputFile = new TFile(outputPath.c_str(), "RECREATE");
    outputFile->cd();
    TTree *outputTree = new TTree(outputTreeName.c_str(), "");

    bool isCosmic(false);
    bool isDT_fit(false);
    bool isCosmicVetoWithTags(false);
    long int nCaloJets(0);
    std::vector<CaloJetType>    skimmedCaloJets;
    std::vector<ecalRecHitType> skimmedEcalRecHitsAK4;
    std::vector<ecalRecHitType> skimmedAcceptanceEcalRecHitsAK4;
    //std::vector<ecalRecHitType> taggedEcalRecHitsAK4;
    //std::vector<ecalRecHitType> taggedAcceptanceEcalRecHitsAK4;
    
    //DBSCAN
    std::vector<Point> points;
    std::vector<Point> ClassifiedPoints;
    std::vector<Point> points_valid_time;
    int n_clusters;
    int n_clusters_3_stations;
    int n_noise;
    int n_clusters_valid_time;
    int n_noise_valid_time;

    //Additional output variables
    //DT fit
    float dt_fit_chi2(9999.);
    float dt_fit_chi2_reduced(9999.);
    float dt_ecal_dist(9999.);
    float dt_ecal_no_tag_dist(9999.);
    float dt_ecal_acc_no_tag_dist(9999.);
    float dt_ecal_acc_dist(9999.);
    float m_xz(-9999.);
    float c_xz(-9999.);
    float m_yz(-9999.);
    float c_yz(-9999.);
    std::vector<float> DT_fit_xx;
    std::vector<float> DT_fit_yy;
    std::vector<float> DT_fit_zz;
    std::vector<float> DT_fit_res;

    // Output branches 
    outputTree->Branch("RunNumber",       &RunNumber,       "RunNumber/L");
    outputTree->Branch("LumiNumber",       &LumiNumber,       "LumiNumber/L");
    outputTree->Branch("EventNumber",       &EventNumber,       "EventNumber/L");
    outputTree->Branch("isCosmic",          &isCosmic,          "isCosmic/O");
    outputTree->Branch("isDT_fit",          &isDT_fit,          "isDT_fit/O");
    outputTree->Branch("isCosmicVetoWithTags", &isCosmicVetoWithTags, "isCosmicVetoWithTags/O");
    outputTree->Branch("nDTSegments",       &nDTSegments,       "nDTSegments/I");
    outputTree->Branch("nCSCSegments",      &nCSCSegments,      "nCSCSegments/I");
    outputTree->Branch("nCosmicMuons",      &nCosmicMuons,      "nCosmicMuons/I");
    outputTree->Branch("nCosmicMuonsOneLeg",      &nCosmicMuonsOneLeg,      "nCosmicMuonsOneLeg/I");
    outputTree->Branch("nCaloJets",       &nCaloJets,       "nCaloJets/I");
    outputTree->Branch("n_clusters", &n_clusters, "n_clusters/I");
    outputTree->Branch("n_clusters_3_stations", &n_clusters_3_stations, "n_clusters_3_stations/I");
    outputTree->Branch("n_noise", &n_noise, "n_noise/I");
    outputTree->Branch("n_clusters_valid_time", &n_clusters_valid_time, "n_clusters_valid_time/I");
    outputTree->Branch("n_noise_valid_time", &n_noise_valid_time, "n_noise_valid_time/I");
    outputTree->Branch("dt_fit_chi2", &dt_fit_chi2, "dt_fit_chi2/F");
    outputTree->Branch("dt_fit_chi2_reduced", &dt_fit_chi2_reduced, "dt_fit_chi2_reduced/F");
    outputTree->Branch("dt_ecal_no_tag_dist", &dt_ecal_no_tag_dist, "dt_ecal_no_tag_dist/F");
    outputTree->Branch("dt_ecal_acc_no_tag_dist", &dt_ecal_acc_no_tag_dist, "dt_ecal_acc_no_tag_dist/F");
    outputTree->Branch("dt_ecal_dist", &dt_ecal_dist, "dt_ecal_dist/F");
    outputTree->Branch("dt_ecal_acc_dist", &dt_ecal_acc_dist, "dt_ecal_acc_dist/F");
    outputTree->Branch("m_xz", &m_xz, "m_xz/F");
    outputTree->Branch("c_xz", &c_xz, "c_xz/F");
    outputTree->Branch("m_yz", &m_yz, "m_yz/F");
    outputTree->Branch("c_yz", &c_yz, "c_yz/F");
    outputTree->Branch("CaloJets", &skimmedCaloJets);
    outputTree->Branch("EcalRecHitsAK4", &EcalRecHitsAK4);
    outputTree->Branch("skimmedEcalRecHitsAK4", &skimmedEcalRecHitsAK4);
    outputTree->Branch("DT_fit_xx", &DT_fit_xx);
    outputTree->Branch("DT_fit_yy", &DT_fit_yy);
    outputTree->Branch("DT_fit_zz", &DT_fit_zz);
    outputTree->Branch("DT_fit_res", &DT_fit_res);
    outputTree->Branch("CaloMEt", &CaloMEt);
    outputTree->Branch("DTSegments", &DTSegments);
    outputTree->Branch("CSCSegments", &CSCSegments);
    outputTree->Branch("Points", &ClassifiedPoints);
    outputTree->Branch("CosmicMuons",      &CosmicMuons);
    outputTree->Branch("CosmicMuonsOneLeg",      &CosmicMuonsOneLeg);
    /*
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
    */

    // Event loop

    //for(int ev = 0; ev < 10; ev++) {
    for(int ev = 0; ev < inputTree->GetEntriesFast(); ev++)
      {

	nCaloJets = 0;

	n_clusters = -1;
	n_clusters_3_stations = -1;
	n_noise = -1;
	n_clusters_valid_time = -1;
	n_noise_valid_time = -1;

	dt_fit_chi2 = 9999.;
	dt_fit_chi2_reduced = 9999.;
	dt_ecal_no_tag_dist = 9999.;
	dt_ecal_acc_no_tag_dist = 9999.;
	dt_ecal_dist = 9999.;
	dt_ecal_acc_dist = 9999.;
	isCosmic = false;
	isCosmicVetoWithTags = false;
	isDT_fit = false;

	m_xz = -9999.;
	c_xz = -9999.;
	m_yz = -9999.;
	c_yz = -9999.;

	//Clear all the vectors
        skimmedCaloJets.clear();
	skimmedEcalRecHitsAK4.clear();
	skimmedAcceptanceEcalRecHitsAK4.clear();
	//taggedEcalRecHitsAK4.clear();
	//taggedAcceptanceEcalRecHitsAK4.clear();
	points.clear();
	points_valid_time.clear();
	ClassifiedPoints.clear();
	DT_fit_xx.clear();
	DT_fit_yy.clear();
	DT_fit_zz.clear();
	DT_fit_res.clear();
        inputTree->GetEntry(ev);


	std::vector<int> validCaloJetIndex;

	for (unsigned int j=0; j<CaloJets->size(); j++)
	  {
	    //Here, whatever condition to have valid calo jet indices
	    //Needed to associate ecal rec hits properly
	    //probably eta <1 (currently using 1.4 for larger stat)
	    if(CaloJets->at(j).nRecHitsEB>0)
	      {
		skimmedCaloJets.push_back(CaloJets->at(j));
		validCaloJetIndex.push_back(j);
	      }
	  }//jet loop

	nCaloJets = skimmedCaloJets.size();

	//if(EventNumber==225437 or EventNumber==716167 or EventNumber==1330987)
	//{
	//std::cout <<"For some reasons going to reject "<< EventNumber << std::endl;
	//std::cout <<"nCaloJets? " << nCaloJets << std::endl;
	//}
	
	if(nCaloJets<1) continue;

	//std::cout << "======== " << std::endl;
	//std::cout << "Event " << ev << std::endl;
	//std::cout << "======== " << std::endl;

	//Re-assign Ecal rec hits in case any selection is applied on jets
	for (unsigned int j=0; j<validCaloJetIndex.size(); j++)
	  {
	    //Defined at each jet
	    //float check_ecal_energy(0.);
	    for(unsigned int p=0; p<EcalRecHitsAK4->size(); p++)
	      {
		//j corresponds to the skimmed jet, validCaloJetIndex.at(j) corresponds to the original jets
		//for each valid jet skimmedCaloJets[j] I want the Rec hits features
		if(int(EcalRecHitsAK4->at(p).jetIndex) == int(validCaloJetIndex.at(j)) )
		  {
		    //0.4 matching
		    if (reco::deltaR(CaloJets->at( int(validCaloJetIndex.at(j)) ).eta, CaloJets->at( int(validCaloJetIndex.at(j)) ).phi, EcalRecHitsAK4->at(p).eta, EcalRecHitsAK4->at(p).phi) < 0.4)
		      {

			skimmedEcalRecHitsAK4.push_back(EcalRecHitsAK4->at(p));
			if(abs(CaloJets->at( int(validCaloJetIndex.at(j)) ).eta)<1) skimmedAcceptanceEcalRecHitsAK4.push_back(EcalRecHitsAK4->at(p));
		      }//fill skimmedEcalRecHitsAK4

		  }//check if considered EB associated to jet indices

	      }//loop on EcalRecHitsAK4

	  }//loop on jet indices


	//Calculate center of gravity ECAL rec hits of tagged jets;
	//Used for cosmic veto
	//Tagged jets |eta|<1.4
	std::vector<float> vec_ECAL_x;
	std::vector<float> vec_ECAL_y;
	std::vector<float> vec_ECAL_z;
	float mean_ECAL_x(-9999999.);
	float mean_ECAL_y(-9999999.);
	float mean_ECAL_z(-9999999.);
	std::transform(skimmedEcalRecHitsAK4.begin(), skimmedEcalRecHitsAK4.end(), std::back_inserter(vec_ECAL_x),[](ecalRecHitType const& er) { return er.x/100.; });
	if(skimmedEcalRecHitsAK4.size()>0) mean_ECAL_x=avg(vec_ECAL_x);
	std::transform(skimmedEcalRecHitsAK4.begin(), skimmedEcalRecHitsAK4.end(), std::back_inserter(vec_ECAL_y),[](ecalRecHitType const& er) { return er.y/100.; });
	if(skimmedEcalRecHitsAK4.size()>0) mean_ECAL_y=avg(vec_ECAL_y);
	std::transform(skimmedEcalRecHitsAK4.begin(), skimmedEcalRecHitsAK4.end(), std::back_inserter(vec_ECAL_z),[](ecalRecHitType const& er) { return er.z/100.; });
	if(skimmedEcalRecHitsAK4.size()>0) mean_ECAL_z=avg(vec_ECAL_z);

	//All jets |eta|<1.
	std::vector<float> vec_acc_ECAL_x;
	std::vector<float> vec_acc_ECAL_y;
	std::vector<float> vec_acc_ECAL_z;
	float mean_acc_ECAL_x(-9999999.);
	float mean_acc_ECAL_y(-9999999.);
	float mean_acc_ECAL_z(-9999999.);
	std::transform(skimmedAcceptanceEcalRecHitsAK4.begin(), skimmedAcceptanceEcalRecHitsAK4.end(), std::back_inserter(vec_acc_ECAL_x),[](ecalRecHitType const& er) { return er.x/100.; });
	if(skimmedAcceptanceEcalRecHitsAK4.size()>0) mean_acc_ECAL_x=avg(vec_acc_ECAL_x);
	std::transform(skimmedAcceptanceEcalRecHitsAK4.begin(), skimmedAcceptanceEcalRecHitsAK4.end(), std::back_inserter(vec_acc_ECAL_y),[](ecalRecHitType const& er) { return er.y/100.; });
	if(skimmedAcceptanceEcalRecHitsAK4.size()>0) mean_acc_ECAL_y=avg(vec_acc_ECAL_y);
	std::transform(skimmedAcceptanceEcalRecHitsAK4.begin(), skimmedAcceptanceEcalRecHitsAK4.end(), std::back_inserter(vec_acc_ECAL_z),[](ecalRecHitType const& er) { return er.z/100.; });
	if(skimmedAcceptanceEcalRecHitsAK4.size()>0) mean_acc_ECAL_z=avg(vec_acc_ECAL_z);

	//Cosmic veto:
	//DBSCAN on DTSegments
	for(unsigned int d=0; d<DTSegments->size(); d++)
	  {
	    Point p;
	    //Currently not removing points with invalid time;
	    //TODO: check if the result changes
	    p.x = DTSegments->at(d).x/100.;
	    p.y = DTSegments->at(d).y/100.;
	    p.z = DTSegments->at(d).z/100.;
	    p.eta = DTSegments->at(d).eta;
	    p.phi = DTSegments->at(d).phi;
	    p.time = DTSegments->at(d).time;
	    p.wheel = DTSegments->at(d).wheel;
	    p.sector = DTSegments->at(d).sector;
	    p.station = DTSegments->at(d).station;
	    p.nRecHits = DTSegments->at(d).nRecHits;
	    p.clusterID = UNCLASSIFIED;
	    p.nStationsPerCluster = -1;
	    points.push_back(p);

	    /*
	    if(DTSegments->at(d).time > -9999.)
	    {
	        Point p;
		p.x = DTSegments->at(d).x/100.;
		p.y = DTSegments->at(d).y/100.;
		p.z = DTSegments->at(d).z/100.;
		p.eta = DTSegments->at(d).eta;
		p.phi = DTSegments->at(d).phi;
		p.time = DTSegments->at(d).time;
		p.wheel = DTSegments->at(d).wheel;
		p.sector = DTSegments->at(d).sector;
		p.station = DTSegments->at(d).station;
		p.nRecHits = DTSegments->at(d).nRecHits;
		p.clusterID = UNCLASSIFIED;
		points_valid_time.push_back(p);
	    }
	    */
	  }

	DBSCAN ds(MINIMUM_POINTS, EPSILON, points);
	ds.run();
	
	//Now: we must consider only good labels!
	std::vector<int> labels;
	std::transform(ds.m_points.begin(), ds.m_points.end(), std::back_inserter(labels),[](Point const& p) { return p.clusterID; });	

	if(labels.size()>0)
	  {
	    n_noise = std::count (labels.begin(), labels.end(), -1);
	    int max = *max_element(labels.begin(), labels.end());
	    if(max == -1) n_clusters = 0;
	    else n_clusters = max+1;// - 1*int( bool(n_noise_) );
	  }

	//need to store cluster objects
	//how many stations
	//their position
	std::vector<int> good_labels;
	if(n_clusters>0)
	  {
	    for(unsigned int l=0; l<labels.size();l++)
	      {
		std::vector<int> stations;
		for(unsigned int p=0;p<ds.m_points.size();p++)
		  {
		    if(ds.m_points.at(p).clusterID==labels[l] and labels[l]!=-1)
		      {
			stations.push_back(ds.m_points.at(p).station);
		      }
		  }

		std::sort( stations.begin(), stations.end() );
		stations.erase( unique( stations.begin(), stations.end() ), stations.end() );
		//2 instead of 3!!!!
		if(stations.size()>=3) good_labels.push_back(labels[l]);
		//if(stations.size()>=2) good_labels.push_back(labels[l]);
	      }
	  }
	

	std::sort( good_labels.begin(), good_labels.end() );
	good_labels.erase( unique( good_labels.begin(), good_labels.end() ), good_labels.end() );
	//if(good_labels.size()>0) std::cout << "The good labels are: ";
	//for(unsigned int gl=0;gl<good_labels.size();gl++) std::cout<< " " <<good_labels[gl];
	//if(good_labels.size()>0) std::cout << " " << std::endl;
	n_clusters_3_stations = good_labels.size();
	//std::cout<<"number of good clusters: " << n_clusters_3_stations<<std::endl;

	//Loop over good_labels and store good points without duplicates
	for(unsigned int p=0;p<ds.m_points.size();p++)
	  {
	    for(unsigned int gl=0;gl<good_labels.size();gl++) 
	      {
		if(good_labels.at(gl)==ds.m_points.at(p).clusterID) ClassifiedPoints.push_back(ds.m_points.at(p));
	      }
	  }

	std::vector<int>   tmp_labels;
	std::vector<float> xx;
	std::vector<float> yy;
	std::vector<float> zz;
	std::vector<float> tt;
	std::vector<int>   ss;
	std::transform(ClassifiedPoints.begin(), ClassifiedPoints.end(), std::back_inserter(tmp_labels),[](Point const& p) { return p.clusterID; });
	std::transform(ClassifiedPoints.begin(), ClassifiedPoints.end(), std::back_inserter(xx),[](Point const& p) { return p.x; });
	std::transform(ClassifiedPoints.begin(), ClassifiedPoints.end(), std::back_inserter(yy),[](Point const& p) { return p.y; });
	std::transform(ClassifiedPoints.begin(), ClassifiedPoints.end(), std::back_inserter(zz),[](Point const& p) { return p.z; });
	std::transform(ClassifiedPoints.begin(), ClassifiedPoints.end(), std::back_inserter(tt),[](Point const& p) { return p.time; });
	std::transform(ClassifiedPoints.begin(), ClassifiedPoints.end(), std::back_inserter(ss),[](Point const& p) { return p.station; });

	//Warning, labels have a lot of duplicates!
	//Now: we must consider only good labels!
	std::vector<std::vector<float>> vec_xx(n_clusters_3_stations,std::vector<float>());
	std::vector<std::vector<float>> vec_yy(n_clusters_3_stations,std::vector<float>());
	std::vector<std::vector<float>> vec_zz(n_clusters_3_stations,std::vector<float>());
	std::vector<std::vector<float>> vec_tt(n_clusters_3_stations,std::vector<float>());
	std::vector<std::vector<int>>   vec_ss(n_clusters_3_stations,std::vector<int>());

	//tmp_labels must have the same size of the others
	for(unsigned int l=0;l<tmp_labels.size();l++)
	  {
	    for(unsigned int gl=0;gl<good_labels.size();gl++)
	      {
		if(tmp_labels.at(l)==good_labels.at(gl))
		  {
		    vec_xx.at( gl ).push_back(xx.at( l ));
		    vec_yy.at( gl ).push_back(yy.at( l ));
		    vec_zz.at( gl ).push_back(zz.at( l ));
		    if(tt.at(l)>-9999.)
		      {
			vec_tt.at(gl).push_back(tt.at(l));
		      }
		    else
		      {
			vec_tt.at(gl).push_back(0.);
		      }
		    vec_ss.at(gl).push_back(ss.at(l));
		
		  }
	      }
	  }

	//Fit of the cosmic trajectory if present
	//choose the right pair of cosmic clouds
	//if(n_clusters_2_stations>=2 and nCosmicMuonsOneLeg>0)
	//if(nCosmicMuonsOneLeg>0 and nCosmicMuons>1)
	//if(nCosmicMuonsOneLeg>0)
	if(n_clusters_3_stations>=2 and nCosmicMuonsOneLeg>0 and nCosmicMuons>1)
	  {
	    //I now have n_clusters_3_stations vectors
	    //I can loop over the clusters
	    int ch_k1 = -1;
	    int ch_k2 = -1;
	    //z position of DT and of ECAL
	    float dz_DT = 1000.;
	    float dz_ECAL = 1000.;
	    float dz_acc_ECAL = 1000.;//in case it's needed?

	    for(int k1 = 0; k1<n_clusters_3_stations; k1++)
	      {

		for(int k2 = k1+1; k2<n_clusters_3_stations && k2!=k1; k2++)
		  {
		    float mean_k1_x=avg(vec_xx.at(k1));
		    float mean_k1_y=avg(vec_yy.at(k1));
		    float mean_k1_z=avg(vec_zz.at(k1));
		    float mean_k1_t=avg(vec_tt.at(k1));
		    float std_k1_t=stdev(vec_tt.at(k1));
 		    std::vector<int> stations_k1 = vec_ss.at(k1); 
		    stations_k1.resize(std::distance(stations_k1.begin(), std::unique(stations_k1.begin(), stations_k1.end())  ));
		    int n_k1_s = stations_k1.size();

		    float mean_k2_x=avg(vec_xx.at(k2));
		    float mean_k2_y=avg(vec_yy.at(k2));
		    float mean_k2_z=avg(vec_zz.at(k2));
		    float mean_k2_t=avg(vec_tt.at(k2));
		    float std_k2_t=stdev(vec_tt.at(k2));
		    std::vector<int> stations_k2 = vec_ss.at(k2); 
		    stations_k2.resize(std::distance(stations_k2.begin(), std::unique(stations_k2.begin(), stations_k2.end())  ));
		    int n_k2_s = stations_k2.size();

		    //Opposite emispheres condition plus comsic tracks
		    if(  (mean_k1_x*mean_k2_x<0 or mean_k1_y*mean_k2_y<0 or mean_k1_z*mean_k2_z<0)  )
		      {
			float tmp_z = abs(mean_k1_z - mean_k2_z);
			dz_DT = std::min(dz_DT,tmp_z);//this is probably useless

			float tmp_ECAL = 99999999.;
			if(skimmedEcalRecHitsAK4.size()>0)
			  {
			    //Distance in z bw center of mass of cluster and center of mass of ecal rec hits
			    //Look at all the 1.4 eta
			    tmp_ECAL = abs((mean_k1_z+mean_k2_z)/2. - mean_ECAL_z);
			  }
			else
			  {
			    tmp_ECAL = 99999999.;//very large number so that this is always false
			  }

			//minimum distance in z bw ecal and dt segments
			dz_ECAL = std::min(dz_ECAL,tmp_ECAL);


			//cout << "Pair: " << k1 << " " << k2 << endl;
			//cout << "tmp_ECAL: " << tmp_ECAL << " dz_ECAL " << dz_ECAL << endl;


			//if this pair of cosmics is the closest to ECAL
			if(dz_ECAL==tmp_ECAL)
			  {
                            ch_k1 = k1;
                            ch_k2 = k2;
			  }
		      }//opposite condition
		  }//loop k2

	      }//loop k1
	    //For printing purposes
	    if(ch_k1>-1 and ch_k2>-1)
	      {
		//cout << "Chosen pair: "<< ch_k1 << " " << ch_k2 << endl;
		DT_fit_xx.reserve(vec_xx.at(ch_k1).size() + vec_xx.at(ch_k2).size() );
		DT_fit_xx.insert( DT_fit_xx.end(), vec_xx.at(ch_k1).begin(), vec_xx.at(ch_k1).end());
		DT_fit_xx.insert( DT_fit_xx.end(), vec_xx.at(ch_k2).begin(), vec_xx.at(ch_k2).end());
		DT_fit_yy.reserve(vec_yy.at(ch_k1).size() + vec_yy.at(ch_k2).size() );
		DT_fit_yy.insert( DT_fit_yy.end(), vec_yy.at(ch_k1).begin(), vec_yy.at(ch_k1).end());
		DT_fit_yy.insert( DT_fit_yy.end(), vec_yy.at(ch_k2).begin(), vec_yy.at(ch_k2).end());
		DT_fit_zz.reserve(vec_zz.at(ch_k1).size() + vec_zz.at(ch_k2).size() );
		DT_fit_zz.insert( DT_fit_zz.end(), vec_zz.at(ch_k1).begin(), vec_zz.at(ch_k1).end());
		DT_fit_zz.insert( DT_fit_zz.end(), vec_zz.at(ch_k2).begin(), vec_zz.at(ch_k2).end());


		//TODO these vector insertion are probably not needed and can fit the VectorXf directly
		Map<VectorXf> VX(DT_fit_xx.data(),DT_fit_xx.size());
		Map<VectorXf> VY(DT_fit_yy.data(),DT_fit_yy.size());
		Map<VectorXf> VZ(DT_fit_zz.data(),DT_fit_zz.size());
		VectorXf One(DT_fit_xx.size());
		VectorXf SolXZ(2);
		VectorXf SolYZ(2);
		One.setOnes();
		//here Axz, Ayz
		MatrixXf Axz(DT_fit_xx.size(),2);
		Axz << VX, One;
		MatrixXf Ayz(DT_fit_xx.size(),2);
		Ayz << VY , One;
		SolXZ = (Axz.transpose() * Axz).ldlt().solve(Axz.transpose() * VZ);
		SolYZ = (Ayz.transpose() * Ayz).ldlt().solve(Ayz.transpose() * VZ);
		m_xz = SolXZ[0];
		c_xz = SolXZ[1];
		m_yz = SolYZ[0];
		c_yz = SolYZ[1];
		
		if(DT_fit_xx.size()>0)
		  {
		    //Only if we have valid fit points: reset chi squared
		    dt_fit_chi2 = 0.;
		    for(unsigned int c=0; c<DT_fit_xx.size(); c++)
		      {
			float tmp_dist2 = distance2(DT_fit_xx.at(c),DT_fit_yy.at(c),DT_fit_zz.at(c),SolXZ,SolYZ);
			dt_fit_chi2    += tmp_dist2;
			DT_fit_res.push_back(tmp_dist2);
		      }
		    
		    if(printFit) cout << "dt_fit_chi2: " << dt_fit_chi2 <<endl;
		    isDT_fit = true;
		    dt_fit_chi2_reduced = dt_fit_chi2/DT_fit_xx.size();
		    if(skimmedEcalRecHitsAK4.size()>0) dt_ecal_dist = sqrt(distance2(mean_ECAL_x,mean_ECAL_y,mean_ECAL_z,SolXZ,SolYZ));
		    if(skimmedAcceptanceEcalRecHitsAK4.size()>0) dt_ecal_acc_dist = sqrt(distance2(mean_acc_ECAL_x,mean_acc_ECAL_y,mean_acc_ECAL_z,SolXZ,SolYZ));
		  }

		if(dt_ecal_dist<0.5)
		  {
		    isCosmic = true;
		  }

	      }


	  }//if 2 clusters


	outputTree->Fill();

      }


    // finalize files
    outputTree->SetWeight(tree_weight);
    outputFile->Write();
    outputFile->Close();
    auto end = std::chrono::system_clock::now();//time!
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "**************************************************" << std::endl;
    std::cout << "Output written: " << outputPath << std::endl;
    std::cout << "\n" << std::endl;

    return 0;
}


//DBSCAN functions
int DBSCAN::run()
{
  int clusterID = 0;//Original was 1!
  vector<Point>::iterator iter;
  for(iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
      if ( iter->clusterID == UNCLASSIFIED )
        {
	  if ( expandCluster(*iter, clusterID) != FAILURE )
            {
	      clusterID += 1;
            }
        }
    }

  return 0;
}

int DBSCAN::expandCluster(Point point, int clusterID)
{    
    vector<int> clusterSeeds = calculateCluster(point);

    if ( clusterSeeds.size() < m_minPoints )
    {
        point.clusterID = NOISE;
        return FAILURE;
    }
    else
    {
        int index = 0, indexCorePoint = 0;
        vector<int>::iterator iterSeeds;
        for( iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
        {
            m_points.at(*iterSeeds).clusterID = clusterID;
            if (m_points.at(*iterSeeds).x == point.x && m_points.at(*iterSeeds).y == point.y && m_points.at(*iterSeeds).z == point.z )
            {
                indexCorePoint = index;
            }
            ++index;
        }
        clusterSeeds.erase(clusterSeeds.begin()+indexCorePoint);

        for( vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i )
        {
            vector<int> clusterNeighors = calculateCluster(m_points.at(clusterSeeds[i]));

            if ( clusterNeighors.size() >= m_minPoints )
            {
                vector<int>::iterator iterNeighors;
                for ( iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors )
                {
                    if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED || m_points.at(*iterNeighors).clusterID == NOISE )
                    {
                        if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED )
                        {
                            clusterSeeds.push_back(*iterNeighors);
                            n = clusterSeeds.size();
                        }
                        m_points.at(*iterNeighors).clusterID = clusterID;
                    }
                }
            }
        }

        return SUCCESS;
    }
}

vector<int> DBSCAN::calculateCluster(Point point)
{
    int index = 0;
    vector<Point>::iterator iter;
    vector<int> clusterIndex;
    for( iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( calculateDistance(point, *iter) <= m_epsilon )
        {
            clusterIndex.push_back(index);
        }
        index++;
    }
    return clusterIndex;
}

inline double DBSCAN::calculateDistance(const Point& pointCore, const Point& pointTarget )
{
    return pow(pointCore.x - pointTarget.x,2)+pow(pointCore.y - pointTarget.y,2)+pow(pointCore.z - pointTarget.z,2);
}


