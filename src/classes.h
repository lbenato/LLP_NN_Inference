#include <vector>
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v6.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/CaloObjects_v6.h"
//#include "Analyzer/LLP2018/plugins/RecoObjects.h"

namespace { 
  struct dictionary {

    //Structures
    JetType dummy0;
    FatJetType dummy1;
    VertexType dummy2;
    PFCandidateType dummy3;
    TriggerObjectType dummy4;
    CaloJetType dummy5;
    SimplifiedJetType dummy6;
    LeptonType dummy7;
    GenPType dummy8;
    DT4DSegmentType dummy9;
    CSCSegmentType dummy10;
    PhotonType dummy11;
    TauType dummy12;
    MEtType dummy13;
    ecalRecHitType dummy14;
    hcalRecHitType dummy15;
    //RecoLeptonType dummy16;
    TrackType dummy17;
    LorentzType dummy18;

    JetCaloType calo0;
    std::vector<JetCaloType> caloVector0;

    //Vector of structures
    std::vector<JetType> dummyVector0;
    std::vector<FatJetType> dummyVector1;
    std::vector<VertexType> dummyVector2;
    std::vector<PFCandidateType> dummyVector3;
    std::vector<TriggerObjectType> dummyVector4;
    std::vector<CaloJetType> dummyVector5;
    std::vector<SimplifiedJetType> dummyVector6;
    std::vector<LeptonType> dummyVector7;
    std::vector<GenPType> dummyVector8;
    std::vector<DT4DSegmentType> dummyVector9;
    std::vector<CSCSegmentType> dummyVector10;
    std::vector<PhotonType> dummyVector11;
    std::vector<TauType> dummyVector12;
    std::vector<MEtType> dummyVector13;
    std::vector<ecalRecHitType> dummyVector14;
    std::vector<hcalRecHitType> dummyVector15;
    //std::vector<RecoLeptonType> dummyVector16;
    std::vector<TrackType> dummyVector17;
    std::vector<LorentzType> dummyVector18;

  };
}
