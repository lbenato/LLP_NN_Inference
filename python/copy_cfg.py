import FWCore.ParameterSet.Config as cms

# Give the process a name
process = cms.Process("PickEvent")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

# Tell the process which files to use as the source
process.source = cms.Source ("PoolSource",
                             fileNames = cms.untracked.vstring (
                                 #'/store/data/Run2017C/MET/RAW-RECO/HighMET-17Nov2017-v1/50000/BC875B4F-92E5-E711-98A6-0CC47ADAF564.root'
                                 #'/store/data/Run2017C/MET/RAW-RECO/HighMET-17Nov2017-v1/70000/A29B3394-AFE8-E711-B522-E0071B6C9DD0.root'
                                 '/store/data/Run2017C/MET/RAW-RECO/HighMET-17Nov2017-v1/60000/F248F57E-39E7-E711-BFF4-801844DEEF1C.root'
                             ),
                             #lumisToProcess = cms.untracked.VLuminosityBlockRange('300576:974'),
                             #eventsToProcess = cms.untracked.VEventRange('300576:1402347190'),
                             #lumisToProcess = cms.untracked.VLuminosityBlockRange('301417:85'),
                             #eventsToProcess = cms.untracked.VEventRange('301417:54438121'),
                             lumisToProcess = cms.untracked.VLuminosityBlockRange('300515:425'),
                             eventsToProcess = cms.untracked.VEventRange('300515:698201101'),
                         )
# tell the process to only run over 100 events (-1 would mean run over
#  everything
#process.maxEvents = cms.untracked.PSet(
#            input = cms.untracked.int32(1)
#)

# Tell the process what filename to use to save the output
process.Out = cms.OutputModule("PoolOutputModule",
         fileName = cms.untracked.string ("cmsShow_problematic_events_run300515_lumi425_event698201101.root")
)

# make sure everything is hooked up
process.end = cms.EndPath(process.Out)
