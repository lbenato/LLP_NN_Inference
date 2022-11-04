#! /usr/bin/env python

sample = {}

samples = {
    'SingleMuon2017' : {
        'order' : 0,
        'files' : [
            'SingleMuonRun2017B-17Nov2017-v1',
            'SingleMuonRun2017C-17Nov2017-v1',
            'SingleMuonRun2017D-17Nov2017-v1',
            'SingleMuonRun2017E-17Nov2017-v1',
            'SingleMuonRun2017F-17Nov2017-v1',
        ],
        'fillcolor' : 800,
        'fillstyle' : 1001,
        'linecolor' : 800,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "SingleMuon 2017",
        'weight': 1.,
        'plot': True,
    },
    'SingleMuon2018' : {
        'order' : 0,
        'files' : [
            'SingleMuonRun2018A-17Sep2018-v2',
            'SingleMuonRun2018B-17Sep2018-v1',
            'SingleMuonRun2018C-17Sep2018-v1',
            'SingleMuonRun2018D-PromptReco-v2'
            ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "SingleMuon 2018",
        'weight': 1.,
        'plot': True,
    },
    'BkgSum' : {
        'order' : 0,
        'files' : [],
        'fillcolor' : 1,
        'fillstyle' : 3003,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Bkg stat.",
        'weight': 1.,
        'plot': True,
    },

}
