import pyhf
pyhf.set_backend("numpy")

spec = {
        'channels': [
            {
                'name': 'mychannel',
                'samples': [
                    {
                        'name': 'signal MC',
                        'data': [2,3,4,5],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                            ],
                        },
                    {
                        'name': 'bkg MC',
                        'data': [30,19,9,4],
                        'modifiers': [
                            {   "name": "theta", 
                                "type": "histosys", 
                                "data": {
                                    "hi_data": [31,21,12,7], 
                                    "lo_data": [29,17,6,1]
                                    }
                                },
                            {   "name": "SF_theta", "type": "normsys", 
                                "data": {"hi": 1.1, "lo": 0.9}
                                }
                            ],
                        },
                    ],
                }
            ],
        "observations": [
            { "name": "mychannel", "data": [34,22,13,11] }
            ],
        "measurements": [
            { "name": "Measurement", "config": {"poi": "mu", "parameters": []} }
            ],
        "version": "1.0.0"
        }
workspace = pyhf.Workspace(spec)
pdf_pyhf = workspace.model(modifier_settings={
    'normsys': {'interpcode': 'code1'},
    'histosys': {'interpcode': 'code0'}}
    )


data_pyhf = workspace.data(pdf_pyhf)

print(pyhf.infer.mle.fit(data_pyhf, pdf_pyhf, return_fitted_val=True))
