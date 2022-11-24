custom_imports = dict(
    imports=[
        'date.models.heads.date',
        'date.models.modules.conv',
        'date.models.modules.identity',
        'date.models.modules.assigner',
        'date.models.predictors.base_predictor',
        'date.models.predictors.defcn_predictor',
        'date.models.predictors.fcos_predictor',
        'date.models.predictors.one2one_predictor',
        'date.models.predictors.retina_predictor',
        'date.datasets.crowdhuman',
    ],
    allow_failed_imports=False)
