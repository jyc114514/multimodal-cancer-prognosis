def test_core_imports():
    import cancer_prognosis
    from cancer_prognosis import data, model, pipeline, preprocessing, splits, survival

    assert cancer_prognosis.CoxRidgeModel is model.CoxRidgeModel
    assert callable(pipeline.run_cross_validation)
    assert callable(data.read_cohort_csv)
