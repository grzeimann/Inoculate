def test_import_and_version():
    import inoculate
    assert hasattr(inoculate, "__version__")
    assert isinstance(inoculate.__version__, str)
    assert len(inoculate.__version__) > 0
