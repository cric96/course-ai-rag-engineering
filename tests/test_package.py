import ragcourse


def test_package_exposes_version():
    # Version must match pyproject metadata.
    assert ragcourse.__version__ == "0.1.0"
