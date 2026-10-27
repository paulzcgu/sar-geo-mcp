def test_package_imports():
    """Guard the src-layout packaging: every subpackage must ship, not just the root."""
    import sar_geo_mcp
    import sar_geo_mcp.caltopo
    import sar_geo_mcp.terrain

    assert sar_geo_mcp.caltopo.__name__ == "sar_geo_mcp.caltopo"
    assert sar_geo_mcp.terrain.__name__ == "sar_geo_mcp.terrain"
