    if(outlier_config["enable"]):
        amp_range = (outlier_config["amp_min"],outlier_config["amp_max"])
        gain_xpol = add_gain_outliers(
            gain_xpol,
            amp_range,
            outlier_config["n_stations"],
            outlier_config["n_channels"],
        )
        gain_ypol = add_gain_outliers(
            gain_ypol,
            amp_range,
            outlier_config["n_stations"],
            outlier_config["n_channels"],
        )