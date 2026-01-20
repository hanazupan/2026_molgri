from workflow.helpers.io import write_object

rule copy_config:
    """
    The idea is to save the config_data file used in this run to some end directory.
    """
    log:
        configfile_record = "<outputs>config_used.yaml"
    run:
        write_object(config, log.configfile_record)