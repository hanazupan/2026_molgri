"""
These rules should be used for all of our Snakemake runs, e.g. general logging.
"""

from workflow.helpers.io import write_object

rule copy_config:
    """
    The idea is to save the config_data file used in this run to the end directory so we can better reconstruct the
    settings later if looking at old calculations.
    """
    log:
        configfile_record = "{some_folder}config_used.yaml"
    run:
        write_object(config, log.configfile_record)