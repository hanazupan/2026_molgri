import yaml

class FlowSeqDumper(yaml.SafeDumper):
    pass

def represent_flow_sequence(dumper, seq):
    """
    This is a quick helper function that forces the yaml to write lists in square brackets on the same line, not as a
    super complicated nested list.
    """
    return dumper.represent_sequence(
        'tag:yaml.org,2002:seq',
        seq,
        flow_style=True
    )

FlowSeqDumper.add_representer(list, represent_flow_sequence)


rule copy_config:
    """
    The idea is to save the config file used in this run to some end directory.
    """
    log:
        configfile_record = "<outputs>config_used.yaml"
    run:
        with open(log.configfile_record,"w") as f:
            yaml.dump(config,f, Dumper=FlowSeqDumper, sort_keys=False)