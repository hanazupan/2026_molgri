

CONFIG_PATH = str(workflow.config_settings.configfiles[0])


rule copy_config:
    input:
        CONFIG_PATH
    output:
        "{end_folder}/config_used.yaml"
    shell:
        "cp {input} {output}"