from os.path import join as j

configfile: "workflow/config.yaml"

DATA_DIR = config["data_dir"]
FIG_DIR = "figs"
PAPER_DIR = config["paper_dir"]
PAPER_SRC, SUPP_SRC = [j(PAPER_DIR, f) for f in ("main.tex", "supp.tex")]
PAPER, SUPP = [j(PAPER_DIR, f) for f in ("main.pdf", "supp.pdf")]

DERIVED_DIR = j("data", "derived")
SIM_R_DIR = j(DERIVED_DIR, "sim_R")
SIM_R_RES = j(SIM_R_DIR, "rvals.csv")

FIG_SIM_WIJ = j(FIG_DIR, "rvals.pdf")

rule all:
    input:
        PAPER, SUPP

rule paper:
    input:
        PAPER_SRC, SUPP_SRC
    params:
        paper_dir = PAPER_DIR
    output:
        PAPER, SUPP
    shell:
        "cd {params.paper_dir}; make"

rule sample_Wij_entry:
    params:
        cin = 30,
        cout = 5,
        num_sample = 100
    output:
        output_file = SIM_R_RES 
    script:
        "workflow/simulate-R-matrix.py"


rule plot_Wij_entry:
    params:
        input_file = SIM_R_RES 
    output:
        output_file = FIG_SIM_WIJ
    script:
        "workflow/plot-rvals.py"

# rule some_data_processing:
    # input:
        # "data/some_data.csv"
    # output:
        # "data/derived/some_derived_data.csv"
    # script:
        # "workflow/scripts/process_some_data.py"
