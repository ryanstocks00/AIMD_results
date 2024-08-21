from pathlib import Path
import json
import time

basis_sets = {
    "cc-pVDZ": "cc-pVDZ-RIFIT",
    "def2-SVP": "def2-SVP-RIFIT",
    "6-31G**": "6-31G**-RIFIT",
    "cc-pVTZ": "cc-pVTZ-RIFIT",
    "def2-TZVP": "def2-TZVP-RIFIT",
    "cc-pVQZ": "cc-pVQZ-RIFIT",
    "def2-QZVP": "def2-QZVP-RIFIT",
}


def generate_input(topology_file, name, basis="cc-pVDZ"):
    j = {}
    with open(topology_file, "r") as tj:
        j["topologies"] = [json.load(tj)]
    j["driver"] = "Dynamics"
    j["title"] = name
    j["system"] = {"ngpus_per_team": 1, "n_teams_per_node": 4}
    model = j["model"] = {}
    model["method"] = "RestrictedRIMP2"
    model["basis"] = basis
    model["aux_basis"] = basis_sets[basis]
    j["system"] = {"max_gpu_memory_mb": 32000, "teams_per_node": 4, "gpus_per_team": 1}
    keywords = j["keywords"] = {}
    keywords["scf"] = {
        "use_ri": True,
        "density_threshold": 1e-10,
        "convergence_threshold": 1e-7,
    }
    keywords["log"] = {"console": {"level": "Performance"}}
    keywords["export"] = {"light_json": True}
    keywords["dynamics"] = {
        "n_timesteps": 100,
        "use_async_timesteps": True,
        "dt": 0.001,
    }
    keywords["frag"] = {
        "level": "Trimer",
        "enable_speed": False,
        "cutoff_type": "Centroid",
        "cutoffs": {"dimer": 20, "trimer": 12},
    }

    return j


time_per_tstep = {32: 150, 64: 80, 128: 40, 256: 25, 512: 15, 1024: 10, 1536: 8}


def generate_slurm(name, j, N=128):
    t = time.strftime(
        "%H:%M:%S",
        time.gmtime(time_per_tstep[N] * j["keywords"]["dynamics"]["n_timesteps"]),
    )
    with open(f"{name}.slurm", "w") as slurm_file:
        slurm_file.write(
            f"""#!/bin/bash
#SBATCH -A m4265
#SBATCH -C gpu&hbm40g
#SBATCH -q regular
#SBATCH -t {t}
#SBATCH -N {N}

export EXESS_PATH="/global/homes/r/ryans/HERMES"
export EXESS_RECORDS_PATH=$EXESS_PATH/records
export EXESS_VALIDATION_PATH=$EXESS_PATH/validation
INPUT_DIR=$(pwd)

OUT_DIR="../../outputs/exess/{name}"
mkdir -p $OUT_DIR
pushd $OUT_DIR
rm *_logs/*
srun -N {N} --export=ALL --ntasks-per-node=9 --gpus-per-node=4 $EXESS_PATH/build/exess $INPUT_DIR/{name}.json 2>&1 | tee {name}.out
popd
                         """
        )


topology_2beg = Path("../2beg/neutral_2beg_topology.json")

for N in [32, 64, 128, 256, 512, 1024, 1536]:
    name = f"n_2beg_{N}_nodes"
    j = generate_input(topology_2beg, name)
    with open(f"{name}.json", "w") as out_file:
        out_file.write(json.dumps(j))
    generate_slurm(name, j, N)

for dt in [0.0001, 0.0005]:
    name = f"n_2beg_{dt}_dt"
    j = generate_input(topology_2beg, name)
    j["keywords"]["dynamics"]["dt"] = dt
    with open(f"{name}.json", "w") as out_file:
        out_file.write(json.dumps(j))
    generate_slurm(name, j)

for sync in [True, False]:
    for speed in [True, False]:
        if sync or speed:
            name = f"n_2beg"
            if sync:
                name += "_sync"
            if speed:
                name += "_speed"
            j = generate_input(topology_2beg, name)
            j["keywords"]["dynamics"]["use_async_timesteps"] = not sync
            j["keywords"]["frag"]["enable_speed"] = speed
            with open(f"{name}.json", "w") as out_file:
                out_file.write(json.dumps(j))
            generate_slurm(name, j)

for cutoff in [10, 20, 30, 40]:
    name = f"n_2beg_dimer_{cutoff}"
    j = generate_input(topology_2beg, name)
    j["keywords"]["frag"]["level"] = "Dimer"
    j["keywords"]["frag"]["cutoffs"]["dimer"] = cutoff
    with open(f"{name}.json", "w") as out_file:
        out_file.write(json.dumps(j))
    generate_slurm(name, j)

for convergence_threshold in [1e-6, 1e-8, 1e-9]:
    name = f"n_2beg_conv_{convergence_threshold}"
    j = generate_input(topology_2beg, name)
    j["keywords"]["scf"]["convergence_threshold"] = convergence_threshold
    with open(f"{name}.json", "w") as out_file:
        out_file.write(json.dumps(j))
    generate_slurm(name, j)
