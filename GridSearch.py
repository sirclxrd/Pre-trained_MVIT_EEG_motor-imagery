import os
import itertools
import yaml
import subprocess

# Base config
base_config_path = "configs/single_config_16_2_2.yaml"
with open(base_config_path, 'r') as f:
    base_config = yaml.safe_load(f)

# Grid sensata
learning_rates = [1e-4, 3e-4, 5e-4]
weight_decays = [0.0, 1e-4, 1e-2]

# Cartella per i config generati
grid_config_dir = "configs/gridsearch_lr_wdecay"
os.makedirs(grid_config_dir, exist_ok=True)

# Ciclo sulle combinazioni
for lr, wd in itertools.product(learning_rates, weight_decays):
    new_config = base_config.copy()
    new_config["train"]["lr"] = lr
    new_config["train"]["wdecay"] = wd  # <-- come richiesto

    # Crea nome file
    config_name = f"config_lr{lr:.0e}_wd{wd:.0e}.yaml"
    config_path = os.path.join(grid_config_dir, config_name)

    # Salva il nuovo file
    with open(config_path, "w") as f:
        yaml.dump(new_config, f)

    print(f"Creato: {config_path}")

    # Lancia il training
    cmd = f"python Train.py --config {config_path}"
    print(f"Lancio: {cmd}")
    subprocess.run(cmd, shell=True)
