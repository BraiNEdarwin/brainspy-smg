if __name__ == "__main__":
    from brainspy.utils.io import load_configs
    from bspysmg.model.data.outputs.train_model import generate_surrogate_model

    configs = load_configs("configs/training/smg_configs_template.yaml")

    generate_surrogate_model(configs)
