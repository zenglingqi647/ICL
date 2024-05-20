import os


def ood_yaml():
    modes = ["standard", "opposite", "random", "orthogonal", "proj"]

    if not os.path.exists("/data1/lzengaf/cs182/ICL/src/conf/ood"):
        os.mkdir("/data1/lzengaf/cs182/ICL/src/conf/ood")

    if not os.path.exists("/data1/lzengaf/cs182/ICL/src/conf/ood/base.yaml"):
        file_path = "/data1/lzengaf/cs182/ICL/src/conf/ood/base.yaml"
        with open(file_path, 'w') as file:
            file.write("inherit: \n"
                       "    - ../models/standard.yaml\n"
                       "    - ../wandb.yaml\n\n"
                       "model:\n"
                       "    n_dims: 20\n"
                       "    n_positions: 101\n\n"
                       "training:\n"
                       "    data: gaussian\n"
                       "    task_kwargs: {}\n"
                       "    batch_size: 64\n"
                       "    learning_rate: 0.0001\n"
                       "    save_every_steps: 1000\n"
                       "    keep_every_steps: 100000\n"
                       "    train_steps: 500001\n"
                       "    random_labels: \"None\"\n"
                       "    train_test_dist: \"standard\"\n"
                       "    curriculum:\n"
                       "        dims:\n"
                       "            start: 5\n"
                       "            end: 20\n"
                       "            inc: 1\n"
                       "            interval: 2000\n\n")

    print("YAML base files created.")

    for mode in modes:
        file_path = f"/data1/lzengaf/cs182/ICL/src/conf/ood/{mode}.yaml"
        with open(file_path, 'w') as file:
            file.write("inherit: \n"
                       "    - base.yaml\n\n"
                       "training:\n"
                       "    task: logistic_regression\n"
                       "    curriculum:\n"
                       "        points:\n"
                       "            start: 11\n"
                       "            end: 41\n"
                       "            inc: 2\n"
                       "            interval: 2000\n"
                       "    train_steps: 32001\n"
                       f"    train_test_dist: {mode}\n\n"
                       "out_dir: ../models/logistic_regression_ood\n\n"
                       "wandb:\n"
                       f"    name: \"logistic_regression_{mode}\"\n")

    print("YAML files created.")


def ood_rbf_yaml():
    modes = ["standard", "opposite", "random", "orthogonal", "proj"]

    if not os.path.exists("/data1/lzengaf/cs182/ICL/src/conf/ood"):
        os.mkdir("/data1/lzengaf/cs182/ICL/src/conf/ood")

    for mode in modes:
        file_path = f"/data1/lzengaf/cs182/ICL/src/conf/ood/{mode}_rbf.yaml"
        with open(file_path, 'w') as file:
            file.write("inherit: \n"
                       "    - base.yaml\n\n"
                       "training:\n"
                       "    task: rbf_logistic_regression\n"
                       "    curriculum:\n"
                       "        points:\n"
                       "            start: 11\n"
                       "            end: 41\n"
                       "            inc: 2\n"
                       "            interval: 2000\n"
                       "    train_steps: 32001\n"
                       f"    train_test_dist: {mode}\n\n"
                       "out_dir: ../models/rbf_logistic_regression_ood\n\n"
                       "wandb:\n"
                       f"    name: \"rbf_logistic_regression_{mode}\"\n")

    print("YAML files created.")


def randlb_yaml():
    modes = ["None", "normal", "uniform", "permute"]

    if not os.path.exists("/data1/lzengaf/cs182/ICL/src/conf/randlb"):
        os.mkdir("/data1/lzengaf/cs182/ICL/src/conf/randlb")

    if not os.path.exists("/data1/lzengaf/cs182/ICL/src/conf/randlb/base.yaml"):
        file_path = "/data1/lzengaf/cs182/ICL/src/conf/randlb/base.yaml"
        with open(file_path, 'w') as file:
            file.write("inherit: \n"
                       "    - ../models/standard.yaml\n"
                       "    - ../wandb.yaml\n\n"
                       "model:\n"
                       "    n_dims: 20\n"
                       "    n_positions: 101\n\n"
                       "training:\n"
                       "    data: gaussian\n"
                       "    task_kwargs: {}\n"
                       "    batch_size: 64\n"
                       "    learning_rate: 0.0001\n"
                       "    save_every_steps: 1000\n"
                       "    keep_every_steps: 100000\n"
                       "    train_steps: 500001\n"
                       "    random_labels: \"None\"\n"
                       "    train_test_dist: \"standard\"\n"
                       "    curriculum:\n"
                       "        dims:\n"
                       "            start: 5\n"
                       "            end: 20\n"
                       "            inc: 1\n"
                       "            interval: 2000\n\n")

    print("YAML base files created.")

    for mode in modes:
        file_path = f"/data1/lzengaf/cs182/ICL/src/conf/randlb/{mode}.yaml"
        with open(file_path, 'w') as file:
            file.write("inherit: \n"
                       "    - base.yaml\n\n"
                       "training:\n"
                       "    task: logistic_regression\n"
                       "    curriculum:\n"
                       "        points:\n"
                       "            start: 11\n"
                       "            end: 41\n"
                       "            inc: 2\n"
                       "            interval: 2000\n"
                       "    train_steps: 32001\n"
                       f"    random_labels: {mode}\n\n"
                       "out_dir: ../models/logistic_regression_randlb\n\n"
                       "wandb:\n"
                       f"    name: \"logistic_regression_{mode}\"\n")

    print("YAML files created.")


def randlb_rbf_yaml():
    modes = ["None", "normal", "uniform", "permute"]

    if not os.path.exists("/data1/lzengaf/cs182/ICL/src/conf/randlb"):
        os.mkdir("/data1/lzengaf/cs182/ICL/src/conf/randlb")

    for mode in modes:
        file_path = f"/data1/lzengaf/cs182/ICL/src/conf/randlb/{mode}_rbf.yaml"
        with open(file_path, 'w') as file:
            file.write("inherit: \n"
                       "    - base.yaml\n\n"
                       "training:\n"
                       "    task: rbf_logistic_regression\n"
                       "    curriculum:\n"
                       "        points:\n"
                       "            start: 11\n"
                       "            end: 41\n"
                       "            inc: 2\n"
                       "            interval: 2000\n"
                       "    train_steps: 32001\n"
                       f"    random_labels: {mode}\n\n"
                       "out_dir: ../models/rbf_logistic_regression_randlb\n\n"
                       "wandb:\n"
                       f"    name: \"logistic_regression_{mode}\"\n")

    print("YAML files created.")


if __name__ == "__main__":
    # ood_yaml()
    # ood_rbf_yaml()
    randlb_yaml()
    randlb_rbf_yaml()