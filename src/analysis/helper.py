import os


def ood_yaml():
    modes = ["standard", "opposite", "random", "orthogonal", "overlapping"]

    if not os.path.exists("/data1/lzengaf/cs182/ICL/src/conf/rand_lb"):
        os.mkdir("/data1/lzengaf/cs182/ICL/src/conf/rand_lb")

    for mode in modes:
        file_path = f"/data1/lzengaf/cs182/ICL/src/conf/rand_lb/{mode}.yaml"
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
                       "out_dir: ../models/logistic_regression\n\n"
                       "wandb:\n"
                       f"    name: \"logistic_regression_{mode}\"\n")

    print("YAML files created.")


def rand_lb():
    modes = ["None", "normal", "uniform", "permute"]

    if not os.path.exists("/data1/lzengaf/cs182/ICL/src/conf/ood"):
        os.mkdir("/data1/lzengaf/cs182/ICL/src/conf/ood")

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
                       f"    random_labels: {mode}\n\n"
                       "out_dir: ../models/logistic_regression\n\n"
                       "wandb:\n"
                       f"    name: \"logistic_regression_{mode}\"\n")

    print("YAML files created.")


if __name__ == "__main__":
    ood_yaml()
    rand_lb()