import subprocess

def main():
    print("Welcome to the Adaptive Personalized RS Benchmark Runner")

    systems = {
        '1': 'ARSLA',
        '2': 'AFNPR',
        '3': 'CoDBand',
        '4': 'IEGN',
        '5': 'DVAR'
    }

    datasets = {
        '1': 'amazon',
        '2': 'movielens'
    }

    print("Select the system/model to run:")
    for key, value in systems.items():
        print(f"{key}. {value}")

    system_choice = input("Enter the number corresponding to the system: ").strip()
    system = systems.get(system_choice)

    if not system:
        print("Invalid system selection. Exiting.")
        return

    print("\nSelect the dataset:")
    for key, value in datasets.items():
        print(f"{key}. {value.capitalize()}")

    dataset_choice = input("Enter the number corresponding to the dataset: ").strip()
    dataset = datasets.get(dataset_choice)

    if not dataset:
        print("Invalid dataset selection. Exiting.")
        return

    # Construct the command based on system
    if system == 'ARSLA':
        command = ['python', 'ARSLA/arsla.py', '--dataset', dataset]

    elif system == 'AFNPR':
        command = ['python', 'AFNPR/afnpr.py', '--dataset', dataset]

    elif system == 'CoDBand':
        command = ['python', 'CodBand/DeliciousLastFMAndMovieLens.py', '--alg', 'codband', '--dataset', dataset]

    elif system == 'IEGN':
        command = ['python', 'IEGN/run.py', '--dataset', dataset]

    elif system == 'DVAR':
        data_path = 'DVAR/data_files/ml' if dataset == 'movielens' else 'DVAR/data_files/amazon'
        command = ['python', 'DVAR/DVAR_train.py', '--data_path', data_path]

    print(f"\nRunning command: {' '.join(command)}\n")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the benchmark: {e}")

if __name__ == "__main__":
    main()
