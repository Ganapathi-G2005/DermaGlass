import splitfolders
import os

def main():
    input_folder = "dataset"
    output_folder = "processed_dataset"
    
    # Check if input directory exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    print(f"Splitting dataset from '{input_folder}' to '{output_folder}'...")
    print("Ratio: Train=0.8, Val=0.2")
    print("Seed: 1337")

    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    try:
        splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.8, .2), group_prefix=None, move=False)
    except Exception as e:
        print(f"An error occurred during splitting: {e}")
        return

    print(f"Splitting complete. Output saved to {output_folder}")

    # Count and print
    print("-" * 30)
    print("Class Distribution:")
    for phase in ['train', 'val']:
        phase_path = os.path.join(output_folder, phase)
        if os.path.exists(phase_path):
            print(f"\nPhase: {phase.upper()}")
            classes = sorted(os.listdir(phase_path))
            for class_name in classes:
                 class_path = os.path.join(phase_path, class_name)
                 if os.path.isdir(class_path):
                     count = len(os.listdir(class_path))
                     print(f"  {class_name}: {count} images")
        else:
            print(f"Warning: Phase folder {phase} not found.")

if __name__ == "__main__":
    main()
