import os
import numpy as np
import shutil

# Define paths  
grasps_dir = "/iris/u/duyi/cs231a/get_a_grip/data/dataset/small_random/final_evaled_grasp_config_dicts"
refined_grasps_dir = "/iris/u/duyi/cs231a/grasp_refine/refined_grasps_2048_BCE/grasps"
initial_indices_dir = "/iris/u/duyi/cs231a/grasp_refine/refined_grasps_2048_BCE/initial_indices"

print("🔍 Analyzing dataset consistency across directories...")

# Get all objects from each directory
dataset_files = set(f.replace('.npy', '') for f in os.listdir(grasps_dir) if f.endswith('.npy'))
refined_files = set(f.replace('refined_grasps_', '').replace('.npy', '') for f in os.listdir(refined_grasps_dir) if f.startswith('refined_grasps_') and f.endswith('.npy'))
indices_files = set(f.replace('initial_indices_', '').replace('.npy', '') for f in os.listdir(initial_indices_dir) if f.startswith('initial_indices_') and f.endswith('.npy'))

print(f"📊 Dataset objects: {len(dataset_files)}")
print(f"📊 Refined grasps: {len(refined_files)}")  
print(f"📊 Initial indices: {len(indices_files)}")

# Find intersection of all three sets (objects that have all required files)
common_objects = dataset_files.intersection(refined_files).intersection(indices_files)
print(f"✅ Objects with all required files: {len(common_objects)}")

# Check for missing objects
missing_refined = dataset_files - refined_files
missing_indices = dataset_files - indices_files
if missing_refined:
    print(f"⚠️  Objects missing refined grasps: {sorted(list(missing_refined))[:5]}{'...' if len(missing_refined) > 5 else ''}")
if missing_indices:
    print(f"⚠️  Objects missing initial indices: {sorted(list(missing_indices))[:5]}{'...' if len(missing_indices) > 5 else ''}")

if not common_objects:
    print("❌ No objects found with all required files. Exiting.")
    exit(1)

# Sort objects for consistent processing order
objects_to_process = sorted(list(common_objects))
print(f"🎯 Processing {len(objects_to_process)} objects: {objects_to_process[:5]}{'...' if len(objects_to_process) > 5 else ''}")

# Create output directory
output_root = "grasps_from_experiments"
os.makedirs(output_root, exist_ok=True)

output_dir = os.path.join(output_root, "grasps_2048_BCE")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
print(f"📁 Created output directory: {output_dir}")

# Process each object
for i, object_name in enumerate(objects_to_process):
    print(f"🔄 [{i+1}/{len(objects_to_process)}] Processing {object_name}...")

    try:
        # ===== FILE 1: Copy refined grasps =====
        refined_grasp_file = os.path.join(refined_grasps_dir, f"refined_grasps_{object_name}.npy")
        if not os.path.exists(refined_grasp_file):
            print(f"❌ Missing refined grasp file. Skipping.")
            continue
            
        output_refined_file = os.path.join(output_dir, f"refined_grasps_{object_name}.npy")
        shutil.copy2(refined_grasp_file, output_refined_file)
        refined_data = np.load(output_refined_file, allow_pickle=True).item()

        # ===== FILE 2: Extract original grasps using indices =====
        indices_file = os.path.join(initial_indices_dir, f"initial_indices_{object_name}.npy")
        if not os.path.exists(indices_file):
            print(f"❌ Missing initial indices file. Skipping.")
            continue
            
        original_indices = np.load(indices_file)
        original_grasp_file = os.path.join(grasps_dir, f"{object_name}.npy")
        if not os.path.exists(original_grasp_file):
            print(f"❌ Missing original grasp file. Skipping.")
            continue
            
        original_data = np.load(original_grasp_file, allow_pickle=True).item()
        
        # Extract and save original grasps
        original_grasps_data = {
            "trans": original_data['trans'][original_indices],
            "rot": original_data['rot'][original_indices],
            "joint_angles": original_data['joint_angles'][original_indices],
            "grasp_orientations": original_data['grasp_orientations'][original_indices],
            "y_coll": original_data['y_coll'][original_indices],
            "y_pick": original_data['y_pick'][original_indices],  
            "y_PGS": original_data['y_PGS'][original_indices],
        }
        
        output_original_file = os.path.join(output_dir, f"original_grasps_{object_name}.npy")
        np.save(output_original_file, original_grasps_data)

        # ===== FILE 3: Random baseline grasps =====
        threshold = 0.0
        y_pgs_all = original_data['y_PGS']
        valid_random_idxs = np.where(y_pgs_all >= threshold)[0]

        if len(valid_random_idxs) < 10:
            random_indices = valid_random_idxs
        else:
            np.random.seed(42 + hash(object_name) % 1000)
            random_indices = np.random.choice(valid_random_idxs, size=10, replace=False)

        random_grasps_data = {
            "trans": original_data['trans'][random_indices],
            "rot": original_data['rot'][random_indices],
            "joint_angles": original_data['joint_angles'][random_indices],
            "grasp_orientations": original_data['grasp_orientations'][random_indices],
            "y_coll": original_data['y_coll'][random_indices],
            "y_pick": original_data['y_pick'][random_indices],
            "y_PGS": original_data['y_PGS'][random_indices],
        }

        output_random_file = os.path.join(output_dir, f"random_{object_name}.npy")
        np.save(output_random_file, random_grasps_data)

        # ===== SUMMARY =====
        refined_scores = refined_data['score'] if 'score' in refined_data else [0] * len(original_indices)
        original_scores = original_grasps_data['y_PGS']
        random_scores = random_grasps_data['y_PGS']

        print(f"✅ Success! Scores - Refined: {np.mean(refined_scores):.3f}, Original: {np.mean(original_scores):.3f}, Random: {np.mean(random_scores):.3f}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}...")
        continue

# ===== FINAL SUMMARY =====
print(f"\n{'='*80}")
print(f"🎉 PROCESSING COMPLETE!")
print(f"{'='*80}")
print(f"✅ Successfully processed {len(objects_to_process)} objects")
print(f"📁 All files saved in: {output_dir}")
print(f"🔢 Total files created: {len(os.listdir(output_dir))}")

# Count files by type
refined_count = len([f for f in os.listdir(output_dir) if f.startswith('refined_grasps_')])
original_count = len([f for f in os.listdir(output_dir) if f.startswith('original_grasps_')])
random_count = len([f for f in os.listdir(output_dir) if f.startswith('random_')])

print(f"   - Refined grasps: {refined_count}")
print(f"   - Original grasps: {original_count}")  
print(f"   - Random baselines: {random_count}")
print(f"\n💡 You can now use these files with the updated eval_grasp_config_dict.py script!")
print(f"📝 Example usage:")
print(f"   python eval_grasp_config_dict.py --object_code_and_scale_str <object> --grasp_file_prefix refined_grasps_")
print(f"   python eval_grasp_config_dict.py --object_code_and_scale_str <object> --grasp_file_prefix original_grasps_")
print(f"   python eval_grasp_config_dict.py --object_code_and_scale_str <object> --grasp_file_prefix random_") 