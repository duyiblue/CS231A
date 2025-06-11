#!/usr/bin/env python3

import os
import subprocess
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_object_code_from_filename(filename: str) -> Optional[str]:
    """Extract object code from grasp filename."""
    # Remove the prefix (random_, original_grasps_, refined_grasps_) and .npy extension
    for prefix in ['random_', 'original_grasps_', 'refined_grasps_']:
        if filename.startswith(prefix):
            return filename[len(prefix):-4]  # Remove prefix and .npy
    return None

def find_all_objects(grasp_dir: Path) -> List[str]:
    """Find all unique object codes in the grasp directory."""
    objects = set()
    
    for filename in os.listdir(grasp_dir):
        if filename.endswith('.npy'):
            obj_code = extract_object_code_from_filename(filename)
            if obj_code:
                objects.add(obj_code)
    
    return sorted(list(objects))

def check_grasp_files_exist(grasp_dir: Path, object_code: str) -> Dict[str, bool]:
    """Check which grasp file types exist for a given object."""
    prefixes = ['random_', 'original_grasps_', 'refined_grasps_']
    availability = {}
    
    for prefix in prefixes:
        filename = f"{prefix}{object_code}.npy"
        filepath = grasp_dir / filename
        availability[prefix.rstrip('_')] = filepath.exists()
    
    return availability

def run_evaluation(object_code: str, grasp_prefix: str, script_dir: Path, 
                  grasp_dir: Path, output_dir: Path, gpu: int = 0) -> Optional[float]:
    """Run evaluation for a specific object and grasp type, return y_PGS score."""
    
    cmd = [
        'python', 'eval_grasp_config_dict.py',
        '--object_code_and_scale_str', object_code,
        '--input_grasp_config_dicts_path', str(grasp_dir),
        '--output_evaled_grasp_config_dicts_path', str(output_dir),
        '--gpu', str(gpu),
        '--use_gui', 'False',
        '--grasp_file_prefix', grasp_prefix
    ]
    
    logger.info(f"Running evaluation: {object_code} with {grasp_prefix}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the evaluation command
        result = subprocess.run(
            cmd, 
            cwd=script_dir,
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Evaluation failed for {object_code} with {grasp_prefix}")
            logger.error(f"stderr: {result.stderr}")
            return None
        
        # Instead of parsing stdout, read the saved .npy file which contains y_PGS
        # The evaluation script saves results to: output_dir / {grasp_prefix}{object_code}.npy
        output_file = output_dir / f"{grasp_prefix}{object_code}.npy"
        
        if output_file.exists():
            try:
                # Load the saved results
                evaled_data = np.load(output_file, allow_pickle=True).item()
                y_pgs_array = evaled_data['y_PGS']
                pgs_score = float(np.mean(y_pgs_array))  # Average y_PGS score
                
                logger.info(f"Successfully extracted y_PGS: {pgs_score:.3f} for {object_code} with {grasp_prefix}")
                return pgs_score
                
            except Exception as e:
                logger.error(f"Failed to load/parse saved results file {output_file}: {e}")
                return None
        else:
            logger.error(f"Expected output file not found: {output_file}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Evaluation timed out for {object_code} with {grasp_prefix}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Batch evaluate grasp performance for all objects')
    parser.add_argument('--grasp_dir', type=str, 
                       default='/iris/u/duyi/cs231a/final_evaluation/top_grasps_output',
                       help='Directory containing grasp files')
    parser.add_argument('--script_dir', type=str,
                       default='/iris/u/duyi/cs231a/get_a_grip/get_a_grip/dataset_generation/scripts',
                       help='Directory containing eval_grasp_config_dict.py')
    parser.add_argument('--results_name', type=str,
                       default='grasp_evaluation_results',
                       help='Base name for output files (.npy and .csv will be added)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU to use for evaluation')
    parser.add_argument('--max_objects', type=int, default=None,
                       help='Maximum number of objects to evaluate (for testing)')
    
    args = parser.parse_args()
    
    grasp_dir = Path(args.grasp_dir)
    script_dir = Path(args.script_dir)
    
    # Create timestamped output directory for intermediate results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    intermediate_base = Path("/iris/u/duyi/cs231a/final_evaluation/intermediate_outputs")
    output_dir = intermediate_base / f"eval_results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Intermediate results will be saved to: {output_dir}")
    
    if not grasp_dir.exists():
        logger.error(f"Grasp directory does not exist: {grasp_dir}")
        return
    
    if not script_dir.exists():
        logger.error(f"Script directory does not exist: {script_dir}")
        return
    
    # Find all objects
    all_objects = find_all_objects(grasp_dir)
    if args.max_objects:
        all_objects = all_objects[:args.max_objects]
    
    logger.info(f"Found {len(all_objects)} objects to evaluate")
    
    # Initialize results storage
    grasp_types = ['random', 'original_grasps', 'refined_grasps']
    results = np.full((len(all_objects), 3), np.nan)  # Initialize with NaN
    object_names = []
    
    successful_evaluations = 0
    total_evaluations = 0
    
    # Process each object
    for i, object_code in enumerate(tqdm(all_objects, desc="Evaluating objects")):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing object {i+1}/{len(all_objects)}: {object_code}")
        logger.info(f"{'='*60}")
        
        object_names.append(object_code)
        
        # Check which grasp files are available
        availability = check_grasp_files_exist(grasp_dir, object_code)
        logger.info(f"Available grasp types: {availability}")
        
        # Evaluate each available grasp type
        for j, grasp_type in enumerate(grasp_types):
            if availability.get(grasp_type, False):
                grasp_prefix = f"{grasp_type}_" if grasp_type != 'random' else 'random_'
                
                total_evaluations += 1
                score = run_evaluation(object_code, grasp_prefix, script_dir, grasp_dir, output_dir, args.gpu)
                
                if score is not None:
                    results[i, j] = score
                    successful_evaluations += 1
                    logger.info(f"✅ {grasp_type}: {score:.3f}")
                else:
                    logger.error(f"❌ {grasp_type}: Failed")
            else:
                logger.warning(f"⚠️  {grasp_type}: File not found")
    
    # Save results
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Successful evaluations: {successful_evaluations}/{total_evaluations}")
    
    # Create comprehensive results dictionary
    results_dict = {
        'scores': results,  # Shape: (num_objects, 3) for [random, original_grasps, refined_grasps]
        'object_names': object_names,
        'grasp_types': grasp_types,
        'successful_evaluations': successful_evaluations,
        'total_evaluations': total_evaluations
    }
    
    # Save results with the specified name
    npy_path = Path(f"{args.results_name}.npy")
    csv_path = Path(f"{args.results_name}.csv")
    
    # Save numpy results
    np.save(npy_path, results_dict)
    logger.info(f"Results saved to: {npy_path}")
    
    # Print summary statistics
    logger.info(f"\n{'='*40}")
    logger.info("SUMMARY STATISTICS")
    logger.info(f"{'='*40}")
    
    for j, grasp_type in enumerate(grasp_types):
        valid_scores = results[:, j][~np.isnan(results[:, j])]
        if len(valid_scores) > 0:
            logger.info(f"{grasp_type:>15}: {len(valid_scores):>2} objects, "
                       f"mean={np.mean(valid_scores):.3f}, "
                       f"std={np.std(valid_scores):.3f}, "
                       f"min={np.min(valid_scores):.3f}, "
                       f"max={np.max(valid_scores):.3f}")
        else:
            logger.info(f"{grasp_type:>15}: No successful evaluations")
    
    # Save CSV summary for easy analysis
    with open(csv_path, 'w') as f:
        f.write("object_name,random,original_grasps,refined_grasps\n")
        for i, obj_name in enumerate(object_names):
            scores_str = ','.join([f"{results[i,j]:.3f}" if not np.isnan(results[i,j]) else "NaN" 
                                  for j in range(3)])
            f.write(f"{obj_name},{scores_str}\n")
    
    logger.info(f"CSV summary saved to: {csv_path}")
    
    return results_dict

if __name__ == "__main__":
    main() 