import os
import subprocess
import sys
import modal

# 1. Define the Modal App
app = modal.App("amazon-ml-challenge-2026")

# 2. Build the cloud container image and attach local code directory
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "xgboost",
        "lightgbm",
        "scikit-learn",
        "rapidfuzz",
        "pandas",
        "numpy",
        "tqdm",
        "joblib",
    )
    .add_local_dir("code", remote_path="/root/AWS_ml_challenge/code")
)

# 3. Persistent Volumes for Dataset and Output Files
data_volume = modal.Volume.from_name("amazon-ml-dataset", create_if_missing=True)
output_volume = modal.Volume.from_name("amazon-ml-output", create_if_missing=True)


# 4. Cloud Worker: 16 high-performance vCPUs & 64 GB RAM
@app.function(
    image=image,
    cpu=16,
    memory=65536,       # 64 GB RAM for full multi-million train set
    timeout=28800,      # 8 hours max (plenty of time to complete full 1.73M dataset)
    volumes={
        "/root/AWS_ml_challenge/student_resource/dataset": data_volume,
        "/root/AWS_ml_challenge/output": output_volume,
    },
)
def run_pipeline_remote(train_cohort: int = 0, full_test: bool = True, load_model: bool = True):
    os.chdir("/root/AWS_ml_challenge")

    cmd = [
        sys.executable,
        "-u",
        "code/business_entity_resolution/src/run_pipeline.py",
        "--output=/root/AWS_ml_challenge/output",
    ]

    saved_model = "/root/AWS_ml_challenge/output/production_model.joblib"
    if load_model and os.path.exists(saved_model):
        print(f"📦 Pre-trained production model detected at: {saved_model}")
        print("   -> Bypassing model training and resuming inference directly!")
        cmd.append(f"--load-model={saved_model}")
    else:
        cmd.append(f"--train-cohort={train_cohort}")

    if full_test:
        cmd.append("--full")

    print(f"🚀 Running on Modal Cloud: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Persist the output files in the volume
    output_volume.commit()
    print("✅ Finished! output/matching_results.tsv and candidate_pairs.tsv committed to Modal volume.")


@app.local_entrypoint()
def main(train_cohort: int = 0, full_test: bool = True, load_model: bool = True):
    cohort_str = f"{train_cohort:,} entities" if train_cohort > 0 else "FULL 2.2M trainset"
    test_str = "FULL 1.73M test set" if full_test else "test sample"
    print(f"☁️ Launching master pipeline on Modal Cloud (16 vCPUs, 64 GB RAM)...")
    print(f"   * Training Cohort: {cohort_str}")
    print(f"   * Test Inference : {test_str}")
    print(f"   * Auto Load Model: {load_model}")
    run_pipeline_remote.remote(train_cohort=train_cohort, full_test=full_test, load_model=load_model)
    print("\n🎉 Cloud run completed!")
    print("To download the submission files to your local output folder, run:")
    print("  modal volume get amazon-ml-output / output/")
