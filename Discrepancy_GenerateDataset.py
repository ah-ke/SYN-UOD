"""
Generating a synthetic discrepancy dataset

The results will be stored in: DIR_DATA/discrepancy_dataset/dataset_name/variant_name

We provide the generated discrepancy dataset used to train our discrepancy network, 
so there is no need to re-generate it unless modifications are required.

The generation process is non-deterministic as it chooses random instances and random classes.
"""

# Import necessary modules
from src.a05_differences.experiments import Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT
from src.pipeline.config import add_experiment

def main():
    # Define the experiment configuration
    class Exp0551_NewDiscrepancyVariant(Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT):
        cfg = add_experiment(Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT.cfg,
                             name='0551_NewDiscrepancyVariant',  # save file
                             gen_name='My_Discrepancy_GenerateDataset',
                             gen_img_ext='.png',  # '.webp' better compression
                             swap_fraction=0.75  # instead of default 0.5
                             )

    # Initialize experiment
    exp = Exp0551_NewDiscrepancyVariant()

    print("Initializing default datasets...")
    exp.init_default_datasets()

    print("Available training dataset:", exp.datasets['train'])

    # Construct the pipeline
    print("Constructing the pipeline and generating dataset...")
    exp.discrepancy_dataset_init_pipeline(write_orig_label=True)
    ######################################################################
    # Test a sample frame
    print("Testing a single frame for visualization...")
    dset = exp.datasets['val']
    dset.set_channels_enabled()  # Disable default loading
    fr = dset[38]  # val: frankfurt_000000_016286
    fr.apply(exp.tr_synthetic_and_show)  # Show the transformed frame

    # Save the transformed frame
    fr.apply(exp.tr_synthetic_and_save)

    # Generate the whole dataset
    print("Generating the whole dataset...")
    exp.discrepancy_dataset_generate(dsets=exp.datasets.values())

    print("Dataset generation completed.")
    ##################################################################################
    # Training the model
    print("Starting training procedure...")
    Exp0551_NewDiscrepancyVariant.training_procedure()

    print("Training completed. Checkpoints are saved in $DIR_EXP/0551_NewDiscrepancyVariant")
    print("Use 'tensorboard --logdir $DIR_EXP/0551_NewDiscrepancyVariant' to visualize training progress.")


if __name__ == "__main__":
    main()