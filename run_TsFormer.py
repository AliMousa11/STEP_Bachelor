from step.step_runner import TSFormerRunner
from step.TSFormer_METRLA_embed_cfg import CFG  # Your adjusted config

def main():

    # Initialize runner
    runner = TSFormerRunner(CFG)
    runner.build_model(CFG)

    # Save encoder embeddings
    runner.save_embeddings(CFG.EMBEDDING_SAVE_PATH)

if __name__ == "__main__":
    main()
