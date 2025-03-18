import argparse
import gc
import logging
import os

import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC
from timm import utils


_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Distill RNA feature")
parser.add_argument(
    "--root",
    type=str,
    default="./input/raw_rna_feature",
    help="The root of the raw rna feature",
)
parser.add_argument(
    "--cohort", required=True, type=str, help="The cohort of TCGA to distill"
)
parser.add_argument(
    "--rna-file",
    type=str,
    default="tcga_RSEM_isoform_fpkm.parquet",
    help="The RNA file",
)
parser.add_argument(
    "--transcript-id-map",
    type=str,
    default="probeMap_gencode.v23.annotation.transcript.probemap",
    help="The transcript ensembl id map",
)
parser.add_argument(
    "--cosmic-genes", type=str, required=True, help="Genes from COSMIC project"
)
parser.add_argument(
    "--wsi-feature-root",
    type=str,
    default="./input/wsi_feature/phikon/TCGA_FEATURE",
    help="The root of the WSI feature",
)
parser.add_argument(
    "--classes",
    type=str,
    nargs="+",
    required=True,
    help="The classes of the WSI feature",
)
parser.add_argument(
    "--output",
    type=str,
    default="./input/pruned_rna_feature",
    help="The output directory",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")


def main():
    utils.setup_default_logging()
    args = parser.parse_args()

    _logger.info("Loading RNA data...")
    transcripts_df = pd.read_parquet(os.path.join(args.root, args.rna_file))
    _logger.info(f"Loaded RNA data with shape: {transcripts_df.shape}")

    _logger.info("Filtering WSI features...")
    wsi_feature_dict = dict(
        [
            (cls, os.listdir(os.path.join(args.wsi_feature_root, cls)))
            for cls in args.classes
        ]
    )
    slides = []
    slide_to_class = {}
    for key, val in wsi_feature_dict.items():
        wsi_feature_dict[key] = [_[:15] for _ in val]
        slides.extend(wsi_feature_dict[key])
        slide_to_class.update({_[:15]: key for _ in val})
    pruned_transcripts_df = transcripts_df.loc[:, transcripts_df.columns.isin(slides)].T
    _logger.info(
        f"Filtered WSI features. Pruned transcriptomics DataFrame shape: {pruned_transcripts_df.shape}"
    )
    del transcripts_df
    gc.collect()

    _logger.info("Filtering based on COSMIC genes...")
    cosmic_df = pd.read_csv(os.path.join(args.root, args.cohort, args.cosmic_genes))
    selected_genes = cosmic_df["Gene Symbol"].tolist()
    transcripts_id_map_df = pd.read_csv(
        os.path.join(args.root, args.transcript_id_map), sep="\t"
    )
    selected_transcripts_id = transcripts_id_map_df.loc[
        transcripts_id_map_df["gene"].isin(selected_genes)
    ]["id"].tolist()
    selected_transcripts_id = [
        t for t in selected_transcripts_id if t in pruned_transcripts_df.columns
    ]
    _logger.info(
        f"Selected {len(selected_transcripts_id)} transcripts from COSMIC database."
    )

    target = pd.Series(
        [
            slide_to_class[slide]
            for slide in pruned_transcripts_df.index
            if slide in slide_to_class
        ]
    )
    x_train, x_test, y_train, y_test = train_test_split(
        pruned_transcripts_df, target, test_size=0.2, random_state=args.seed
    )

    _logger.info(
        "Performing Recursive Feature Elimination with Cross-Validation (RFECV)..."
    )
    clf = LinearSVC(random_state=args.seed, max_iter=5000, dual="auto", verbose=0)
    cv = StratifiedKFold(5)
    rfecv = RFECV(
        estimator=clf,
        step=0.05,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=1,
        n_jobs=-1,
        verbose=1,
    )
    rfecv.fit(x_train, y_train)

    optimal_features = pruned_transcripts_df.columns[rfecv.support_].tolist()
    _logger.info(f"RFECV selected {rfecv.n_features_} optimal features.")

    final_features = list(set(optimal_features + selected_transcripts_id))
    _logger.info(f"Number of final features after merging: {len(final_features)}")

    pruned_df = pruned_transcripts_df[final_features]
    _logger.info(f"Pruned DataFrame shape: {pruned_df.shape}")
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, f"{args.cohort}_pruned_rna.csv")
    pruned_df.to_csv(output_file)
    _logger.info(f"Pruned RNA features saved to: {output_file}")

    _logger.info("Training and evaluating with pruned features...")
    x_train_pruned, x_test_pruned, y_train_pruned, y_test_pruned = train_test_split(
        pruned_df, target, test_size=0.2, random_state=args.seed
    )

    clf.fit(x_train_pruned, y_train_pruned)
    y_pred_pruned = clf.predict(x_test_pruned)

    accuracy = accuracy_score(y_test_pruned, y_pred_pruned)
    precision = precision_score(y_test_pruned, y_pred_pruned, average="weighted")
    recall = recall_score(y_test_pruned, y_pred_pruned, average="weighted")
    f1 = f1_score(y_test_pruned, y_pred_pruned, average="weighted")

    _logger.info("Model Performance Metrics (with pruned features):")
    _logger.info(f"Accuracy: {accuracy:.4f}")
    _logger.info(f"Precision: {precision:.4f}")
    _logger.info(f"Recall: {recall:.4f}")
    _logger.info(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
