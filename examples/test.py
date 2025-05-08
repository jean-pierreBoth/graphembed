import os
os.environ["RUST_LOG"] = "info"
import graphembed_rs.graphembed_rs as ge
import graphembed_rs.load_utils as ge_utils
### sketching only
ge.embed_sketching("BlogCatalog.txt", decay=0.3, dim=128, nbiter=5, symetric=True, output="embedding_output")
out_vectors=ge_utils.load_embedding_bson("embedding_output.bson")
print("OUT embedding shape :", out_vectors.shape)
print("first OUT vector    :", out_vectors[0])


### validate accuracy
auc_scores = ge.validate_sketching("BlogCatalog.txt",decay=0.3, dim=128, nbiter=3, nbpass=1, skip_frac=0.2,symetric=True, centric=True)
print("Standard AUC per pass:", auc_scores)
