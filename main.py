from cluster import DataClusterer
import pandas as pd

model = DataClusterer(min_k=2, max_k=15)
model.load_dataset("dataset.csv")

model.scan_k()
model.select_optimal_k()
model.show_report()

model.train()
model.export_model("cluster_model.joblib")
model.export_results("dataset_clustered.csv")

model.save_metrics_plot("plot_choose_k.png")
model.save_pca_plot("plot_clusters.png")

# Проверка сохранённой модели на первых 5 строках
restored = DataClusterer()
restored.import_model("cluster_model.joblib")

test_rows = pd.read_csv("dataset.csv").head(5)
pred_labels, pred_conf = restored.assign_clusters(test_rows)

print("\nПроверка предсказаний:")
for idx, (lbl, conf) in enumerate(zip(pred_labels, pred_conf)):
    print(f"  объект {idx}: кластер {lbl} (уверенность {conf * 100:.1f}%)")
    