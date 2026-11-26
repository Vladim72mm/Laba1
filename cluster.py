import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class DataClusterer:
    """Подбор оптимального числа кластеров и сегментация данных через GMM.

    Основной критерий — минимум BIC для гауссовой смеси (устойчив к масштабу
    и эллиптической форме кластеров). Дополнительно считаются метрики KMeans
    (silhouette и инерция) для наглядного сравнения.
    """

    COVARIANCES = ("full", "tied", "diag")

    def __init__(self, min_k=2, max_k=15, seed=42):
        self.k_values = list(range(min_k, max_k + 1))
        self.seed = seed
        self.scaler = StandardScaler()

        self.feature_names = None
        self.df = None
        self.raw_matrix = None
        self.scaled_matrix = None
        self._sample_idx = None

        self.scores = {}
        self.optimal_cov = None
        self.optimal_k = None
        self.gmm = None
        self.cluster_ids = None
        self.probs = None
        self._label_map = None

    # --- Загрузка и подготовка данных ---

    def load_dataset(self, filepath):
        raw_df = pd.read_csv(filepath, sep=None, engine="python")
        self.df = raw_df.select_dtypes(include="number").dropna().reset_index(drop=True)
        self.feature_names = self.df.columns.tolist()
        self.raw_matrix = self.df.to_numpy()
        self.scaled_matrix = self.scaler.fit_transform(self.raw_matrix)

        rng = np.random.default_rng(self.seed)
        n_rows = len(self.scaled_matrix)
        self._sample_idx = rng.choice(n_rows, size=min(n_rows, 10_000), replace=False)
        print(f"Данные загружены: {n_rows} объектов, колонки: {self.feature_names}")

    # --- Оценка числа кластеров ---

    def _calc_silhouette(self, matrix, labels):
        return silhouette_score(matrix[self._sample_idx], labels[self._sample_idx])

    @staticmethod
    def _find_elbow(k_list, inertias):
        x = (np.asarray(k_list) - k_list[0]) / (k_list[-1] - k_list[0])
        y = (np.asarray(inertias) - inertias[-1]) / (inertias[0] - inertias[-1])
        dist = np.abs(x + y - 1.0)
        return k_list[int(np.argmax(dist))]

    def scan_k(self):
        """Вычисляет метрики KMeans и BIC для GMM в заданном диапазоне K."""
        res = {
            "sil_raw": [],
            "sil_norm": [],
            "inertia": [],
            "bic": {cov: [] for cov in self.COVARIANCES},
        }

        for k in self.k_values:
            km_raw = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(self.raw_matrix)
            km_norm = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(self.scaled_matrix)

            res["sil_raw"].append(self._calc_silhouette(self.raw_matrix, km_raw.labels_))
            res["sil_norm"].append(self._calc_silhouette(self.scaled_matrix, km_norm.labels_))
            res["inertia"].append(km_norm.inertia_)

            for cov in self.COVARIANCES:
                model = GaussianMixture(
                    n_components=k, covariance_type=cov, n_init=3, random_state=self.seed
                ).fit(self.scaled_matrix)
                res["bic"][cov].append(model.bic(self.scaled_matrix))

            print(
                f"K={k:2d} | sil(raw)={res['sil_raw'][-1]:.3f} | "
                f"sil(norm)={res['sil_norm'][-1]:.3f} | BIC(tied)={res['bic']['tied'][-1]:.0f}"
            )

        self.scores = res
        self.optimal_cov = min(self.COVARIANCES, key=lambda c: min(res["bic"][c]))

    def select_optimal_k(self):
        """Выбирает K с минимальным значением BIC для лучшего типа ковариации."""
        bic_curve = self.scores["bic"][self.optimal_cov]
        self.optimal_k = self.k_values[int(np.argmin(bic_curve))]
        return self.optimal_k

    def get_report(self):
        s, ks = self.scores, self.k_values
        return {
            "BIC гауссовой смеси (основной)": self.optimal_k,
            "Silhouette (исходные данные)": ks[int(np.argmax(s["sil_raw"]))],
            "Silhouette (нормированные)": ks[int(np.argmax(s["sil_norm"]))],
            "Метод локтя (KMeans inertia)": self._find_elbow(ks, s["inertia"]),
        }

    def show_report(self):
        print(f"\nВыбранный тип ковариации (BIC): {self.optimal_cov}")
        for metric_name, k_val in self.get_report().items():
            print(f"  {metric_name:<38s} -> K = {k_val}")
        print(f"\n=== ИТОГОВОЕ K = {self.optimal_k} ===")

    # --- Обучение и предсказание ---

    def train(self, k=None):
        n_clusters = k or self.optimal_k
        self.gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=self.optimal_cov,
            n_init=10,
            random_state=self.seed,
        ).fit(self.scaled_matrix)

        posteriors = self.gmm.predict_proba(self.scaled_matrix)
        raw_labels = posteriors.argmax(axis=1)

        # Сортировка меток: 0 — самый крупный кластер
        counts = np.bincount(raw_labels, minlength=n_clusters)
        order = np.argsort(-counts)
        self._label_map = np.empty(n_clusters, dtype=int)
        self._label_map[order] = np.arange(n_clusters)

        self.cluster_ids = self._label_map[raw_labels]
        self.probs = posteriors.max(axis=1)

        sizes = np.bincount(self.cluster_ids).tolist()
        high_conf = (self.probs > 0.9).mean()
        print(f"\nРазмеры кластеров: {sizes}")
        print(f"Средняя уверенность: {self.probs.mean():.3f} (доля > 0.9: {high_conf:.1%})")

    def assign_clusters(self, new_data):
        """Возвращает номера кластеров и уверенность модели для новых объектов."""
        if isinstance(new_data, pd.DataFrame):
            arr = new_data[self.feature_names].to_numpy()
        else:
            arr = np.asarray(new_data)
        posteriors = self.gmm.predict_proba(self.scaler.transform(arr))
        labels = self._label_map[posteriors.argmax(axis=1)]
        confidences = posteriors.max(axis=1)
        return labels, confidences

    # --- Сохранение и загрузка ---

    def export_model(self, filepath):
        payload = {
            "scaler": self.scaler,
            "gmm": self.gmm,
            "label_map": self._label_map,
            "features": self.feature_names,
            "k": self.optimal_k,
            "cov": self.optimal_cov,
        }
        joblib.dump(payload, filepath)
        print(f"Модель сохранена в {filepath}")

    def import_model(self, filepath):
        payload = joblib.load(filepath)
        self.scaler = payload["scaler"]
        self.gmm = payload["gmm"]
        self._label_map = payload["label_map"]
        self.feature_names = payload["features"]
        self.optimal_k = payload["k"]
        self.optimal_cov = payload["cov"]
        print(f"Модель загружена из {filepath} (K={self.optimal_k})")

    def export_results(self, filepath):
        result_df = self.df.copy()
        result_df["cluster"] = self.cluster_ids
        result_df["confidence"] = np.round(self.probs, 4)
        result_df.to_csv(filepath, index=False)
        print(f"Датасет с метками сохранён в {filepath}")

    # --- Визуализация ---

    def save_metrics_plot(self, filepath):
        s, ks = self.scores, self.k_values
        fig = plt.figure(figsize=(14, 9), facecolor="#f8f9fa")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.35, wspace=0.22)

        ax_sil = fig.add_subplot(gs[0, 0])
        ax_elb = fig.add_subplot(gs[0, 1])
        ax_bic = fig.add_subplot(gs[1, :])

        # 1. Silhouette
        for key, lbl, color, marker in [
            ("sil_raw", "исходные", "#e63946", "s"),
            ("sil_norm", "нормированные", "#0077b6", "D"),
        ]:
            ax_sil.plot(ks, s[key], marker=marker, ms=5, lw=2, color=color, label=lbl)
            best_idx = int(np.argmax(s[key]))
            ax_sil.scatter(
                [ks[best_idx]], [s[key][best_idx]],
                s=180, marker="*", color=color, edgecolors="#111111", linewidths=1, zorder=5
            )
        ax_sil.set_title("Silhouette (KMeans) — больше лучше", fontsize=11, fontweight="bold", pad=10)
        ax_sil.legend(frameon=True, facecolor="white", edgecolor="#dddddd")

        # 2. Локоть (инерция)
        ax_elb.plot(ks, s["inertia"], marker="o", ms=5, lw=2.2, color="#6a4c93")
        ax_elb.fill_between(ks, s["inertia"], min(s["inertia"]), color="#6a4c93", alpha=0.08)
        k_elbow = self._find_elbow(ks, s["inertia"])
        elb_val = s["inertia"][ks.index(k_elbow)]
        ax_elb.axvline(k_elbow, ls=":", lw=1.5, color="#6a4c93", alpha=0.7)
        ax_elb.scatter(
            [k_elbow], [elb_val],
            s=180, marker="*", color="#ffca3a", edgecolors="#111111", linewidths=1, zorder=5
        )
        ax_elb.set_title(f"Инерция KMeans (излом при K = {k_elbow})", fontsize=11, fontweight="bold", pad=10)

        # 3. BIC (на всю ширину снизу)
        bic_styles = [
            ("full", "#1d3557", "o"),
            ("tied", "#2a9d8f", "^"),
            ("diag", "#e76f51", "v"),
        ]
        for cov, color, marker in bic_styles:
            ax_bic.plot(ks, s["bic"][cov], marker=marker, ms=5, lw=2, color=color, label=cov)

        ax_bic.axvline(self.optimal_k, ls="--", lw=1.8, color="#264653", alpha=0.8)
        min_bic = min(min(vals) for vals in s["bic"].values())
        ref_idx = min(2, len(ks) - 1)
        upper_bic = max(s["bic"][c][ref_idx] for c in self.COVARIANCES)
        ax_bic.set_ylim(min_bic - 0.002 * abs(min_bic), upper_bic + 0.001 * abs(min_bic))
        ax_bic.set_title(
            f"BIC для GMM — меньше лучше (минимум при K = {self.optimal_k}, cov = {self.optimal_cov})",
            fontsize=11, fontweight="bold", pad=10
        )
        ax_bic.legend(title="ковариация", frameon=True, facecolor="white", edgecolor="#dddddd")

        for ax in (ax_sil, ax_elb, ax_bic):
            ax.set_facecolor("white")
            ax.set_xlabel("K", fontsize=10)
            ax.set_xticks(ks)
            ax.grid(True, linestyle="--", alpha=0.35, color="#aaaaaa")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(
            f"Подбор числа кластеров (K ∈ [{ks[0]}, {ks[-1]}])  |  Оптимум: K = {self.optimal_k}",
            fontsize=14, fontweight="bold", y=0.97
        )
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def save_pca_plot(self, filepath):
        rng = np.random.default_rng(self.seed)
        n_rows = len(self.scaled_matrix)
        subset = rng.choice(n_rows, size=min(n_rows, 15_000), replace=False)

        pca = PCA(n_components=3, random_state=self.seed)
        proj = pca.fit_transform(self.scaled_matrix)
        var_ratio = pca.explained_variance_ratio_

        palette = [
            "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51",
            "#6a4c93", "#1982c4", "#8ac926", "#ff595e", "#3a86ff",
        ]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), facecolor="#f8f9fa")
        pairs = [((0, 1), "PC1 vs PC2"), ((0, 2), "PC1 vs PC3")]

        for ax, ((i, j), subtitle) in zip(axes, pairs):
            ax.set_facecolor("white")
            for c in range(self.optimal_k):
                color = palette[c % len(palette)]
                mask = subset[self.cluster_ids[subset] == c]
                total_c = int(np.sum(self.cluster_ids == c))

                ax.scatter(
                    proj[mask, i], proj[mask, j],
                    s=6, alpha=0.45, color=color, edgecolors="none",
                    label=f"Кластер {c} (n={total_c})"
                )

                # Центр кластера
                full_mask = self.cluster_ids == c
                cx, cy = proj[full_mask, i].mean(), proj[full_mask, j].mean()
                ax.scatter(
                    [cx], [cy], s=90, marker="X", color=color,
                    edgecolors="white", linewidths=1.5, zorder=6
                )

            ax.set_xlabel(f"PC{i + 1} ({var_ratio[i]:.1%} дисперсии)", fontsize=10)
            ax.set_ylabel(f"PC{j + 1} ({var_ratio[j]:.1%} дисперсии)", fontsize=10)
            ax.set_title(subtitle, fontsize=12, fontweight="bold", pad=8)
            ax.grid(True, linestyle="--", alpha=0.35, color="#aaaaaa")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=min(self.optimal_k, 5),
            markerscale=3.5, fontsize=9, frameon=True, facecolor="white", edgecolor="#dddddd",
            bbox_to_anchor=(0.5, -0.02)
        )

        n_dims = len(self.feature_names)
        fig.suptitle(
            f"PCA-проекция кластеров (K = {self.optimal_k}, {n_dims}D → 2D)",
            fontsize=14, fontweight="bold", y=0.98
        )
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        