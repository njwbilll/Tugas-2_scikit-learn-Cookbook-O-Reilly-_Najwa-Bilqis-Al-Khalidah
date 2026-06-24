# scikit-learn Cookbook (Third Edition) - Code Reproduction

**Tugas 2 Enrichment Machine Learning | Individual Task**

Reproduksi kode dan penjelasan teori dari buku **scikit-learn Cookbook, Third Edition** oleh John Sukup (Packt Publishing, 2025). Setiap chapter direproduksi sebagai Jupyter Notebook terpisah dengan penjelasan teori dalam Bahasa Indonesia.

---

## Identifikasi
 
| Field | Detail |
|-------|--------|
| Nama | Najwa Bilqis Al Khalidah |
| NIM | 101032300186 |
| Kelas | TK-46-GAB |
| Mata Kuliah | Machine Learning |
| Referensi | Introduction to Machine Learning with Python by Andreas C. Muller & Sarah Guido (O'Reilly) |
 
---
 

## Daftar Isi

| # | Notebook | Topik Utama |
|---|----------|-------------|
| 01 | [Common Conventions and API Elements](#chapter-01) | API scikit-learn, Estimators, Pipelines |
| 02 | [Pre-Model Workflow and Data Preprocessing](#chapter-02) | Imputasi, Scaling, Encoding |
| 03 | [Dimensionality Reduction Techniques](#chapter-03) | PCA, LDA, t-SNE |
| 04 | [Building Models with Distance Metrics and KNN](#chapter-04) | Distance Metrics, KNN |
| 05 | [Linear Models and Regularization](#chapter-05) | Ridge, Lasso, ElasticNet, Polynomial |
| 06 | [Advanced Logistic Regression and Extensions](#chapter-06) | Logistic Regression, Multiclass, Multilabel |
| 07 | [Support Vector Machines and Kernel Methods](#chapter-07) | SVM, Kernel Trick, Hyperparameter Tuning |
| 08 | [Tree-Based Algorithms and Ensemble Methods](#chapter-08) | Decision Tree, Random Forest, GBM, Stacking |
| 09 | [Text Processing and Multiclass Classification](#chapter-09) | NLP, TF-IDF, Text Classification |
| 10 | [Clustering Techniques](#chapter-10) | K-Means, DBSCAN, GMM, Evaluasi Cluster |
| 11 | [Novelty and Outlier Detection](#chapter-11) | Isolation Forest, One-Class SVM, LOF |
| 12 | [Cross-Validation and Model Evaluation Techniques](#chapter-12) | Cross-Validation, Learning Curves, GridSearchCV |
| 13 | [Deploying scikit-learn Models in Production](#chapter-13) | Serialisasi, Monitoring, Model Lifecycle |

---

## Penjelasan Setiap Chapter

---

### Chapter 01
### Common Conventions and API Elements

Notebook ini memperkenalkan konvensi standar dan elemen inti API scikit-learn. scikit-learn dirancang di atas empat prinsip utama: **consistency**, **simplicity**, **modularity**, dan **reusability**. Chapter ini adalah fondasi bagi seluruh pembahasan dalam buku.

**Topik yang Dibahas:**
- Filosofi desain scikit-learn: konsistensi, modularitas, reusability
- **Estimators**: `fit()`, `predict()`, `fit_predict()` dengan contoh `LinearRegression` dan `KMeans`
- **Transformers**: `fit()`, `transform()`, `fit_transform()` dan kapan menggunakannya
- Membuat **custom estimator** menggunakan `BaseEstimator` dan `TransformerMixin`
- **Pipelines**: menghubungkan langkah preprocessing dan model dalam satu objek
- Atribut umum model: `coef_`, `intercept_`, `score()`
- **Hyperparameter tuning**: `set_params()`, `get_params()`, `GridSearchCV`
- Metadata routing dan estimator tags
- Best practices penggunaan API scikit-learn

**Kelas dan Fungsi Utama:** `LinearRegression`, `KMeans`, `StandardScaler`, `Pipeline`, `GridSearchCV`, `BaseEstimator`, `TransformerMixin`

---

### Chapter 02
### Pre-Model Workflow and Data Preprocessing

Notebook ini membahas teknik preprocessing data yang esensial. Prinsip utama yang ditekankan adalah **"garbage in, garbage out"**: kualitas data adalah fondasi dari semua model ML yang baik. Selalu pisahkan data sebelum transformasi untuk mencegah data leakage.

**Topik yang Dibahas:**
- Dampak raw data terhadap performa model: missing data, outlier, data leakage
- **Handling missing data**:
  - `SimpleImputer` (mean/median/modus, untuk missing data < 5%)
  - `KNNImputer` (berbasis tetangga terdekat, untuk missing data 5-10%)
  - `IterativeImputer` (imputasi iteratif berbasis regresi, untuk kasus kompleks)
- **Teknik scaling**:
  - `StandardScaler` (Z-score standardization, cocok untuk data Gaussian)
  - `MinMaxScaler` (normalisasi ke range [0,1])
  - `Normalizer` (L1/L2 normalisasi per baris)
- **Encoding variabel kategorikal**:
  - `OneHotEncoder` (untuk variabel nominal)
  - `LabelEncoder` (untuk variabel ordinal atau label target)
  - `ColumnTransformer` (preprocessing berbeda untuk kolom berbeda sekaligus)
- **Pipeline** scikit-learn dan pentingnya split data sebelum transformasi
- **Feature engineering**: `PolynomialFeatures`, `KBinsDiscretizer`, `RFE`, `SelectFromModel`

**Kelas dan Fungsi Utama:** `SimpleImputer`, `KNNImputer`, `IterativeImputer`, `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, `PolynomialFeatures`, `RFE`

---

### Chapter 03
### Dimensionality Reduction Techniques

Notebook ini membahas teknik reduksi dimensi untuk menyederhanakan dataset berdimensi tinggi sambil mempertahankan informasi yang relevan. Reduksi dimensi membantu mengurangi kompleksitas komputasi, meningkatkan performa model, dan memudahkan visualisasi data.

**Topik yang Dibahas:**
- Pengantar reduksi dimensi: Feature Selection vs Feature Extraction
- **PCA (Principal Component Analysis)**:
  - Cara kerja: eigenvalue, eigenvector, explained variance ratio
  - Visualisasi principal components dan contribution vectors
  - Menentukan jumlah komponen optimal via cumulative explained variance
- **LDA (Linear Discriminant Analysis)**:
  - Perbedaan PCA vs LDA: unsupervised vs supervised
  - Memaksimalkan separabilitas kelas
  - Perbandingan side-by-side PCA vs LDA pada Wine dataset
- **t-SNE**:
  - Visualisasi data berdimensi tinggi (digits dataset)
  - Mengapa t-SNE hanya untuk visualisasi, bukan feature extraction
- Panduan memilih teknik reduksi dimensi yang tepat
- Dampak reduksi dimensi terhadap performa model KNN

**Kelas dan Fungsi Utama:** `PCA`, `LinearDiscriminantAnalysis`, `TSNE`

---

### Chapter 04
### Building Models with Distance Metrics and KNN

Notebook ini membahas konsep jarak (distance metrics) dan algoritma K-Nearest Neighbors (KNN). KNN mengklasifikasikan titik data baru berdasarkan mayoritas label dari K tetangga terdekatnya dalam ruang fitur, seperti prinsip "birds of a feather flock together".

**Topik yang Dibahas:**
- **Distance metrics**: Euclidean (jarak garis lurus), Manhattan (jarak jalur grid), Minkowski (generalisasi)
- Visualisasi intuitif perbedaan Euclidean vs Manhattan distance
- Cara kerja KNN: **lazy learning**, non-parametric, sensitif terhadap skala
- Visualisasi decision boundary KNN dengan berbagai nilai K
- Perbandingan distance metrics pada dataset berbeda (circles vs checkerboard)
- **Hyperparameter tuning KNN**:
  - Elbow method untuk nilai K optimal
  - `GridSearchCV` untuk tuning `n_neighbors`, `weights`, `metric`
- **Evaluasi KNN**:
  - Confusion matrix, classification report
  - ROC curve dan AUC
- **KNN untuk regresi** dengan `KNeighborsRegressor`
- Curse of Dimensionality dan dampaknya pada KNN

**Kelas dan Fungsi Utama:** `KNeighborsClassifier`, `KNeighborsRegressor`, `GridSearchCV`, `confusion_matrix`, `roc_curve`

---

### Chapter 05
### Linear Models and Regularization

Notebook ini membahas model linear dan teknik regularisasi. Regularisasi mencegah overfitting dengan menambahkan penalty ke fungsi loss, mendorong model untuk lebih sederhana dan lebih mampu menggeneralisasi pada data baru.

**Topik yang Dibahas:**
- **Linear Regression** dan Ordinary Least Squares (OLS)
- **Regularisasi**:
  - `Ridge` (L2): mengecilkan koefisien, mempertahankan semua fitur
  - `Lasso` (L1): dapat membuat koefisien = 0, feature selection otomatis
  - `ElasticNet` (L1 + L2): hybrid, cocok untuk multikolinearitas tinggi
- Coefficient path plot untuk visualisasi dampak alpha
- **Bias-Variance Trade-off**:
  - Pengaruh nilai alpha terhadap training vs test performance
  - `RidgeCV` untuk pemilihan alpha optimal via cross-validation
- **Polynomial Regression**: menangkap hubungan non-linear dengan `PolynomialFeatures`
- **Spline Interpolation** dengan `SplineTransformer`: alternatif yang lebih stabil
- Latihan praktis: pipeline komprehensif + cross-validation pada California Housing dataset

**Kelas dan Fungsi Utama:** `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `RidgeCV`, `PolynomialFeatures`, `SplineTransformer`

---

### Chapter 06
### Advanced Logistic Regression and Extensions

Notebook ini membahas logistic regression secara mendalam. Meskipun namanya mengandung kata "regression", logistic regression adalah algoritma **klasifikasi** yang menggunakan sigmoid function untuk memprediksi probabilitas keanggotaan kelas.

**Topik yang Dibahas:**
- **Logistic Regression binary**: sigmoid function, log-odds, decision boundary, threshold
- **Multiclass classification**:
  - One-vs-Rest (OvR): satu classifier binary per kelas
  - Multinomial (SoftMax): memodelkan semua kelas sekaligus
- **Regularisasi dalam Logistic Regression**:
  - Parameter `C` (kebalikan kekuatan regularisasi)
  - L1 vs L2: feature selection vs shrinkage
  - Dampak nilai C: bias-variance trade-off
- **Multilabel Classification**:
  - `MultiOutputClassifier`: satu classifier independen per label
  - `ClassifierChain`: prediksi label sebelumnya sebagai fitur
  - Evaluasi: Hamming Loss, Jaccard Score
- **Metrik evaluasi klasifikasi**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC curve, AUC
  - Precision-Recall curve (untuk imbalanced dataset)
- Decision boundary visualization dan `GridSearchCV`

**Kelas dan Fungsi Utama:** `LogisticRegression`, `OneVsRestClassifier`, `MultiOutputClassifier`, `ClassifierChain`, `roc_auc_score`, `confusion_matrix`

---

### Chapter 07
### Support Vector Machines and Kernel Methods

Notebook ini membahas Support Vector Machines (SVM), salah satu algoritma ML paling kuat yang menggunakan geometri untuk memisahkan kelas. Inti dari SVM adalah menemukan **hyperplane** yang memaksimalkan **margin** antara dua kelas.

**Topik yang Dibahas:**
- **Konsep inti SVM**: hyperplane, support vectors, margin (hard vs soft)
- Visualisasi intuitif: margin, support vectors, dan decision boundary pada dataset 2D
- **Kernel functions dan kernel trick**:
  - Linear kernel: cocok untuk data high-dimensional atau linearly separable
  - Polynomial kernel: menangkap hubungan polinomial antar fitur
  - RBF (Gaussian) kernel: kernel paling fleksibel untuk hubungan non-linear
  - Visualisasi decision boundary ketiga kernel side-by-side
- **Hyperparameter tuning SVM**:
  - Parameter `C`: trade-off antara margin dan misklasifikasi
  - Parameter `gamma`: pengaruh lokal setiap training sample
  - `GridSearchCV` dengan heatmap C vs gamma
- **SVM pada data berdimensi tinggi**: pipeline SVM + PCA
- **`SVR`** untuk tugas regresi
- Evaluasi: confusion matrix, ROC curve, classification report

**Kelas dan Fungsi Utama:** `SVC`, `SVR`, `LinearSVC`, `GridSearchCV`, `Pipeline`

---

### Chapter 08
### Tree-Based Algorithms and Ensemble Methods

Notebook ini membahas algoritma berbasis pohon dan metode ensemble. Metode ensemble menggabungkan banyak model lemah menjadi satu model yang lebih kuat melalui tiga strategi: **Bagging** (mengurangi variance), **Boosting** (mengurangi bias), dan **Stacking**.

**Topik yang Dibahas:**
- **Decision Tree**:
  - Cara kerja: Gini impurity, recursive splitting
  - Visualisasi tree dengan `plot_tree()`
  - Dampak `max_depth` pada bias-variance trade-off
- **Random Forest (Bagging)**:
  - Bootstrap aggregating: pohon paralel pada bootstrap samples
  - Random feature selection untuk menambah diversitas
  - Feature importance: mengidentifikasi fitur paling berpengaruh
  - Pengaruh `n_estimators` terhadap performa
- **Gradient Boosting Machine (GBM)**:
  - Boosting sekuensial: setiap pohon memperbaiki residuals pohon sebelumnya
  - Learning rate, validation curves, dan early stopping
- **Hyperparameter tuning**: `GridSearchCV` dan `RandomizedSearchCV` untuk tree dan ensemble
- **Stacking** dengan `StackingClassifier`: meta-model menggabungkan prediksi base models
- **AdaBoost** sebagai alternatif GBM
- Latihan praktis: perbandingan komprehensif semua model pada Breast Cancer dataset

**Kelas dan Fungsi Utama:** `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier`, `StackingClassifier`, `RandomizedSearchCV`

---

### Chapter 09
### Text Processing and Multiclass Classification

Notebook ini membahas Natural Language Processing (NLP) menggunakan scikit-learn. Diperkirakan 80-90% data di dunia tersimpan dalam bentuk tidak terstruktur, dengan teks sebagai yang paling dominan. Komputer tidak secara native memahami teks, sehingga diperlukan teknik transformasi khusus.

**Topik yang Dibahas:**
- **Pengantar text processing**: preprocessing pipeline (lowercase, tokenisasi, stopword removal)
- **Teknik vektorisasi teks**:
  - `CountVectorizer` (Bag-of-Words): hitungan kemunculan kata per dokumen
  - `TfidfVectorizer` (TF-IDF): bobot kata berdasarkan frekuensi dan keunikan (diskriminatif)
  - Perbandingan BoW vs TF-IDF side-by-side
- **Feature extraction dari teks**:
  - N-gram dengan `ngram_range=(1,2)`: menangkap konteks urutan kata
  - POS Tagging dengan NLTK: kategorisasi gramatikal setiap kata
- **Implementasi model klasifikasi teks**:
  - `MultinomialNB` (Naive Bayes): efisien untuk data sparse
  - `SVC` dengan linear kernel: efektif untuk teks berdimensi tinggi
  - `LogisticRegression`: memberikan interpretabilitas melalui koefisien
- **Strategi multiclass untuk teks**: OvR vs OvO vs Native Multinomial
- **Evaluasi dan interpretasi model**: analisis kata-kata paling berpengaruh (koefisien LR)

**Library dan Kelas Utama:** `CountVectorizer`, `TfidfVectorizer`, `MultinomialNB`, `OneVsRestClassifier`, `OneVsOneClassifier`, `nltk`

---

### Chapter 10
### Clustering Techniques

Notebook ini membahas teknik **unsupervised learning** melalui clustering. Berbeda dari semua chapter sebelumnya yang menggunakan supervised learning, clustering bekerja pada data tanpa label. Tujuannya adalah menemukan struktur tersembunyi dalam data, seperti segmentasi pelanggan dalam bisnis.

**Topik yang Dibahas:**
- **Pengantar clustering**: tiga kategori (centroid-based, connectivity-based, density-based)
- **K-Means Clustering**:
  - Cara kerja: inisialisasi centroid, assignment step, update step, konvergensi
  - Elbow Method untuk menentukan K optimal
  - K-Means++ vs random initialization
- **Hierarchical Clustering (Agglomerative)**:
  - Bottom-up: merging cluster secara rekursif
  - Dendrogram: visualisasi hierarki penggabungan
  - Perbandingan metode linkage: Ward, Complete, Average, Single
- **DBSCAN (Density-Based)**:
  - Core points, border points, noise points (label -1)
  - K-distance plot untuk memilih eps optimal
  - Keunggulan: tidak perlu K, menangani outlier, cluster non-spherical
- **Cluster evaluation metrics**:
  - Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index (internal)
  - Adjusted Rand Index (eksternal, memerlukan ground truth)
- **Panduan memilih algoritma clustering** berdasarkan karakteristik data
- **Teknik lanjutan**: Spectral Clustering dan Gaussian Mixture Models (GMM)

**Kelas dan Fungsi Utama:** `KMeans`, `AgglomerativeClustering`, `DBSCAN`, `SpectralClustering`, `GaussianMixture`, `silhouette_score`, `davies_bouldin_score`

---

### Chapter 11
### Novelty and Outlier Detection

Notebook ini membahas teknik deteksi anomali dalam machine learning. Dua konsep utama yang dibedakan adalah **outlier detection** (mendeteksi data menyimpang yang sudah ada dalam dataset training) dan **novelty detection** (mendeteksi data baru yang tidak menyerupai data training normal).

**Topik yang Dibahas:**
- **Pengantar deteksi anomali**: perbedaan outlier vs novelty, konsep contamination
- **Isolation Forest**:
  - Cara kerja: random partitioning, path length sebagai anomaly score
  - Efisien untuk dataset besar dan berdimensi tinggi
  - Parameter `contamination`, `n_estimators`, `decision_function()`
- **One-Class SVM**:
  - Dilatih hanya pada data normal, lalu membentuk boundary of normality
  - Kernel RBF untuk boundary non-linear
  - Parameter `nu` dan `gamma`
- **Local Outlier Factor (LOF)**:
  - Berbasis kepadatan lokal: membandingkan densitas suatu titik dengan tetangganya
  - Lebih efektif untuk dataset dengan densitas bervariasi
  - Tidak dapat memprediksi pada data baru (strictly unsupervised)
- **Evaluasi model deteksi outlier**: precision, recall, F1 untuk anomali
- **Strategi menangani outlier**: hapus, transformasi, atau pertahankan dengan penanda
- **Panduan memilih teknik**: tabel perbandingan Isolation Forest vs LOF vs One-Class SVM

**Kelas dan Fungsi Utama:** `IsolationForest`, `OneClassSVM`, `LocalOutlierFactor`

---

### Chapter 12
### Cross-Validation and Model Evaluation Techniques

Notebook ini membahas teknik evaluasi model yang robust untuk memastikan model ML dapat menggeneralisasi dengan baik pada data yang belum dilihat. Seperti kata statistikawan George Box: *"All models are wrong, but some are useful."* Tujuannya adalah menemukan model yang paling "useful" untuk kebutuhan kita.

**Topik yang Dibahas:**
- **Pengantar cross-validation**: mengapa train-test split tunggal tidak cukup
- **Metode K-Fold Cross-Validation**: `KFold`, `StratifiedKFold`, `RepeatedKFold`
- **Metode CV lanjutan**:
  - LOOCV (Leave-One-Out): k = n_sampel, hampir tidak ada bias tapi mahal
  - Shuffle Split: fleksibel untuk dataset besar
  - Time Series Split: untuk data time-series yang tidak boleh di-shuffle
- **Implementasi CV di scikit-learn**: `cross_val_score()`, `cross_validate()`, `cross_val_predict()`
- **Pemilihan model**: `GridSearchCV` (exhaustive) vs `RandomizedSearchCV` (sampling)
- **Diagnostik overfitting/underfitting**:
  - Learning curves: training vs validation score vs jumlah sampel
  - Validation curves: training vs validation score vs nilai hyperparameter
- Latihan komprehensif: pipeline evaluasi lengkap dari CV hingga deployment-ready model

**Kelas dan Fungsi Utama:** `KFold`, `StratifiedKFold`, `cross_val_score`, `cross_validate`, `GridSearchCV`, `learning_curve`, `validation_curve`

---

### Chapter 13
### Deploying scikit-learn Models in Production

Notebook ini membahas cara membawa model ML dari lingkungan pengembangan ke produksi yang sesungguhnya. Deployment adalah tahap akhir dari ML lifecycle dan sering kali menjadi tantangan terbesar: model yang bagus secara akademis tidak selalu mudah di-deploy di dunia nyata.

**Topik yang Dibahas:**
- **Overview Model Deployment**: ML lifecycle, tantangan deployment, perbedaan dev vs prod
- **Teknik serialisasi dan persistensi**:
  - `joblib.dump()` / `joblib.load()`: direkomendasikan untuk scikit-learn (efisien untuk array NumPy)
  - `pickle`: format Python standar, lebih portabel
  - Menyimpan dan memuat seluruh Pipeline (preprocessing + model)
- **Penskalaan model untuk produksi**:
  - Batch prediction vs real-time prediction
  - `MiniBatchKMeans` untuk incremental learning
  - `warm_start` untuk retraining inkremental tanpa restart dari awal
- **Pemantauan model di produksi**:
  - Model drift dan data drift: mengapa performa model menurun seiring waktu
  - Deteksi drift statistik: KL divergence, PSI (Population Stability Index)
  - Threshold-based alerting untuk performa model
- **Manajemen siklus hidup model**: versioning, reproducibility, audit trail
- **Pipeline deployment**: satu artefak tunggal untuk prod consistency
- Latihan komprehensif: pipeline deployment lengkap dengan monitoring dan retraining

**Kelas dan Fungsi Utama:** `joblib`, `pickle`, `Pipeline`, `MiniBatchKMeans`, `warm_start`, monitoring utilities

---

## Teknologi yang Digunakan

```
scikit-learn >= 1.4
numpy
pandas
matplotlib
seaborn
scipy
nltk
joblib
```

---

## Cara Menjalankan Notebook

1. Clone repository ini:
   ```bash
   git clone https://github.com/njwbilll/Tugas-2_scikit-learn-Cookbook-O-Reilly-_Najwa-Bilqis-Al-Khalidah.git
   cd Tugas-2_scikit-learn-Cookbook-O-Reilly-_Najwa-Bilqis-Al-Khalidah
   ```

2. Install dependencies:
   ```bash
   pip install scikit-learn numpy pandas matplotlib seaborn scipy nltk joblib
   ```

3. Download NLTK resources (diperlukan untuk Chapter 09):
   ```python
   import nltk
   nltk.download('movie_reviews')
   nltk.download('brown')
   nltk.download('reuters')
   nltk.download('stopwords')
   nltk.download('punkt_tab')
   nltk.download('averaged_perceptron_tagger_eng')
   ```

4. Buka Jupyter Notebook atau jalankan di Google Colab.

---

## Referensi

- **Buku:** scikit-learn Cookbook, Third Edition - John Sukup (Packt Publishing, 2025)
- **Dokumentasi resmi scikit-learn:** https://scikit-learn.org/stable/
- **Repository buku:** https://github.com/PacktPublishing/scikit-learn-Cookbook-Third-Edition

---

*Dibuat untuk memenuhi Tugas 2 Enrichment Machine Learning - Individual Task*
