# Hi, I'm Kent 👋

Recent UC Berkeley Master of Information and Data Science (MIDS) graduate with experience building ML systems that deliver measurable business impact. I leverage machine learning, causal inference, time series, NLP, and cloud deployment skills alongside consulting experience to deliver stakeholder-ready results.

**🎓 Education:**  
- Master of Information and Data Science, UC Berkeley School of Information — *December 2025* (GPA: 4.0/4.0)  
- Bachelor of Science, Chemical Engineering, UCLA — *June 2023* (GPA: 3.73/4.0)  

---

## 🛠️ Skills

- **Programming Languages:** Python, R, SQL, MATLAB
- **Visualization:** Matplotlib, Seaborn, ggplot2
- **Frameworks/Apps:** Streamlit, FastAPI
- **Databases/Data Engineering:** PostgreSQL, Neo4j (Cypher), Redis, ETL, Docker
- **ML & AI:** scikit-learn, PyTorch, TensorFlow/Keras, Transformers (Hugging Face), NLP, LLMs, RAG/GraphRAG (LangGraph), Random Forest, KNN, CNN, K-Means, PCA/SVD, Gradient Boosting, AdaBoost, MLP
- **Cloud/MLOps:** AWS, Kubernetes (EKS), Istio, Grafana, k6
- **Statistics/Experimentation:** A/B Testing, Experimental Design, Causal Inference, Linear/Logistic Regression, Time Series (ARIMA/SARIMA), Panel Data, pandas, NumPy, RStudio
- **Distributed Computing:** Spark, PySpark, Databricks, Hadoop, HDFS, MapReduce
- **Tools:** Git/GitHub, Jupyter, VS Code
- **Platforms:** Windows, Linux, Bash
- **Languages:** Spanish (native)

---

## 🚀 Projects Portfolio

### 🏛️ [AI-Powered Legal Citation Analyzer](https://github.com/kentbourgoing/ai-powered-legal-citation-analyzer) (UC Berkeley × Wolters Kluwer Client Capstone)
Built an AI legal research MVP that helps attorneys assess whether a case remains “Good Law” by analyzing how later courts treated it. Delivered as a dual-mode Streamlit product: Case Lookup (search ADA cases, Good/Bad/Moderate labeling, citation analysis, CSV export) and a chatbot for plain-English questions about precedent and citation history. Developed a Neo4j knowledge graph of ADA cases and citations and added paragraph-level rationales so users can see why the system labels a citation Positive, Negative, or Neutral. The project was delivered through a 14-week client engagement with weekly stakeholder reviews and iterative milestones.
* **Tech/Methods:** Python, Streamlit, Neo4j/Cypher, knowledge graphs, RAG/GraphRAG (LangGraph), agentic tools, LLM ensemble (Claude 3.5 Sonnet, Mistral-7B, Llama 3-70B), prompt engineering, vector embeddings (Amazon Titan), AWS Bedrock, CourtListener API, model evaluation.

### 📝 [Text Detoxification System](https://github.com/kentbourgoing/english-text-detoxification-system-/tree/main)
Built a modular NLP system that rewrites toxic text into safer alternatives while preserving meaning and fluency. Tested 11 pipeline variants and introduced a multi-objective reranking approach that balances toxicity reduction, semantic similarity, and language quality. Achieved a 75% toxicity reduction while keeping outputs natural and close to the original intent.
* **Tech/Methods:** Python, PyTorch, Hugging Face Transformers, DecompX, masking/infilling/reranking pipelines, Mistral-7B, MaRCo BART, T5, XLM-R toxicity scoring, LaBSE semantic similarity, GPT-2 perplexity, benchmark evaluation.

### 🚗 [Child Safety Field Experiment](https://github.com/kentbourgoing/-reducing-child-accident-risk-through-visual-cues/tree/main)
Designed and ran a randomized field experiment to test whether low-cost “kids at play” visual cues reduce vehicle speeds on a residential street. Measured free-flow speeds and estimated causal effects across multiple conditions (control, sign only, sign + toys, sign + toys + balloons). Results show that higher-visibility cues can produce meaningful speed reductions without major road changes.
* **Tech/Methods:** R, experimental design, randomized field experiment, causal inference, linear regression, robust standard errors, hypothesis testing, data visualization (ggplot2).

### ☁️ [Full End-to-End ML API Deployment on AWS EKS](https://github.com/kentbourgoing/full-end-to-end-ML-API/tree/main)
Deployed a production-style sentiment analysis API on AWS EKS and validated performance under sustained load. Built a FastAPI service with Docker, Redis caching, Kubernetes autoscaling, and monitoring. Sustained 70–100 requests/second with zero 5xx errors during load tests and low tail latency.
* **Tech/Methods:** Python, FastAPI, Docker, Redis caching, AWS (EKS/ECR), Kubernetes (HPA, health checks), Istio routing, load testing (k6), monitoring (Grafana), NLP model serving (DistilBERT), reliability/performance testing.

### ✈️ [Flight Delay Prediction at Scale](https://github.com/kentbourgoing/flight-delay-prediction-at-scale)
Built a distributed ML pipeline to predict flight departure status (Early, On-Time, Delayed) using 28M flight records combined with NOAA weather. Engineered features at scale, prevented time leakage using blocked time-series validation, and handled class imbalance with resampling and calibration. Delivered a modular Spark/Databricks workflow designed for large, multimodal data.
* **Tech/Methods:** Apache Spark, PySpark, Databricks, distributed ML, feature engineering, blocked time-series cross-validation, class imbalance handling (over/under sampling, SMOTE), Logistic Regression, Random Forest, MLP, hyperparameter tuning (Grid Search, Optuna), probability calibration/recalibration, model evaluation.

### 🌍 [Atmospheric CO2 Trend Forecasting (Keeling Curve Analysis)](https://github.com/kentbourgoing/atmospheric-CO2-trend-forecasting/tree/master)
Analyzed decades of Mauna Loa CO2 measurements to forecast long-term thresholds and quantify uncertainty. Built time-series models from a historical “1997 viewpoint,” then validated predictions against observed data through 2024 to understand model drift and changing dynamics. Highlighted where forecasts break down and why periodic model updates matter.
* **Tech/Methods:** R, time-series EDA, ARIMA/SARIMA, seasonal decomposition, stationarity checks/differencing, model selection (AIC/BIC), residual diagnostics (Ljung-Box), probabilistic forecasting, tidyverse, fable, ggplot2.

### ❤️ [Heart Failure Survival Prediction System](https://github.com/kentbourgoing/heart-failure-survival-prediction-system/tree/main)
Developed a supervised ML system to predict survival outcomes for heart failure patients using clinical records. Compared multiple model families, tuned hyperparameters, and used synthetic data augmentation to address small-sample limits. Designed the evaluation to reduce missed high-risk cases by tuning decision thresholds.
* **Tech/Methods:** Python, scikit-learn, TensorFlow/Keras, supervised learning, model comparison, hyperparameter tuning, Logistic Regression, K-Nearest Neighbors (KNN), K-Means Clustering, Gradient Boosting, Decision Tree, Random Forest, AdaBoost, Neural Networks (MLP), synthetic data augmentation (GMM), evaluation (accuracy/precision/recall), threshold tuning.

### 🚇 [Bay Area Food Delivery Optimization (BART Transit Analysis)](https://github.com/kentbourgoing/bay-area-food-delivery-optimization/tree/main)
Designed a data + graph analytics solution to optimize pickup locations and delivery routing across the Bay Area. Built ETL workflows into PostgreSQL and modeled the transit network in Neo4j to compute shortest paths, rank hubs, and identify service communities. Demonstrates why graph approaches can outperform relational-only setups for routing and network problems.
* **Tech/Methods:** PostgreSQL, Neo4j/Cypher, Python (pandas, psycopg2), ETL pipelines, graph analytics, shortest paths (Dijkstra), community detection (Louvain), centrality/PageRank, geospatial analysis, data visualization.

### 📊 [Labor Economics Analysis](https://github.com/kentbourgoing/labor-economics-analysis)
Analyzed US Census microdata (69,000+ workers) to quantify the relationship between weekly work hours and annual earnings. Cleaned and transformed skewed income data, tested multiple regression forms, and validated model assumptions using residual diagnostics. Delivered interpretable estimates that support policy and workforce discussions.
* **Tech/Methods:** R, data cleaning, log transformation, OLS regression, model comparison (linear/log-linear/polynomial), RMSE, residual diagnostics, hypothesis testing, tidyverse, ggplot2.

---

## 📫 Connect With Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kent_Bourgoing-blue?style=flat&logo=linkedin)](https://linkedin.com/in/kent-bourgoing)
[![Email](https://img.shields.io/badge/Email-kent1bp%40berkeley.edu-red?style=flat&logo=gmail)](mailto:kent1bp@berkeley.edu)

---

