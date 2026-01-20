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
UC Berkeley School of Information partnered with Wolters Kluwer, a global leader in professional information services for legal, tax, healthcare, and compliance markets, to explore how modern AI can improve legal citation analysis. In a 14-week client capstone with weekly stakeholder meetings, we built an AI-powered legal research MVP showing that LLM-based systems can approach traditional citator accuracy while providing clearer, human-readable reasoning. The platform helps attorneys quickly assess whether a case remains “Good Law” by analyzing how later courts have treated it.
* **Tech/Methods:** Python, Streamlit, Neo4j/Cypher, knowledge graphs, RAG/GraphRAG (LangGraph), agentic tools, LLM ensemble (Claude 3.5 Sonnet, Mistral-7B, Llama 3-70B), prompt engineering, vector embeddings (Amazon Titan), AWS Bedrock, CourtListener API, model evaluation.

### 📝 [Text Detoxification System](https://github.com/kentbourgoing/english-text-detoxification-system-/tree/main)
Toxic language is common online and can harm users and communities. This project built an automated system that rewrites toxic text into safer alternatives, reducing toxicity by 75% while keeping the text natural. After testing 11 approaches, we identified the most effective methods for making online spaces safer without changing the original meaning.
* **Tech/Methods:** Python, PyTorch, Hugging Face Transformers, DecompX, masking/infilling/reranking pipelines, Mistral-7B, MaRCo BART, T5, XLM-R toxicity scoring, LaBSE semantic similarity, GPT-2 perplexity, benchmark evaluation.

### 🚗 [Child Safety Field Experiment](https://github.com/kentbourgoing/-reducing-child-accident-risk-through-visual-cues/tree/main)
In the U.S., child pedestrians continue to face serious risk, with about 385 deaths and 9,300+ injuries reported in 2023. Vehicle speed strongly affects both crash risk and injury severity, so even small reductions can matter. This project tested whether simple, low-cost “kids at play” cues can slow drivers without expensive road changes.
* **Tech/Methods:** R, experimental design, randomized field experiment, causal inference, linear regression, robust standard errors, hypothesis testing, data visualization (ggplot2).

### ☁️ [Full End-to-End ML API Deployment on AWS EKS](https://github.com/kentbourgoing/full-end-to-end-ML-API/tree/main)
Deployed an end-to-end sentiment analysis API on AWS that processes 70-100 requests per second with 100% uptime. This project demonstrates enterprise-level ML system engineering by combining containerization, orchestration, caching, and monitoring into a scalable production system. The implementation showcases critical skills for deploying ML models in real-world environments where performance, reliability, and cost efficiency matter.
* **Tech/Methods:** Python, FastAPI, Docker, Redis caching, AWS (EKS/ECR), Kubernetes (HPA, health checks), Istio routing, load testing (k6), monitoring (Grafana), NLP model serving (DistilBERT), reliability/performance testing.

### ✈️ [Flight Delay Prediction at Scale](https://github.com/kentbourgoing/flight-delay-prediction-at-scale)
Flight delays disrupt 2.9M+ daily passengers and spread across connected airline networks. This project built a scalable ML pipeline to predict departure status (Early, On-Time, Delayed) using 28M flight records (2015–2019), combining U.S. DOT flight data with NOAA weather on a distributed Apache Spark/Databricks platform.
* **Tech/Methods:** Apache Spark, PySpark, Databricks, distributed ML, feature engineering, blocked time-series cross-validation, class imbalance handling (over/under sampling, SMOTE), Logistic Regression, Random Forest, MLP, hyperparameter tuning (Grid Search, Optuna), probability calibration/recalibration, model evaluation.

### 🌍 [Atmospheric CO2 Trend Forecasting (Keeling Curve Analysis)](https://github.com/kentbourgoing/atmospheric-CO2-trend-forecasting/tree/master)
This project analyzed 40 years of CO2 data from Mauna Loa Observatory to forecast critical environmental thresholds. Understanding CO2 trends is essential for climate policy and planning. By building forecasting models from a 1997 perspective and validating them against actual data through 2024, the project revealed important insights about model performance and accelerating CO2 growth.
* **Tech/Methods:** R, time-series EDA, ARIMA/SARIMA, seasonal decomposition, stationarity checks/differencing, model selection (AIC/BIC), residual diagnostics (Ljung-Box), probabilistic forecasting, tidyverse, fable, ggplot2.

### ❤️ [Heart Failure Survival Prediction System](https://github.com/kentbourgoing/heart-failure-survival-prediction-system/tree/main)
Developed a machine learning system to predict survival outcomes in heart failure patients using clinical data from electronic medical records. With cardiovascular diseases causing 17 million deaths annually worldwide, this project demonstrates how machine learning can help healthcare professionals identify high-risk patients early and improve clinical decision-making.
* **Tech/Methods:** Python, scikit-learn, TensorFlow/Keras, supervised learning, model comparison, hyperparameter tuning, Logistic Regression, K-Nearest Neighbors (KNN), K-Means Clustering, Gradient Boosting, Decision Tree, Random Forest, AdaBoost, Neural Networks (MLP), synthetic data augmentation (GMM), evaluation (accuracy/precision/recall), threshold tuning.

### 🚇 [Bay Area Food Delivery Optimization (BART Transit Analysis)](https://github.com/kentbourgoing/bay-area-food-delivery-optimization/tree/main)
A simulated company, Acme Gourmet Meals (AGM), needed to expand its food delivery across the Bay Area by placing pickup points at BART stations and optimizing delivery routes. This project shows how a multi-database (NoSQL + graph) approach can outperform a relational-only setup for real-time delivery operations and network optimization.
* **Tech/Methods:** PostgreSQL, Neo4j/Cypher, Python (pandas, psycopg2), ETL pipelines, graph analytics, shortest paths (Dijkstra), community detection (Louvain), centrality/PageRank, geospatial analysis, data visualization.

### 📊 [Labor Economics Analysis](https://github.com/kentbourgoing/labor-economics-analysis)
This project analyzed the relationship between weekly work hours and annual earnings using real-world US Census data from 69,000+ employed workers. Understanding this relationship is crucial for informing labor policy, workplace regulations, and helping workers make informed career decisions. Working in a 4-person team, we applied statistical modeling techniques to quantify how additional work hours translate into salary gains, revealing insights into the economic trade-offs of work-life balance.
* **Tech/Methods:** R, data cleaning, log transformation, OLS regression, model comparison (linear/log-linear/polynomial), RMSE, residual diagnostics, hypothesis testing, tidyverse, ggplot2.

---

## 📫 Connect With Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kent_Bourgoing-blue?style=flat&logo=linkedin)](https://linkedin.com/in/kent-bourgoing)
[![Email](https://img.shields.io/badge/Email-kent1bp%40berkeley.edu-red?style=flat&logo=gmail)](mailto:kent1bp@berkeley.edu)

---

