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

### 🏛️ [AI-Powered Legal Citation Analyzer](https://github.com/kentbourgoing/legal-citation-analyzer) (Wolters Kluwer Capstone)
Delivered a legal research MVP in a 14-week client capstone with weekly stakeholder meetings. Built a Neo4j knowledge graph (3.5K cases, 5.5K citations), optimized citation sentiment classification to 67% accuracy using a 3-model LLM ensemble (Claude 3.5 Sonnet, Mistral-7B, Llama 3-70B), and developed an agentic GraphRAG chatbot with LangGraph orchestrating 8 tools for legal research.  
**Tech:** Python, Neo4j, LangGraph, Mistral-7B, Claude API, Streamlit, AWS | **Impact:** Functional MVP delivered on schedule with configurable case labeling for different practice areas

### 📝 [Text Detoxification System](https://github.com/kentbourgoing/text-detoxification)
Built a modular NLP pipeline that reduced toxicity by 75% (0.208→0.051) while preserving meaning (93.6% similarity) and improving fluency. Tested 11 configurations combining DecompX token attribution with LLM-based masking and infilling, developing a novel Global Reranking method that jointly optimized toxicity, semantic similarity, and perplexity.  
**Tech:** PyTorch, HuggingFace Transformers, Mistral-7B, T5, XLM-R, GPT-2 | **Impact:** 35-75% toxicity reduction across all configurations

### ☁️ [Scalable ML API on AWS EKS](https://github.com/kentbourgoing/aws-ml-api)
Deployed a production sentiment analysis API (DistilBERT) handling 70-100 req/s with zero downtime and p99 latency under 2s. Implemented Redis caching (95%+ hit rate) to cut infrastructure costs 40% and configured Kubernetes HPA for auto-scaling (1-70 pods) with Istio service mesh routing.  
**Tech:** FastAPI, Docker, Redis, AWS EKS, Kubernetes, Istio, Grafana, k6 | **Impact:** 28K+ requests tested with 100% success rate

### 🚗 [Child Safety Field Experiment](https://github.com/kentbourgoing/child-safety-experiment)
Designed a randomized 1×4 field experiment (n=497) to estimate the causal impact of low-cost visual cues on residential vehicle speed. Estimated a 1.9 mph speed reduction and 18 percentage point increase in 25-mph compliance (p<0.001) using linear regression with robust standard errors.  
**Tech:** R, experimental design, linear regression, ggplot2 | **Impact:** Statistically significant results demonstrating effectiveness of low-cost traffic calming measures

### ✈️ [Flight Delay Prediction at Scale](https://github.com/kentbourgoing/flight-delay-prediction)
Built a distributed ML pipeline on Databricks processing 28M flight records (2015-2019) to predict departure status (Early/On-Time/Delayed). Implemented blocked time-series cross-validation to prevent temporal leakage and improved F1 by 6% through probability calibration to correct oversampling bias.  
**Tech:** PySpark, Databricks, Apache Spark, MLP, Random Forest | **Impact:** 54.6% F1 with 80.3% recall on majority class

### ❤️ [Heart Failure Survival Prediction](https://github.com/kentbourgoing/heart-failure-prediction)
Predicted heart failure survival with 85% accuracy (77% precision, 63% recall) by training 21 ML models including Gradient Boosting, Random Forest, AdaBoost, and Neural Networks. Implemented GMM data augmentation to expand dataset from 299 to 5,299 samples and optimized decision thresholds to prioritize detection of at-risk patients.  
**Tech:** Python, scikit-learn, TensorFlow/Keras, GMM | **Impact:** 85% test accuracy with models prioritizing recall to reduce missed diagnoses

### 🌍 [Atmospheric CO₂ Forecasting (Keeling Curve)](https://github.com/kentbourgoing/co2-forecasting)
Forecasted CO₂ trends using 468 observations (1958-1997) with seasonal ARIMA models. Projected 420 ppm by April 2030 and 500 ppm by May 2072 with 95% confidence intervals. Validated forecasts against 2024 data revealed actual CO₂ growth exceeded predictions by 8 years.  
**Tech:** R, ARIMA/SARIMA, time series analysis, fable | **Impact:** Long-term probabilistic forecasts with quantified uncertainty

### 📊 [Labor Economics Analysis](https://github.com/kentbourgoing/labor-economics-analysis)
Analyzed the relationship between work hours and salary using 69,148 CPS ASEC 2023 workers. Applied log transformation and tested multiple regression specifications (linear, log-linear, polynomial), estimating a significant earnings effect of +$2,232 per hour (linear) and ~5.6% per hour (log model), improving R² from 0.083 to 0.307.  
**Tech:** R, linear regression, statistical modeling, ggplot2 | **Impact:** Evidence-based insights for labor policy discussions on work-life balance and income

### 🚇 [Bay Area Food Delivery Optimization](https://github.com/kentbourgoing/bart-delivery-optimization)
Built a polyglot data platform (PostgreSQL + Neo4j) for 12 months of BART ridership data (50 stations). Identified 11 distribution zones using Louvain community detection and recommended 9 strategic pickup locations plus 1 central hub using Dijkstra shortest path and centrality metrics (closeness/betweenness/PageRank).  
**Tech:** PostgreSQL, Neo4j, Cypher, Python ETL, graph analytics | **Impact:** Strategic recommendations for delivery network optimization using graph-based routing

---

## 💼 Professional Experience

**Environmental Engineer Consultant** @ Yorke Engineering, LLC (Sept 2023 - Present)  
- Reduced client compliance costs by $100K+ annually leading air quality projects for 100+ facilities
- Supported regulator-approved GHG report revision saving $5M by auditing emissions data and delivering analysis
- Improved project delivery efficiency 30% by integrating AI tools into workflows

---

## 📫 Connect With Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kent_Bourgoing-blue?style=flat&logo=linkedin)](https://linkedin.com/in/kent-bourgoing)
[![Email](https://img.shields.io/badge/Email-kent1bp%40berkeley.edu-red?style=flat&logo=gmail)](mailto:kent1bp@berkeley.edu)

---

