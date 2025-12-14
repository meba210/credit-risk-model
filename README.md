## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability

The Basel II Capital Accord emphasizes accurate measurement and management of credit risk to ensure financial stability and sufficient capital reserves. This regulatory framework requires banks to use transparent, well-documented, and explainable models for risk estimation. As a result, credit scoring models must be interpretable so that risk decisions can be justified to regulators, auditors, and internal stakeholders. Clear documentation and explainability help ensure compliance, trust, and effective risk governance.

### 2. Need for a Proxy Default Variable

The dataset does not contain an explicit default label indicating whether a customer failed to repay a loan. Therefore, creating a proxy target variable is necessary to enable supervised learning. In this project, customer engagement behavior derived from transaction data is used as a proxy for credit risk. However, relying on a proxy introduces business risks such as misclassification, where disengaged customers may not necessarily be defaulters. This can lead to incorrect credit decisions, potential revenue loss, or unfair customer exclusion.

### 3. Trade-offs Between Simple and Complex Models

Simple and interpretable models, such as Logistic Regression with Weight of Evidence (WoE), offer transparency, ease of explanation, and regulatory acceptance. However, they may have lower predictive performance. Complex models like Gradient Boosting can capture non-linear relationships and achieve higher accuracy but are harder to interpret and explain. In a regulated financial environment, the trade-off lies between maximizing predictive power and ensuring compliance, explainability, and trust in decision-making.
