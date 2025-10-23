# Online Retail Analysis and Customer Segmentation Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--learn%20%7C%20Matplotlib-orange.svg)

## 1. Project Description

This is an end-to-end data analysis project that uses a real dataset of transactions from a UK-based online retailer. The primary goal is to transform raw sales data into strategic insights and actionable business recommendations.

The analysis covers everything from data cleaning and preparation to advanced customer segmentation using the RFM (Recency, Frequency, Monetary) model and the K-Means clustering algorithm, culminating in the construction of machine learning models to predict customer churn.

## 2. Data Source

The dataset used is the **"Online Retail Dataset"**, publicly available from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/online+retail).

* **Description:** Contains real transactions that occurred between 12/01/2010 and 12/09/2011.
* **Key Attributes:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

## 3. Methodology and Analysis Phases

The project was structured into five main phases, following a standard Data Science workflow:

### Phase 1: Data Cleaning and Preparation
* Handling missing values (especially `CustomerID`).
* Removing duplicate transactions.
* Handling canceled transactions (invoices starting with 'C') and zero unit prices.
* Feature Engineering:
    * Creation of the `TotalPrice` column (`Quantity` * `UnitPrice`).
    * Extraction of information from `InvoiceDate`, such as `Month`, `DayOfWeek`, and `Hour`.

### Phase 2: Exploratory Data Analysis (EDA)
* Analysis of sales trends over time (monthly, daily, and hourly).
* Identification of top-selling products (by volume and by revenue).
* Analysis of the geographical distribution of sales and customers by country.
* Visualization of purchasing patterns.

### Phase 3: Customer Segmentation (RFM + K-Means)
The objective of this phase was to identify different customer profiles for personalized marketing actions.

1.  **RFM Calculation:**
    * **Recency (R):** Number of days since the last purchase.
    * **Frequency (F):** Total number of unique transactions.
    * **Monetary (M):** Total amount spent by the customer.
2.  **K-Means Clustering:**
    * RFM data was standardized using `StandardScaler`.
    * The **Elbow Method** was used to determine the optimal number of clusters.
    * The K-Means algorithm was applied to group customers into distinct segments.
3.  **Segment Analysis:** Interpretation and naming of each cluster (e.g., "Champion Customers", "At-Risk Customers", "New Customers").

### Phase 4: Hypothesis Testing
Statistical validation of business assumptions:
* **T-Test (Independent Samples):** Checked for a statistically significant difference in the average order value between two countries (e.g., Germany vs. France).
* **Chi-Squared Test:** Checked for an association between the day of the week and the likelihood of a high-value item being purchased.

### Phase 5: Predictive Modeling (Machine Learning)
Two main models were developed to support decision-making:

1.  **Churn Prediction Model (Classification):**
    * **Objective:** Predict the probability of a customer being classified as "inactive" or "churned" in the coming months.
    * **Models:** Logistic Regression (baseline) and Random Forest Classifier.
    * **Metrics:** Accuracy, Precision, Recall, F1-Score, and ROC/AUC Curve.

2.  **Spending Prediction Model (Regression):**
    * **Objective:** Predict the monetary value a customer will spend over the next 3 months.
    * **Models:** Linear Regression and Random Forest Regressor.
    * **Metrics:** R² (R-squared) and RMSE (Root Mean Squared Error).

## 4. Key Insights and Business Recommendations

## 5. Technologies Used

* **Language:** Python 3
* **Core Libraries:**
    * `pandas`: For data manipulation and analysis.
    * `numpy`: For numerical operations.
    * `matplotlib`: For static visualizations.
    * `scikit-learn`: For preprocessing (StandardScaler) and modeling (KMeans, Logistic Regression, Random Forest).
    * `Jupyter Notebook`: For iterative analysis development.

## 6. How to Run the Project

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/your-username/project-name.git](https://github.com/your-username/project-name.git)
    cd project-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Create a `requirements.txt` file with `pip freeze > requirements.txt` after installing the libraries).*

4.  **Download the data:**
    * Download the `Online Retail.xlsx` file from [this link](http://archive.ics.uci.edu/ml/machine-learning-databases/00352/).
    * Place the file in a folder named `/data/` within the project.

5.  **Run the Notebook:**
    * Start Jupyter Notebook:
        ```bash
        jupyter notebook
        ```
    * Open and run the main project notebook (e.g., `retail_analysis.ipynb`).
