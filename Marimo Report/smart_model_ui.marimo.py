

import marimo as mo

__generated_with = "0.11.17"
app = mo.App(
    width="medium",
    app_title="Mathematic classification",
    html_head_file=r"D:\Group_D_21\Marimo Report\design.html",
    auto_download=["html"],
)


@app.cell
def _():
    def _():
        return mo.center(mo.image(r"D:\images.png"))
    _()
    return


@app.cell
def abstract():
    mo.md(
        """
        ###**Abstract**

        - This project presents an interactive machine learning application developed using the Marimo framework.
        - It enables users to select, train, and evaluate classification models on a Smart Meter DDoS attack dataset.
        - Essential machine learning modules are integrated, along with dynamic data preprocessing, model selection, and visualization of results.
        - Here, we have emphasised 3 important models which might help in understanding the data provided by our dataset and how they help us identify the label or the class: **Naive Bayes, Decision Tree and Logistic Regression.**
        - The application emphasizes modularity, user interactivity, and streamlined evaluation, providing a flexible and intuitive platform for understanding model performance and comparison.
        """
    )
    return


@app.cell
def _():
    mo.md(
        """
        ###**Introduction**

        This project detects anomalies in smart meter readings using machine learning.

        Analysis is done across MATLAB, Marimo, and Weka platforms.

        Models like Random Forest, Naive Bayes, Decision Tree, and Logistic Regression are applied.

        The dataset is split into 80% training and 20% testing.

        The goal is to identify abnormal patterns to ensure reliable and secure energy monitoring.
        """
    )
    return


@app.cell
def _():
    mo.md(
        """
        ###**Dataset Description**

        **Datasets Used:**

        Normal Traffic: Contains clean network traffic without any attacks or anomalies.

        TCP_SYN Attack with Sniffing: Captures traffic during a TCP SYN flood attack while network sniffing is active.

        TCP_SYN Attack without Sniffing: Captures traffic during a TCP SYN flood attack without any additional sniffing activity.

        **Feature Categories:**

        Network Info: Identifies source and destination (e.g., Src_ip, Dst_ip, Protocol).

        Traffic Stats: Measures data flow rate (e.g., flow_byts_s, flow_pkts_s).

        Packet Details: Tracks packet size variations (e.g., Pkt_len_max, Pkt_len_mean).

        TCP Flags: Identifies connection behavior (e.g., Syn_flag_cnt, Ack_flag_cnt).

        Time-Based: Detects timing irregularities (e.g., Flow_duration, Flow_iat_mean).

        Labels:

        Indicate whether the traffic is Normal or an Anomaly, serving as ground truth for classification.
        """
    )
    return


@app.cell
def _():
    mo.md(
        """
        ###**Methodology**

        **Dataset Preparation:**
        Load the smart meter readings from the CSV file.

        **Data Splitting:**
        Split the data into 80% training and 20% testing sets.

        **Platform-Specific Implementation:**

        In **MATLAB**: Apply Random Forest, Naive Bayes, and Decision Tree models.

        In **Marimo**: Use Logistic Regression, Naive Bayes, and Decision Tree models.

        In **Weka**: Apply Random Forest, Naive Bayes, and Decision Tree classifiers.

        **Model Training and Evaluation:**
        Train the models using the training data and evaluate their performance on the testing data.

        ###**Result Comparison:**
        Compare the accuracy and performance metrics of the models across different platforms.
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    mo.md("### 🤖 Choose a Machine Learning Model")

    # Model selector (only Naive Bayes, Decision Trees, Logistic Regression)
    model_selector = mo.ui.dropdown(
        {
            "Naive Bayes": "nb",
            "Decision Tree": "dtree",
            "Logistic Regression": "lr"
        },
        label="Select Model"
    )

    # Checkbox to enable/disable comparison plot
    show_comparison_plot = mo.ui.checkbox(label="Show Accuracy Comparison Plot")

    mo.hstack([model_selector, show_comparison_plot])
    return mo, model_selector, show_comparison_plot


@app.cell
def _():
    mo.md(
        """
        ###**📖Modules used in our code**

        - pandas
        - matplotlib.pyplot
        - seaborn
        - train_test_split from sklearn.model_selection
        - StandardScaler from sklearn.preprocessing 
        - accuracy_score, classification_report, confusion_matrix from sklearn.metrics 
        - LogisticRegression from sklearn.linear_model 
        - GaussianNB from sklearn.naive_bayes 
        - DecisionTreeClassifier from sklearn.tree
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    return (
        DecisionTreeClassifier,
        GaussianNB,
        LogisticRegression,
        StandardScaler,
        accuracy_score,
        classification_report,
        confusion_matrix,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _():
    mo.md(
        """
        ###**🔄Loading the dataset**

        - **smoted_dataset1.csv** The file where we contain the data for classification.
        """
    )
    return


@app.cell
def _(pd):
    file_path = r"C:\Users\yashw_d6scpoj\Desktop\S2\MFC2\smoted_dataset1.csv"
    df = pd.read_csv(file_path)
    return df, file_path


@app.cell
def _():
    mo.md(
        """
        ### 🧹 **Data Preparation**

        - Splits data into features (`X`) and labels (`y`)
        - Performs a train-test split `(80% train, 20% test)` **X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)**
        - Scales the features using `StandardScaler` - Used in specifying the range of values present as data, so that training and testing becomes easy.
        """
    )
    return


@app.cell
def _(StandardScaler, df, train_test_split):
    if "Label" not in df.columns:
        mo.md("⚠️ The dataset must contain a 'Label' column.")
    else:
        # Prepare data
        X = df.drop("Label", axis=1)
        y = df["Label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    return (
        X,
        X_test,
        X_test_scaled,
        X_train,
        X_train_scaled,
        scaler,
        y,
        y_test,
        y_train,
    )


@app.cell
def _():
    mo.md(
        """
        ### 🤖 **Model Selection & Training**

        - Retrieves selected model from dropdown
        - Trains the model on scaled training data
        - Makes predictions on test data
        - Calculates accuracy, classification report, and confusion matrix
        """
    )
    return


@app.cell
def _(
    DecisionTreeClassifier,
    GaussianNB,
    LogisticRegression,
    X_test_scaled,
    X_train_scaled,
    accuracy_score,
    classification_report,
    confusion_matrix,
    model_selector,
    y_test,
    y_train,
):
    # Store accuracy for comparison
    accuracy_dict = {}

    # Model list
    models = {
        "nb": GaussianNB(),
        "dtree": DecisionTreeClassifier(),
        "lr": LogisticRegression(max_iter=200, random_state=42)
    }

    # Selected model
    selected_model_type = model_selector.value
    selected_model = models[selected_model_type]

    # Fit selected model
    selected_model.fit(X_train_scaled, y_train)
    y_pred = selected_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return (
        acc,
        accuracy_dict,
        cm,
        models,
        report,
        selected_model,
        selected_model_type,
        y_pred,
    )


@app.cell
def _():
    mo.md(
        """
        ### **Results and Discussion**

        - Shows model accuracy
        - Displays classification report
        - Plots a confusion matrix using Seaborn
        """
    )
    return


@app.cell
def _(acc, cm, mo, plt, report, selected_model_type, sns):
    # Show selected model results
    mo.md(f"### ✅ Accuracy: `{acc:.4f}`")
    mo.md("### 📊 Classification Report:")
    mo.md(f"```\n{report}\n```")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {selected_model_type}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    mo.md(
        """
        ### 📊 **Accuracy Comparison (Optional)**

        If the checkbox is selected, this block:

        - Trains all three models
        - Calculates their accuracies
        - Displays a bar chart comparing model performance
        """
    )
    return


@app.cell
def _(
    X_test_scaled,
    X_train_scaled,
    accuracy_dict,
    accuracy_score,
    models,
    plt,
    show_comparison_plot,
    sns,
    y_test,
    y_train,
):
    # If checkbox enabled, calculate accuracies for all models and show comparison
    if show_comparison_plot.value:
        for key, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred_all = model.predict(X_test_scaled)
            accuracy_dict[key] = accuracy_score(y_test, y_pred_all)

        # Plotting accuracy comparison
        plt.figure(figsize=(6, 4))
        model_names = {
            "nb": "Naive Bayes",
            "dtree": "Decision Tree",
            "lr": "Logistic Regression"
        }
        labels = [model_names[k] for k in accuracy_dict.keys()]
        scores = list(accuracy_dict.values())

        sns.barplot(x=labels, y=scores, palette="viridis")
        plt.title("📈 Accuracy Comparison of Models")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
    return key, labels, model, model_names, scores, y_pred_all


@app.cell
def _():
    mo.md(
        """
        ###**Conclusion**

        **Anomaly detection** was successfully implemented using machine learning.

        Models were tested on **MATLAB, Marimo, and Weka**.

        Random Forest, Naive Bayes, Decision Tree, and Logistic Regression were used for classification.

        The project demonstrated the effectiveness of machine learning in **smart meter** data analysis.

        Future improvements could include advanced models and additional features for better accuracy.
        """
    )
    return


@app.cell
def _():
    mo.md(
        """
        ###**References**

        [1] M. A. Hossain and M. S. Islam, “Enhancing DDoS attack detection with hybrid feature selection and ensemble-based classifier: A promising solution for robust cybersecurity,” Measurement: Sensors, vol. 32, p. 101037, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2665917424000138

        [2] M. Mohurle and V. V. Panchbhai, "Review on realization of AES encryption and decryption with power and area optimization," 2016 IEEE 1st International Conference on Power Electronics, Intelligent Control and Energy Systems (ICPEICES), Delhi, India, 2016, pp. 1-3. [Online]. Available: https://www.researchgate.net/publication/313805219_Review_on_realization_of_AES_encryption_and_decryption_with_power_and_area_optimization

        [3] B. Zhang, G. Ma, X. Lu, and W. Xu, "Study on Hybrid Encryption Technology of Power Gateway Based on AES and RSA Algorithm," 2022 14th International Conference on Signal Processing Systems (ICSPS), Jiangsu, China, 2022, pp. 640-644. [Online]. Available: https://www.computer.org/csdl/proceedings-article/icsps/2022/363100a640/1PTOEN5363m

        [4] B. Harjito, H. M. Sukarno, and Winarno, "Performance Analysis of Kyber-DNA and RSA-Base64 Algorithms in Symmetric Key-Exchange Protocol," 2024 Ninth International Conference on Informatics and Computing (ICIC), Medan, Indonesia, 2024, pp. 1-6. [Online]. Available: https://icic-aptikom.org/2024/wp-content/uploads/2024/10/Program-Book-ICIC-2024-Binder-2.pdf
        """
    )
    return


@app.cell
def _():
    mo.md(
        """
        ###**Acknowledgement**

        I would like to express my sincere gratitude to all those who contributed to the success of this project.

        I am deeply grateful to my professor **Sunil Kumar Sir** for their valuable guidance and continuous support throughout the project.

        I would also like to thank my friends and teammate for their collaboration and insightful feedback during the development process.

        Special thanks to the creators and maintainers of the MATLAB, Marimo, and Weka platforms for providing powerful tools that made this project possible.

        Lastly, I would like to acknowledge the authors of the papers I referenced, whose work inspired and supported my research in anomaly detection.
        """
    )
    return


if __name__ == "__main__":
    app.run()
