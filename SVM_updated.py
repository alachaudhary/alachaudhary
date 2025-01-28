{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/alachaudhary/alachaudhary/blob/main/SVM_updated.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Y4PcW3h2wh51"
      },
      "source": [
        "---   \n",
        " <img align=\"left\" width=\"75\" height=\"75\"  src=\"https://upload.wikimedia.org/wikipedia/commons/c/c8/Umt_logo.png\">\n",
        "\n",
        "<h1 align=\"center\">Department of Computer Science</h1>\n",
        "<h1 align=\"center\">Course: Machine Learning</h1>\n",
        "\n",
        "---\n",
        "<h3><div align=\"right\">Instructor: Hafiz Abdul Rehman</div></h3>    \n",
        "  "
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "s4eMpJawIwsb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ut6i2Gll3H-t"
      },
      "source": [
        "<h1 align=\"center\">Assignment 3: SVM Kernel Selection</h1>\n",
        "<h1 align=\"center\">Submitted by: F2021266434</h1>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "QDgEBXxf3H-u"
      },
      "outputs": [],
      "source": [
        "# Step 1: Load the Dataset\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "from sklearn.metrics import classification_report, confusion_matrix\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Load the new dataset\n",
        "\n",
        "df = pd.read_csv('User_Data.csv')"
      ],
      "metadata": {
        "id": "PDt-nCcA4BV2"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Dataset Information:\")\n",
        "print(df.info())\n",
        "print(\"First few rows:\")\n",
        "print(df.head())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yPxgLTyA5CwS",
        "outputId": "e9fac967-ab8b-404c-bbf8-f5d759863d61"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dataset Information:\n",
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 400 entries, 0 to 399\n",
            "Data columns (total 4 columns):\n",
            " #   Column           Non-Null Count  Dtype \n",
            "---  ------           --------------  ----- \n",
            " 0   Gender           400 non-null    object\n",
            " 1   Age              400 non-null    int64 \n",
            " 2   EstimatedSalary  400 non-null    int64 \n",
            " 3   Purchased        400 non-null    int64 \n",
            "dtypes: int64(3), object(1)\n",
            "memory usage: 12.6+ KB\n",
            "None\n",
            "First few rows:\n",
            "   Gender  Age  EstimatedSalary  Purchased\n",
            "0    Male   19            19000          0\n",
            "1    Male   35            20000          0\n",
            "2  Female   26            43000          0\n",
            "3  Female   27            57000          0\n",
            "4    Male   19            76000          0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 2: Exploratory Data Analysis (EDA)\n",
        "print(\"Summary Statistics:\")\n",
        "print(df.describe())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RrwTdNLY5NbH",
        "outputId": "34fedb5c-c402-4778-dcfd-79391949bafc"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Summary Statistics:\n",
            "              Age  EstimatedSalary   Purchased\n",
            "count  400.000000       400.000000  400.000000\n",
            "mean    37.655000     69742.500000    0.357500\n",
            "std     10.482877     34096.960282    0.479864\n",
            "min     18.000000     15000.000000    0.000000\n",
            "25%     29.750000     43000.000000    0.000000\n",
            "50%     37.000000     70000.000000    0.000000\n",
            "75%     46.000000     88000.000000    1.000000\n",
            "max     60.000000    150000.000000    1.000000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Missing values in the dataset:\")\n",
        "print(df.isnull().sum())\n",
        "\n",
        "# Step 3: Preprocessing\n",
        "# Encode the Gender column (categorical)\n",
        "df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8Pz3Ao405UEZ",
        "outputId": "b880fda8-94b8-4494-c984-6e298c5bbdba"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Missing values in the dataset:\n",
            "Gender             0\n",
            "Age                0\n",
            "EstimatedSalary    0\n",
            "Purchased          0\n",
            "dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Define features (X) and target (y)\n",
        "X = df[['Gender', 'Age', 'EstimatedSalary']]\n",
        "y = df['Purchased']"
      ],
      "metadata": {
        "id": "NJOkUO5v6ApI"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "scaler = StandardScaler()\n",
        "X = scaler.fit_transform(X)"
      ],
      "metadata": {
        "id": "XgeXPc2N7X46"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 4: Train-Test Split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n"
      ],
      "metadata": {
        "id": "k1pDRDRX7b7i"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 5: Hyperparameter and Kernel Selection\n",
        "# Using GridSearchCV to find the best hyperparameters for SVM\n",
        "param_grid = {\n",
        "    'C': [0.1, 1, 10, 100],\n",
        "    'gamma': [1, 0.1, 0.01, 0.001],\n",
        "    'kernel': ['linear', 'poly', 'rbf']\n",
        "}\n",
        "\n",
        "grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)\n",
        "grid.fit(X_train, y_train)\n",
        "\n",
        "print(\"Best Parameters:\", grid.best_params_)\n",
        "best_model = grid.best_estimator_"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XXGW5Bw57mpf",
        "outputId": "6cbd0a5a-1902-4946-9202-682086dd1316"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Fitting 5 folds for each of 48 candidates, totalling 240 fits\n",
            "[CV] END ......................C=0.1, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ........................C=0.1, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END ........................C=0.1, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END ........................C=0.1, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END ........................C=0.1, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END ........................C=0.1, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ....................C=0.1, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=0.1, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=0.1, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=0.1, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=0.1, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ...................C=0.1, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ...................C=0.1, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ...................C=0.1, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ...................C=0.1, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ...................C=0.1, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END .....................C=0.1, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=0.1, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=0.1, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=0.1, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=0.1, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ..................C=0.1, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ..................C=0.1, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ..................C=0.1, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ..................C=0.1, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ..................C=0.1, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=0.1, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ....................C=0.1, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ....................C=0.1, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ....................C=0.1, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ....................C=0.1, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=0.1, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .....................C=0.1, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .....................C=0.1, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .....................C=0.1, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .....................C=0.1, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ..........................C=1, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END ..........................C=1, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END ..........................C=1, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END ..........................C=1, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END ..........................C=1, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=1, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=1, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=1, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=1, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=1, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .....................C=1, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END .....................C=1, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END .....................C=1, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END .....................C=1, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END .....................C=1, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END .......................C=1, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .......................C=1, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .......................C=1, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .......................C=1, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .......................C=1, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ....................C=1, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=1, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=1, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=1, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=1, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=1, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=1, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=1, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=1, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=1, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END .........................C=10, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END .........................C=10, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END .........................C=10, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END .........................C=10, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END .........................C=10, gamma=1, kernel=poly; total time=   0.0s\n",
            "[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ...................C=10, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ...................C=10, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ...................C=10, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ...................C=10, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ...................C=10, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.0s\n",
            "[CV] END ........................C=100, gamma=1, kernel=poly; total time=   0.3s\n",
            "[CV] END ........................C=100, gamma=1, kernel=poly; total time=   0.3s\n",
            "[CV] END ........................C=100, gamma=1, kernel=poly; total time=   0.2s\n",
            "[CV] END ........................C=100, gamma=1, kernel=poly; total time=   0.3s\n",
            "[CV] END ........................C=100, gamma=1, kernel=poly; total time=   0.2s\n",
            "[CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=0.1, kernel=poly; total time=   0.0s\n",
            "[CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s\n",
            "[CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.0s\n",
            "[CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s\n",
            "[CV] END ..................C=100, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ..................C=100, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ..................C=100, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ..................C=100, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ..................C=100, gamma=0.001, kernel=linear; total time=   0.0s\n",
            "[CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s\n",
            "[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s\n",
            "Best Parameters: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 6: Model Evaluation\n",
        "y_pred = best_model.predict(X_test)\n",
        "print(\"Classification Report:\")\n",
        "print(classification_report(y_test, y_pred))\n",
        "print(\"Confusion Matrix:\")\n",
        "print(confusion_matrix(y_test, y_pred))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "j5bvSLxo73WC",
        "outputId": "45ef9064-9d76-42af-dd68-61c18ec73cd0"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Classification Report:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.96      0.95      0.95        73\n",
            "           1       0.92      0.94      0.93        47\n",
            "\n",
            "    accuracy                           0.94       120\n",
            "   macro avg       0.94      0.94      0.94       120\n",
            "weighted avg       0.94      0.94      0.94       120\n",
            "\n",
            "Confusion Matrix:\n",
            "[[69  4]\n",
            " [ 3 44]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import joblib\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "# Save the scaler\n",
        "scaler = StandardScaler()\n",
        "X_scaled = scaler.fit_transform(X)  # Fit on training data\n",
        "\n",
        "# Save both model and scaler\n",
        "joblib.dump(scaler, 'scaler.pkl')\n",
        "joblib.dump(best_model, 'svm_best_model.pkl')\n",
        "print(\"Model and scaler saved successfully.\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3qgqwCUr8DtL",
        "outputId": "f83d354e-d4c0-4450-9367-ac6184cdea40"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model and scaler saved successfully.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "files.download('svm_best_model.pkl')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 17
        },
        "id": "DkE4m4JGLBPj",
        "outputId": "b94bf061-beb7-4f03-b8db-03d25fa972dd"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "    async function download(id, filename, size) {\n",
              "      if (!google.colab.kernel.accessAllowed) {\n",
              "        return;\n",
              "      }\n",
              "      const div = document.createElement('div');\n",
              "      const label = document.createElement('label');\n",
              "      label.textContent = `Downloading \"${filename}\": `;\n",
              "      div.appendChild(label);\n",
              "      const progress = document.createElement('progress');\n",
              "      progress.max = size;\n",
              "      div.appendChild(progress);\n",
              "      document.body.appendChild(div);\n",
              "\n",
              "      const buffers = [];\n",
              "      let downloaded = 0;\n",
              "\n",
              "      const channel = await google.colab.kernel.comms.open(id);\n",
              "      // Send a message to notify the kernel that we're ready.\n",
              "      channel.send({})\n",
              "\n",
              "      for await (const message of channel.messages) {\n",
              "        // Send a message to notify the kernel that we're ready.\n",
              "        channel.send({})\n",
              "        if (message.buffers) {\n",
              "          for (const buffer of message.buffers) {\n",
              "            buffers.push(buffer);\n",
              "            downloaded += buffer.byteLength;\n",
              "            progress.value = downloaded;\n",
              "          }\n",
              "        }\n",
              "      }\n",
              "      const blob = new Blob(buffers, {type: 'application/binary'});\n",
              "      const a = document.createElement('a');\n",
              "      a.href = window.URL.createObjectURL(blob);\n",
              "      a.download = filename;\n",
              "      div.appendChild(a);\n",
              "      a.click();\n",
              "      div.remove();\n",
              "    }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "download(\"download_563ebbab-673f-47f7-976c-16a252dcab1d\", \"svm_best_model.pkl\", 5339)"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install streamlit"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "J_14b7STLGnw",
        "outputId": "4384d759-eb99-4891-d6ac-7f8284c78a83"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.11/dist-packages (1.41.1)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.5.1)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (8.1.8)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.11/dist-packages (from streamlit) (1.26.4)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.11/dist-packages (from streamlit) (24.2)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (11.1.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.11/dist-packages (from streamlit) (4.25.5)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (17.0.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (13.9.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (9.0.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.11/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (4.12.2)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.11/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.11/dist-packages (from streamlit) (3.1.44)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.11/dist-packages (from streamlit) (0.9.1)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.11/dist-packages (from streamlit) (6.3.3)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (3.1.5)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (1.23.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.11/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (3.4.1)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (2.3.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (2024.12.14)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich<14,>=10.14.0->streamlit) (2.18.0)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.11/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (24.3.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2024.10.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.36.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.22.3)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "import numpy as np\n",
        "import joblib\n",
        "\n",
        "# Load the saved model and scaler\n",
        "model = joblib.load('svm_best_model.pkl')\n",
        "scaler = joblib.load('scaler.pkl')  # Load the saved scaler\n",
        "\n",
        "# Streamlit app\n",
        "st.title(\"Purchase Prediction Using SVM\")\n",
        "st.write(\"This app predicts whether a user will purchase a product based on their details.\")\n",
        "\n",
        "# Input fields\n",
        "gender = st.radio(\"Select Gender\", (\"Male\", \"Female\"))\n",
        "age = st.number_input(\"Enter Age\", min_value=0, max_value=100, value=25)\n",
        "estimated_salary = st.number_input(\"Enter Estimated Salary\", min_value=0.0, value=50000.0, step=1000.0)\n",
        "\n",
        "# Preprocess input\n",
        "gender_numeric = 1 if gender == \"Male\" else 0\n",
        "user_input = np.array([[gender_numeric, age, estimated_salary]])\n",
        "user_input_scaled = scaler.transform(user_input)\n",
        "\n",
        "# Predict and display results\n",
        "if st.button(\"Predict\"):\n",
        "    prediction = model.predict(user_input_scaled)\n",
        "    prediction_label = \"Purchased\" if prediction[0] == 1 else \"Not Purchased\"\n",
        "    st.write(f\"The predicted outcome is: **{prediction_label}**\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gs-YC0WTLPg3",
        "outputId": "d9e243b2-342a-405c-a908-39b855f92dbb"
      },
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-01-28 13:55:01.527 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.615 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2025-01-28 13:55:01.617 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.621 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.623 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.627 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.628 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.630 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.633 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.635 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.637 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.639 Session state does not function when running a script without `streamlit run`\n",
            "2025-01-28 13:55:01.641 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.642 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.643 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.644 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.645 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.646 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.647 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.649 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.650 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.651 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.652 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.653 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.654 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.655 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.657 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.660 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.661 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.663 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-01-28 13:55:01.664 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3 (ipykernel)",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.9.21"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}