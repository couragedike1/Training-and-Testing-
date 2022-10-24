{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMueAU7TUWquDTrI0OWI8JZ",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/couragedike1/Training-and-Testing-/blob/main/Musictrain.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tDyYikTNIx_t",
        "outputId": "1e3719bd-0e44-451e-e306-b3e49d202811"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1.0"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ],
      "source": [
        "import pandas as pd\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import  accuracy_score\n",
        "\n",
        "df = pd.read_csv('music.csv')\n",
        "X = df.drop(columns=['genre'])\n",
        "y = df['genre']\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)\n",
        "\n",
        "model = DecisionTreeClassifier()\n",
        "model.fit(X_train,y_train)\n",
        "predictions = model.predict(X_test)\n",
        "score = accuracy_score(y_test, predictions)\n",
        "score "
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn import tree\n",
        "\n",
        "\n",
        "df = pd.read_csv('music.csv')\n",
        "X = df.drop(columns=['genre'])\n",
        "y = df['genre']\n",
        "\n",
        "model = DecisionTreeClassifier()\n",
        "model.fit(X,y)\n",
        "\n",
        "tree.export_graphviz(model, out_file = 'music.dot', feature_names = ['age', 'gender'], class_names = sorted (y.unique ()), label ='all', rounded = True, filled = True )\n"
      ],
      "metadata": {
        "id": "t2adq9PKbLPc"
      },
      "execution_count": 7,
      "outputs": []
    }
  ]
}