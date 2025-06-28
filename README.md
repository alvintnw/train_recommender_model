Movie Recommender Model Training
This project demonstrates how to train a movie recommender system using the SVD (Singular Value Decomposition) algorithm from the Surprise library. The trained model is then saved for future use, such as deployment in an recommendation API.

Code Overview
The Google Colab notebook train_recommender_model.ipynb performs the following key steps:

Install Necessary Libraries:

The notebook includes !pip install scikit-surprise to ensure the Surprise library is installed.

It also includes !pip install numpy==1.26.4, likely to address potential compatibility issues with Surprise or other libraries.

Load Dataset:

Loads movie ratings data from ratings.csv and movie metadata from movies.csv.

These CSV files are expected to be present in the same directory as the notebook.

Prepare Data for Surprise Library:

A Reader object is initialized with a rating_scale of (0.5, 5.0) to match the expected rating range of the MovieLens dataset.

The ratings_df (specifically userId, movieId, and rating columns) is loaded into a Surprise Dataset object.

Train SVD Model:

An SVD model (surprise.SVD) is instantiated with n_epochs=20, n_factors=50, and a random_state=42 for reproducibility. verbose=True is set to display training progress.

The model is trained on the full dataset using data.build_full_trainset() and algo.fit(trainset).

Although commented out, the code shows an optional step to perform 5-fold cross-validation (cross_validate) to evaluate the model's performance using metrics like RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error). This step is typically for evaluation and not strictly necessary if the goal is only to save a fully trained model.

Save Trained Model:

The trained SVD model (algo) is saved to a file named svd_recommender_model.pkl using Python's pickle module. This allows the model to be loaded and used later without retraining.

Additionally, the movies_df is saved as movies_data.csv. This CSV file can be used to map movieId values back to actual movie titles when making recommendations.

Dataset Structure (MovieLens)
This project uses a simplified MovieLens dataset, typically consisting of two main CSV files:

ratings.csv: Contains user ratings for movies.
| userId | movieId | rating | timestamp |
| :----- | :------ | :----- | :-------- |
| 1      | 1       | 4.0    | 964988270 |
| 1      | 3       | 4.0    | 964988270 |
| ...    | ...     | ...    | ...       |

movies.csv: Contains movie metadata.
| movieId | title             | genres                   |
| :------ | :---------------- | :----------------------- |
| 1       | Toy Story (1995)  | Adventure|Animation|Children|Comedy|Fantasy |
| 2       | Jumanji (1995)    | Adventure|Children|Fantasy |
| ...     | ...               | ...                      |

Libraries Used
pandas

surprise

numpy

pickle

How to Use (in Google Colab)
Upload ratings.csv and movies.csv to your Google Colab environment.

Open the train_recommender_model.ipynb file in Google Colab.

Run all cells in the notebook.

After execution, the trained model (svd_recommender_model.pkl) and the movie data (movies_data.csv) will be saved in your Colab session's file system. You can then download these files for deployment or further use.
