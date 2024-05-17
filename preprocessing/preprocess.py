################################################################################
#                                                                              #
#                         Author: Bc. Petr Pouč                                #
#                         Date: April 4, 2024                                  #
#                         School: Brno University of Technology (BUT)          #
#                                                                              #
#         Master's Thesis: Optimization of Classification Models               #
#                         for Malicious Domain Detection                       #
#                                                                              #
################################################################################


# Standard library imports
import os
import datetime
import logging
import pickle
import joblib
import warnings
from concurrent.futures import ThreadPoolExecutor

# Third-party imports for data handling and computation
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import Table
from pandas.core.dtypes import common as com
from pandas import DataFrame

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning and feature selection libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
#import cross_val_score from sklearn.model_selection
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import shap
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Other utilities
import click
from colorama import init, Fore, Style
from tabulate import tabulate
import torch

# import hash_countries from geo_mapping.py
from mapping import country_ids, continent_ids
from category_encoders import BinaryEncoder


# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.api.types")
warnings.filterwarnings("ignore", message="is_sparse is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="is_categorical_dtype is deprecated", category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)



init(autoreset=True)

class FeatureEngineeringCLI:
    def __init__(self, input_data, one_line_processing: bool):
        self.logger = self.configure_logger()
        self.one_line_processing = one_line_processing
        if self.one_line_processing:
            self.single_record_df = input_data
        else:
            self.benign_path = input_data.get('benign')
            self.malign_path = input_data.get('malign')
            self.single_record_df = None

        self.DEFAULT_INPUT_DIR = ""
        self.nontraining_fields = [
            "dns_evaluated_on",
            "rdap_evaluated_on",
            "tls_evaluated_on",
            "ip_data",
            "countries",
            "latitudes",
            "longitudes",
            "dns_dnssec",
            "dns_zone_dnskey_selfsign_ok",
            "dns_email_extras",
            "dns_ttls",
            "dns_zone",
            "dns_zone_SOA",
            *[f"dns_{t}" for t in ('A', 'AAAA', 'CNAME', 'MX', 'NS', 'SOA', 'TXT')],
            "rdap_registration_date",
            "rdap_last_changed_date",
            "rdap_expiration_date",
            "rdap_dnssec",
            "rdap_entities"
        ]
        self.scaler_path = 'scaler.joblib'
        self.outliers_path = 'outliers.joblib'
        self.scaler = None
        self.outliers = None
        self.scaler_saved = False
        self.outliers_saved = False
        self.DEFAULT_INPUT_DIR = ""
        self.borders_dir = "trained_borders"
        self.scaler_path = os.path.join(self.borders_dir, self.scaler_path)
        self.outliers_path = os.path.join(self.borders_dir, self.outliers_path)

        self.model_path = os.path.join(self.borders_dir, 'decision_tree_model.joblib')
        self.model = None


    def print_header(self, message: str) -> None:
        header = f"{'=' * len(message)}"
        self.logger.info(self.color_log(header, Fore.CYAN))
        self.logger.info(self.color_log(message, Fore.CYAN))
        self.logger.info(self.color_log(header, Fore.CYAN))


    def configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler('NDF.log')
            file_handler.setLevel(logging.INFO)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger


    def save_borders(self):
        """Save both scaler and outlier thresholds to the specified directory."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)

        if self.scaler is not None and not self.scaler_saved:
            joblib.dump(self.scaler, self.scaler_path)
            self.logger.info(self.color_log(f"Scaler saved to {self.scaler_path}", Fore.GREEN))
            self.scaler_saved = True  
        
        if self.outliers is not None and not self.outliers_saved:
            os.makedirs(os.path.dirname(self.outliers_path), exist_ok=True)  # Ensure directory for outliers exists
            joblib.dump(self.outliers, self.outliers_path)
            self.logger.info(self.color_log(f"Outliers thresholds saved to {self.outliers_path}", Fore.GREEN))
            self.outliers_saved = True  


    def load_borders(self):
        """Load both scaler and outlier thresholds from the specified directory."""
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            self.logger.info(self.color_log(f"Scaler loaded from {self.scaler_path}", Fore.GREEN))
        
        if os.path.exists(self.outliers_path):
            self.outliers = joblib.load(self.outliers_path)
            self.logger.info(self.color_log(f"Outliers thresholds loaded from {self.outliers_path}", Fore.GREEN))


    def save_model(self):
        """Save the decision tree model to the specified directory."""
        if self.model is not None:
            if not os.path.exists(self.model_path):
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            self.logger.info(self.color_log(f"Decision tree model saved to {self.model_path}", Fore.GREEN))


    def load_model(self):
        """Load the decision tree model from the specified directory."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.logger.info(self.color_log(f"Decision tree model loaded from {self.model_path}", Fore.GREEN))


    def color_log(self, message: str, color: str = Fore.GREEN) -> str:
        return f"{color}{message}{Style.RESET_ALL}"
    

    def drop_nontrain(self, table: Table) -> Table:
        """
        Drop non-training columns.
        """
        fields = [x for x in self.nontraining_fields if x in table.column_names]
        return table.drop(fields)
    
    def scaler_recommendation(self, df: pd.DataFrame) -> dict:
        """
        Recommend scalers for SVM, XGBoost, and CNN based on the dataset characteristics.

        Args:
        df (pd.DataFrame): The dataset after EDA.

        Returns:
        dict: Dictionary containing scaler recommendations for SVM, XGBoost, and CNN.
        """
        recommendations = {}

        # Check for outliers using Z-score
        numeric_df = df.select_dtypes(include=[np.number])
        outliers = np.any(np.abs(zscore(numeric_df)) > 3, axis=1)
        outlier_proportion = np.mean(outliers)

        # Check for missing values
        missing_values = df.isnull().any().sum()

        # Recommendations for SVM
        if outlier_proportion > 0.05 or missing_values > 0:
            recommendations['svm'] = 'RobustScaler'
        else:
            recommendations['svm'] = 'StandardScaler'

        # Recommendations for XGBoost
        # XGBoost is less sensitive to the scale of data
        recommendations['xgboost'] = 'MinMaxScaler'

        # Recommendations for CNN
        #using the sigmoid function to map values from an arbitrary range to the range [0, 1]
        recommendations['cnn'] = 'MinMaxScaler + Sigmoid'

        return recommendations
    

    def apply_scaling(self, df: pd.DataFrame, scaler_type: str = 'StandardScaler'):
        """
        Apply scaling to the dataset based on the scaler type.
        Returns a new DataFrame with scaled data.
        """
        numeric_df = df.select_dtypes(include=[np.number])

        # Branch logic based on one_line_processing
        if not self.one_line_processing:
            if scaler_type == 'StandardScaler':
                self.scaler = StandardScaler()
                scaled_data = self.scaler.fit_transform(numeric_df)
            elif scaler_type == 'MinMaxScaler':
                self.scaler = MinMaxScaler()
                scaled_data = self.scaler.fit_transform(numeric_df)
            elif scaler_type == 'RobustScaler':
                self.scaler = RobustScaler()
                scaled_data = self.scaler.fit_transform(numeric_df)
            elif scaler_type == 'MinMaxScaler + Sigmoid':
                self.scaler = MinMaxScaler()
                scaled_data = self.scaler.fit_transform(numeric_df)
                # Apply sigmoid scaling
                scaled_data = 1 / (1 + np.exp(-scaled_data))
            else:
                raise ValueError(f"Unsupported scaler type: {scaler_type}")
            self.save_borders()  # Save the scaler for future use
        else:
            self.load_borders()  # Load the previously saved scaler
            scaled_data = self.scaler.transform(numeric_df)

        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns, index=df.index)

        # Combine scaled numeric columns with non-numeric data
        for col in df.columns:
            if col not in numeric_df.columns:
                scaled_df[col] = df[col]

        return scaled_df


    def get_feature_with_highest_shap(self, shap_values: np.ndarray, dataset: pd.DataFrame, sample_index: int) -> tuple:
        """
        Returns the feature with the highest SHAP value for a sample.
        """
        abs_shap_values = np.abs(shap_values[sample_index, :])
        highest_shap_index = np.argmax(abs_shap_values)

        # Get the corresponding feature name and value from the dataset
        feature_name = dataset.columns[highest_shap_index]
        feature_value = dataset.iloc[sample_index, highest_shap_index]

        return feature_name, feature_value
    

    def reverse_map_continent(self, continent_id):
        for name, id in continent_ids.items():
            if id == continent_id:
                return name
        return "Unknown"

    def reverse_map_country(self, country_id):
        for name, id in country_ids.items():
            if id == country_id:
                return name
        return "Unknown"
        

    def categorical_encoding_geo(self, df: DataFrame) -> DataFrame:
        # Handling geographical features: geo_continent_hash, geo_countries_hash
        
        if 'geo_continent_hash' in df.columns:
            df['geo_continent'] = df['geo_continent_hash'].apply(self.reverse_map_continent)
            df.drop('geo_continent_hash', axis=1, inplace=True)
        
        if 'geo_countries_hash' in df.columns:
            df['geo_countries'] = df['geo_countries_hash'].apply(self.reverse_map_country)
            df.drop('geo_countries_hash', axis=1, inplace=True)

        # One-hot encoding for geographical features
        features_to_encode = ['geo_continent', 'geo_countries']
        existing_features = [feature for feature in features_to_encode if feature in df.columns]
        
        if existing_features:
            for feature in existing_features:
                encoded_features = pd.get_dummies(df[feature], prefix=feature, drop_first=False)
                df.drop(feature, axis=1, inplace=True)
                df = pd.concat([df, encoded_features], axis=1)
                self.logger.info(self.color_log(f"Applied one-hot encoding to feature: {feature}", Fore.GREEN))

        return df

    def categorical_encoding_lex(self, df: DataFrame) -> DataFrame:
        # Handling lexical features: tld_hash
        
        if 'lex_tld_hash' in df.columns:
            binary_encoder = BinaryEncoder(cols=['lex_tld_hash'])
            df = binary_encoder.fit_transform(df)
            self.logger.info(self.color_log("Applied binary encoding to feature: lex_tld_hash", Fore.GREEN))

        return df

    def categorical_encoding_tls_rdap(self, df: DataFrame) -> DataFrame:
        # Handling TLS/RDAP features: registrar_name_hash, root_authority_hash, leaf_authority_hash
        
        # Drop features without encoding, as they are not directly encoded but instead removed or handled differently.
        if 'rdap_registrar_name_hash' in df.columns:
            df.drop('rdap_registrar_name_hash', axis=1, inplace=True)
            self.logger.info(self.color_log("Dropped feature: rdap_registrar_name_hash", Fore.GREEN))
        
        if 'tls_root_authority_hash' in df.columns:
            df.drop('tls_root_authority_hash', axis=1, inplace=True)
            self.logger.info(self.color_log("Dropped feature: tls_root_authority_hash", Fore.GREEN))
        
        if 'tls_leaf_authority_hash' in df.columns:
            df.drop('tls_leaf_authority_hash', axis=1, inplace=True)
            self.logger.info(self.color_log("Dropped feature: tls_leaf_authority_hash", Fore.GREEN))

        return df
    

    def cast_timestamp(df: DataFrame):
        for col in df.columns:
            if com.is_timedelta64_dtype(df[col]):
                df[col] = df[col].dt.total_seconds()  # This converts timedelta to float (seconds)
            elif com.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(np.int64) // 10**9  # Converts datetime64 to Unix timestamp (seconds)

        return df
    
    def remove_outliers(self, features, labels, std_multiplier=8):
        if not self.one_line_processing:
            # Directly update self.outliers with new thresholds
            self.outliers = {column: (mean_val - std_multiplier * std_val, mean_val + std_multiplier * std_val)
                            for column in features.select_dtypes(include=[np.number]).columns
                            for mean_val, std_val in [(features[column].mean(), features[column].std())]}
            self.save_borders()  # This now also includes saving the newly computed outlier thresholds
        else:
            self.load_borders()  # Load existing borders, including outlier thresholds

        if not self.outliers:
            raise ValueError("Outlier thresholds must be available for outlier removal.")

        # Apply loaded or newly computed thresholds to remove outliers
        for column, (lower_bound, upper_bound) in self.outliers.items():
            if column in features.columns:  # Ensure the column exists in the current dataset
                initial_row_count = len(features)  # Store initial number of rows

                outlier_condition = (features[column] < lower_bound) | (features[column] > upper_bound)
                outlier_indices = features[outlier_condition].index

                # Remove outliers from features and labels
                features.drop(outlier_indices, inplace=True)
                labels.drop(outlier_indices, inplace=True)

                removed_count = initial_row_count - len(features)  # Calculate number of rows removed
                if removed_count > 0:  # Only log if any rows were removed
                    self.logger.info(f"Outliers removed from {column}: {self.color_log(removed_count, Fore.RED)} rows")

        self.logger.info("Completed outlier removal.")
        return features, labels


    def perform_eda(self, model=None, apply_scaling=False) -> None:
            # Load the data
            if self.one_line_processing:
                # For single-record processing, use the provided DataFrame
                combined_df = self.single_record_df
                self.logger.info(self.color_log(f"Single-record processing: {combined_df.shape[0]} rows", Fore.GREEN))
            else:
                benign_path = os.path.join(self.DEFAULT_INPUT_DIR, self.benign_path) if self.benign_path else None
                malign_path = os.path.join(self.DEFAULT_INPUT_DIR, self.malign_path) if self.malign_path else None
                
                self.logger.info(self.color_log(f'Benign dataset path: {benign_path}', Fore.GREEN))
                self.logger.info(self.color_log(f'Malign dataset path: {malign_path}', Fore.GREEN))

                print(f'Malign dataset path: {malign_path}')
                print(f'Benign dataset path: {benign_path}')
                # For full-dataset processing, load and combine benign and malign datasets
                benign_data = pq.read_table(os.path.join(self.DEFAULT_INPUT_DIR, self.benign_path)) if self.benign_path else None
                malign_data = pq.read_table(os.path.join(self.DEFAULT_INPUT_DIR, self.malign_path)) if self.malign_path else None


                # Drop non-training columns and realign schemas
                benign_data = self.drop_nontrain(benign_data)
                malign_data = self.drop_nontrain(malign_data)
                benign_data = benign_data.cast(malign_data.schema)


                #print number of records in benign and malign datasets
                benign_records = len(benign_data)
                malign_records = len(malign_data)
                self.logger.info(self.color_log(f"Number of records in benign dataset: {benign_records}", Fore.GREEN))
                self.logger.info(self.color_log(f"Number of records in malign dataset: {malign_records}", Fore.GREEN))

                #get missing values distribution for benign and malign datasets for each feature, save this distribution to txt file
                benign_missing = benign_data.to_pandas().isnull().sum()
                malign_missing = malign_data.to_pandas().isnull().sum()
                missing_values = pd.concat([benign_missing, malign_missing], axis=1)

                # Calculate percentage of missing values
                total_records = len(benign_data) + len(malign_data)
                missing_values['percent_missing_benign'] = (missing_values.iloc[:, 0] / len(benign_data)) * 100
                missing_values['percent_missing_malign'] = (missing_values.iloc[:, 1] / len(malign_data)) * 100

                # Sort them by biggest difference
                missing_values['difference_in_percent'] = abs(missing_values['percent_missing_benign'] - missing_values['percent_missing_malign'])
                missing_values = missing_values.sort_values(by='difference_in_percent', ascending=False)
                missing_values.to_csv('missing_values.csv')

                #print total percentage of missing values in benign and malign datasets
                benign_total_missing = (missing_values['percent_missing_benign'].sum() / total_records) * 100
                malign_total_missing = (missing_values['percent_missing_malign'].sum() / total_records) * 100
                self.logger.info(self.color_log(f"Total percentage of missing values in benign dataset: {benign_total_missing:.2f}%", Fore.GREEN))
                self.logger.info(self.color_log(f"Total percentage of missing values in malign dataset: {malign_total_missing:.2f}%", Fore.GREEN))
                

                # Concatenate tables and convert to pandas DataFrame
                combined_data = pa.concat_tables([benign_data, malign_data])
                combined_df = combined_data.to_pandas()

            # randomly shuffle the records
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)

            # Categorical Encoding
            # combined_df = self.categorical_encoding_lex(combined_df)

            # Extract labels
            if 'label' in combined_df.columns:
                labels = combined_df['label']
            else:
                raise ValueError("Label column not found in the dataframe.")

            categorical_features = ['geo_continent_hash', 'geo_countries_hash', 'rdap_registrar_name_hash', 'tls_root_authority_hash', 'tls_leaf_authority_hash']

            #creating new feature as probability from decision tree predictions trained on categorical features
            if not self.one_line_processing:
                X_categorical = combined_df[categorical_features]
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ],
                    remainder='passthrough'
                )
                # Full-dataset processing logic, including training of models, remains here

                # Split the dataset into training and testing sets with stratification to maintain label distribution
                X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
                    X_categorical, labels, range(X_categorical.shape[0]), test_size=0.2, random_state=42, stratify=labels)

                dt_classifier = DecisionTreeClassifier(random_state=42)
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', dt_classifier)
                ])

                # Train the pipeline on the training set
                dt_pipeline = pipeline.fit(X_train, y_train)
                self.model = dt_pipeline
                self.save_model()  

                # Predict probabilities for both training and testing sets
                probabilities_train = pipeline.predict_proba(X_train)[:, 1]  # Probability of class 1
                probabilities_test = pipeline.predict_proba(X_test)[:, 1]

                combined_df['dtree_prob'] = np.nan

                # Assign probabilities to their respective rows in combined_df
                combined_df.loc[X_train.index, 'dtree_prob'] = probabilities_train
                combined_df.loc[X_test.index, 'dtree_prob'] = probabilities_test

                # Logging the creation of the new feature and model performance
                self.logger.info("New feature 'dtree_prob' created from decision tree predictions.")
                train_accuracy = pipeline.score(X_train, y_train)
                test_accuracy = pipeline.score(X_test, y_test)
                self.logger.info(f"Decision Tree Train Accuracy: {train_accuracy:.2f}")
                self.logger.info(f"Decision Tree Test Accuracy: {test_accuracy:.2f}")

                # Perform cross-validation for a more robust estimate of model performance
                scores = cross_val_score(pipeline, combined_df[categorical_features], labels, cv=3)
                self.logger.info(f"Decision Tree Cross-Validation Scores: {scores}")
            else:
                # Single-record processing logic
                self.load_model()  # Load the entire pipeline
                if self.model:
                    # Use the loaded pipeline to predict the probability for the single record
                    probabilities_single_record = self.model.predict_proba(self.single_record_df[categorical_features])[:, 1]
                    combined_df['dtree_prob'] = probabilities_single_record[0]  # Assuming single record, extract the first probability
                    self.logger.info("Applied loaded decision tree pipeline to generate 'dtree_prob' for the single record.")
                else:
                    raise ValueError("Pipeline not found. Please train the pipeline first.")


            unique_labels = combined_df['label'].unique()
            class_map = {}
            for label in unique_labels:
                if label.startswith("benign"):
                    class_map[label] = 0
                elif label.startswith("malware"):
                    class_map[label] = 1
                elif label.startswith("misp") and "phishing" in label:
                    class_map[label] = 1 
    
            self.logger.info(self.color_log(f"Generated class map: {class_map}", Fore.GREEN))

            # Separate labels and features
            labels = combined_df['label'].apply(lambda x: class_map.get(x, -1))  # -1 for any label not in class_map
            features = combined_df.drop('label', axis=1).copy()

            # Process timestamps
            for col in features.columns:
                if com.is_timedelta64_dtype(features[col]):
                    features[col] = features[col].dt.total_seconds()
                elif com.is_datetime64_any_dtype(features[col]):
                    features[col] = features[col].astype(np.int64) // 10**9

            # Convert bool columns to float
            for column in features.columns:
                if features[column].dtype == 'bool':
                    features[column] = features[column].astype('float64')

            features = features.drop(features.columns[0], axis=1)

            # Handling missing values in features
            features.fillna(-1, inplace=True)

            features, labels = self.remove_outliers(features, labels, std_multiplier=8)


            # Apply scaling if requested
            if apply_scaling:
                scaler_recommendations = self.scaler_recommendation(features)
                scaler_type = scaler_recommendations.get(model.lower(), 'StandardScaler')
                self.logger.info(self.color_log(f"Applying {scaler_type} scaling to the features.", Fore.YELLOW))
                features = self.apply_scaling(features, scaler_type)
                self.logger.info(self.color_log("Scaling applied to the features\n", Fore.GREEN))

            # Save the modified dataset as a Parquet file
            modified_data = pa.Table.from_pandas(features)
            output_path = os.path.join(self.DEFAULT_INPUT_DIR, 'modified_dataset.parquet')
            feature_names = features.columns

            pq.write_table(modified_data, output_path)
            self.logger.info(self.color_log(f"Modified combined dataset saved to {output_path}", Fore.GREEN))

            self.logger.info(self.color_log("Head of modified combined dataset:", Fore.YELLOW))
            self.logger.info(features)


            return torch.tensor(features.values).float(), torch.tensor(labels.values).float(), feature_names


def choose_dataset(files, dataset_type):
    print(f"Choose {dataset_type.upper()} dataset:")
    print("-" * 22)
    for idx, file in enumerate(files, start=1):
        print(Fore.GREEN + f"[{idx}]: {file}")

    choice = int(input(Fore.YELLOW + f"Enter the number corresponding to the {dataset_type} dataset: "))
    return files[choice - 1]

def display_dataset_subset(x_train, y_train, dataset_name, dimension, subset_size=10):
    subset_features = pd.DataFrame(x_train[:subset_size].numpy(), columns=[f"Feature_{i}" for i in range(x_train.shape[1])])
    subset_labels = pd.DataFrame(y_train[:subset_size].numpy(), columns=['Label'])

    print("\nDataset Subset:")
    print(f"Name: {dataset_name}")
    print("Features:")
    print(subset_features)
    print("Labels:")
    print(subset_labels)
    print("Dimension:", dimension)


def NDF(model: str, scaling: bool, input_data, one_line_processing: bool):
    fe_cli = FeatureEngineeringCLI(input_data=input_data, one_line_processing=one_line_processing)

    features, labels, feature_names = fe_cli.perform_eda(model, scaling)
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if one_line_processing:
        dataset_name = f"single_record_dataset_{current_date}"
    else:
        # For full dataset processing, extract the dataset names from the input paths.
        benign_name = ''.join(input_data['benign'].split('_')[:2])
        malign_name = ''.join(input_data['malign'].split('_')[:2])
        dataset_name = f"dataset_{benign_name}_{malign_name}_{current_date}"
        dataset_name = dataset_name.replace('.parquet', '') + '.parquet'

    dataset = {
        'name': dataset_name,
        'features': features,
        'labels': labels,
        'dimension': features.shape[1],
        'feature_names': feature_names,
        'one_line_processing': one_line_processing  # Include this flag in your dataset dictionary
    }        

    directory = 'datasets'
    file_path = os.path.join(directory, 'preprocessed_dataset.pickle')
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)

    # Adjust the split to only proceed if not one_line_processing
    if not one_line_processing:
        x_train, x_test, y_train, y_test = train_test_split(
            dataset['features'],
            dataset['labels'],
            test_size=0.2,
            random_state=42
        )
        display_dataset_subset(x_train, y_train, dataset['name'], dataset['dimension'])
    else:
        # For single record processing, simply set x_train and y_train to the entire dataset
        x_train, y_train = dataset['features'], dataset['labels']
        # No x_test, y_test available here
        display_dataset_subset(x_train, y_train, dataset['name'], dataset['dimension'], subset_size=1)

    return dataset

if __name__ == '__main__':
    input_data = {
    'benign': '../floor/benign_2312.parquet',
    'malign': '../floor/phishing_2311.parquet'
    }
    dataset = NDF("cnn", True, input_data=input_data, one_line_processing=False)