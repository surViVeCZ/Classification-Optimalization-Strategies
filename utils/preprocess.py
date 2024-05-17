# Standard library imports
import os
import datetime
import logging
import pickle
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
import shap
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Other utilities
import click
from colorama import init, Fore, Style
from tabulate import tabulate
import torch

# import hash_countries from geo_mapping.py
from utils.mapping import country_ids, continent_ids
from category_encoders import BinaryEncoder

#scaler saving
from joblib import dump, load


# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.api.types")
warnings.filterwarnings("ignore", message="is_sparse is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="is_categorical_dtype is deprecated", category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)



init(autoreset=True)

class FeatureEngineeringCLI:
    def __init__(self, benign_path: str, malign_path: str):
        self.benign_path = benign_path
        self.malign_path = malign_path
        self.logger = self.configure_logger()
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
    def print_header(self, message: str) -> None:
        header = f"{'=' * len(message)}"
        self.logger.info(self.color_log(header, Fore.CYAN))
        self.logger.info(self.color_log(message, Fore.CYAN))
        self.logger.info(self.color_log(header, Fore.CYAN))




    def configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Only add handlers if they don't exist to prevent duplication
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
        # Assuming the data is not image data as it's not the typical use case for EDA
        
        #using the sigmoid function to map values from an arbitrary range to the range [0, 1]
        recommendations['cnn'] = 'MinMaxScaler_Sigmoid'

        return recommendations
    

    def apply_scaling(self, df: pd.DataFrame, scaler_type: str) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=[np.number])

        if scaler_type == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaler_type == 'RobustScaler':
            scaler = RobustScaler()
        elif scaler_type == 'MinMaxScaler_Sigmoid':
            scaler = MinMaxScaler()
            # Note: For 'MinMaxScaler + Sigmoid', additional logic for sigmoid scaling is needed after this block
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

        # Fit and transform the data
        scaled_data = scaler.fit_transform(numeric_df)

        # Save the scaler object to a file
        scaler_filename = f"{scaler_type}_scaler.joblib"
        dump(scaler, scaler_filename)
        self.logger.info(f"Scaler of type '{scaler_type}' saved to {scaler_filename}")

        # Update the DataFrame with scaled data
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns, index=df.index)

        # Combine scaled numeric columns with non-numeric data
        for col in df.columns:
            if col not in numeric_df.columns:
                scaled_df[col] = df[col]

        return scaled_df



    def get_feature_with_highest_shap(self, shap_values: np.ndarray, dataset: pd.DataFrame, sample_index: int) -> tuple:
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

    def perform_eda(self, model=None, apply_scaling=False) -> None:
            benign_path = os.path.join(self.DEFAULT_INPUT_DIR, self.benign_path) if self.benign_path else None
            malign_path = os.path.join(self.DEFAULT_INPUT_DIR, self.malign_path) if self.malign_path else None
            
            self.logger.info(self.color_log(f'Benign dataset path: {benign_path}', Fore.GREEN))
            self.logger.info(self.color_log(f'Malign dataset path: {malign_path}', Fore.GREEN))

            print(f'Malign dataset path: {malign_path}')
            print(f'Benign dataset path: {benign_path}')
            # Load the data
            benign_data = pq.read_table(benign_path)
            malign_data = pq.read_table(malign_path)

            # Drop non-training columns and realign schemas
            benign_data = self.drop_nontrain(benign_data)
            malign_data = self.drop_nontrain(malign_data)
            benign_data = benign_data.cast(malign_data.schema)

            # Concatenate tables and convert to pandas DataFrame
            combined_data = pa.concat_tables([benign_data, malign_data])
            combined_df = combined_data.to_pandas()

            # randomly shuffle the records
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)

          # Categorical Encoding
            combined_df = self.categorical_encoding_lex(combined_df)

            # Extract labels
            if 'label' in combined_df.columns:
                labels = combined_df['label']
            else:
                raise ValueError("Label column not found in the dataframe.")

            categorical_features = ['geo_continent_hash', 'geo_countries_hash', 'rdap_registrar_name_hash', 'tls_root_authority_hash', 'tls_leaf_authority_hash']
            X_categorical = combined_df[categorical_features]

            # Split the dataset into training and testing sets with stratification to maintain label distribution
            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
                X_categorical, labels, range(X_categorical.shape[0]), test_size=0.2, random_state=42, stratify=labels)

            # Train the decision tree on the training set
            decision_tree = DecisionTreeClassifier(random_state=42)
            decision_tree.fit(X_train, y_train)

            # Predict probabilities for both training and testing sets
            probabilities_train = decision_tree.predict_proba(X_train)[:, 1]  # Probability of class 1
            probabilities_test = decision_tree.predict_proba(X_test)[:, 1]

            # Create a full array of probabilities with the same order as the original dataset
            # Initialize an array to hold the probabilities
            probabilities_full = np.zeros(X_categorical.shape[0])

            # Place the probabilities back according to the original indices
            probabilities_full[indices_train] = probabilities_train
            probabilities_full[indices_test] = probabilities_test

            # Add this array as a new column to combined_df
            combined_df['dtree_prob'] = probabilities_full
            self.logger.info(self.color_log("New feature 'dtree_prob' created from all the categorical features", Fore.GREEN))


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

            if True:

                # Remove outliers
                for column in features.select_dtypes(include=[np.number]).columns:
                    # Ignore columns starting with "geo_countries" and "geo_continent"
                    if not column.startswith("geo_countries") and not column.startswith("geo_continent"):
                        # Saving the original length of the dataframe for later comparison
                        original_len = len(features)

                        # Calculating the mean and standard deviation of the current column
                        mean_val = features[column].mean()
                        std_val = features[column].std()

                        # Defining a multiplier for standard deviation to identify outliers
                        std_multiplier = 8

                        # Outliers are defined as values that are more than 'std_multiplier' standard deviations away from the mean
                        is_outlier = (features[column] < (mean_val - std_multiplier * std_val)) | \
                                    (features[column] > (mean_val + std_multiplier * std_val))
                        outlier_index = features[is_outlier].index

                        # Removing the identified outliers from the 'features' and 'labels' dataframes
                        features.drop(outlier_index, inplace=True)
                        labels.drop(outlier_index, inplace=True)

                        # Calculating the number of rows removed
                        removed_count = original_len - len(features)
                        self.logger.info(f"Outliers removed from {column}: {self.color_log(removed_count, Fore.RED)} rows")

    
                self.logger.info(self.color_log("Outlier Removal Completed for Combined Dataset\n", Fore.GREEN))


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

            return torch.tensor(features.values).float(), torch.tensor(labels.values).float(), feature_names, class_map


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


def NDF(model: str, scaling: bool, benign: str, malign: str):
    fe_cli = FeatureEngineeringCLI(benign, malign)


    features, labels, feature_names, class_map = fe_cli.perform_eda(model, scaling)
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    benign_name = ''.join(benign.split('_')[:2])
    malign_name = ''.join(malign.split('_')[:2])
    dataset_name = f"dataset_{benign_name}_{malign_name}_{current_date}"
    dataset_name = dataset_name.replace('.parquet', '') + '.parquet'

    dataset = {
        'name': dataset_name,
        'features': features,
        'labels': labels,
        'dimension': features.shape[1],
        'feature_names': feature_names,
        'class_map': class_map
    }        

    directory = 'datasets'
    file_path = os.path.join(directory, 'preprocessed_dataset.pickle')
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)


    x_train, x_test, y_train, y_test = train_test_split(
        dataset['features'],
        dataset['labels'],
        test_size=0.2,
        random_state=42
    )

    display_dataset_subset(x_train, y_train, dataset['name'], dataset['dimension'])
    return dataset
