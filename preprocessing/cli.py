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
import shap
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeClassifier


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
    def __init__(self, benign_path: str, malign_path: str):
        self.benign_path = benign_path
        self.malign_path = malign_path
        self.logger = self.configure_logger()
        self.DEFAULT_INPUT_DIR = "../floor"
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

    def perform_RFE(self, X_train, y_train, shap_threshold=0.03, n_features_to_select=None):

        # Filter out only numeric columns
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns
        X_train_numeric = X_train[numeric_columns]

        # Train the model on numeric data
        model = XGBClassifier()
        model.fit(X_train_numeric, y_train)

        # Calculate SHAP values
        explainer = shap.Explainer(model, X_train_numeric)
        shap_values = explainer(X_train_numeric)

        # Sum absolute SHAP values for each feature
        shap_sum = np.abs(shap_values.values).mean(axis=0)

        # Filter features based on SHAP value significance
        significant_features = [feature for feature, shap_value in zip(X_train_numeric.columns, shap_sum) if shap_value > shap_threshold]


        # If n_features_to_select is not set, select a smaller number of significant features
        if n_features_to_select is None:
            n_features_to_select = max(5, len(significant_features) // 4)  # Example: Select top 25% of significant features, with a minimum of 5

        # Apply RFE on significant features
        rfe = RFE(model, n_features_to_select=n_features_to_select)
        rfe.fit(X_train_numeric[significant_features], y_train)

        # Get features to remove
        features_to_remove = np.array(significant_features)[~rfe.support_]

        # Log features that could be removed
        removable_features_log = ', '.join([self.color_log(feature, Fore.RED) for feature in features_to_remove])
        self.logger.info(f"Removable features calculated by RFE: {removable_features_log}")


        return features_to_remove



    def configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler('feature_engineering.log')
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

    def load_pickled_data(self) -> tuple:
        shap_values = pickle.load(open("../models/shap_values.pickle.dat", "rb"))
        model = pickle.load(open("../models/xgboost_model.pickle.dat", "rb"))
        X_train = pickle.load(open("../models/X_train.pickle.dat", "rb"))
        return shap_values, model, X_train
    
    def load_training_data(self):
        """
        Load training features and labels.
        
        Returns:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        """
        # Example: Loading data from pickled files. Adjust this as per your data storage.
        X_train = pickle.load(open("../models/X_train.pickle.dat", "rb"))
        y_train = pickle.load(open("../models/y_train.pickle.dat", "rb"))
        return X_train, y_train

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
    

    def apply_scaling(self, df: pd.DataFrame, scaler_type: str) -> pd.DataFrame:
        """
        Apply scaling to the dataset based on the scaler type.
        Returns a new DataFrame with scaled data.
        """
        numeric_df = df.select_dtypes(include=[np.number])

        if scaler_type == 'StandardScaler':
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
        elif scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(numeric_df)
        elif scaler_type == 'RobustScaler':
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(numeric_df)
        elif scaler_type == 'MinMaxScaler + Sigmoid':
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            # Apply sigmoid scaling
            scaled_data = 1 / (1 + np.exp(-scaled_data))
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

        # Update the DataFrame with scaled data
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
        """
        Perform one-hot encoding for geographical features.
        Returns a new DataFrame with encoded features.
        """
        
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
        """
        Perform binary encoding for lexical features.
        Returns a new DataFrame with encoded features.
        """
        
        if 'lex_tld_hash' in df.columns:
            binary_encoder = BinaryEncoder(cols=['lex_tld_hash'])
            df = binary_encoder.fit_transform(df)
            self.logger.info(self.color_log("Applied binary encoding to feature: lex_tld_hash", Fore.GREEN))

        return df

    def categorical_encoding_tls_rdap(self, df: DataFrame) -> DataFrame:
        """
        Handling TLS/RDAP features: registrar_name_hash, root_authority_hash, leaf_authority_hash
        """
        
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


    def perform_feature_engineering(self) -> None:
        """
        Perform feature engineering on the dataset. This includes:
        - Recursive Feature Elimination (RFE)
        - Feature Selection
        - Creating Plots
        - Saving Results
        """

        benign_path = os.path.join(self.DEFAULT_INPUT_DIR, self.benign_path) if self.benign_path else None
        malign_path = os.path.join(self.DEFAULT_INPUT_DIR, self.malign_path) if self.malign_path else None
        #load shap values, model and X_train
        shap_values, model, X_train = self.load_pickled_data()
        X_train, y_train = self.load_training_data()

        # Performing Recursive Feature Elimination (RFE)
        self.print_header("Performing Recursive Feature Elimination (RFE)")
        removable_features = self.perform_RFE(X_train, y_train)
        for feature in removable_features:
            formatted_feature = self.color_log(feature, Fore.YELLOW)
            self.logger.info(f"{formatted_feature}")

        
        self.print_header("Performing Feature Engineering")
        self.logger.info(f'Benign dataset path: {benign_path}')
        self.logger.info(f'Malign dataset path: {malign_path}')       
        data = pq.read_table(benign_path)
        data2 = pq.read_table(malign_path)

        benign_domain_names = data['domain_name']
        malign_domain_names = data2['domain_name']

        # Drop non-training columns
        data = self.drop_nontrain(data)
        data2 = self.drop_nontrain(data2)
        data = data.drop(columns=['domain_name'])
        data2 = data2.drop(columns=['domain_name'])

        df1 = data.to_pandas()
        df2 = data2.to_pandas()

        # Calculate ratio of benign to phishing for features containing 'tls'
        features_with_validity_len = [col for col in df1.columns if 'validity' in col]
        features_with_lifetime = [col for col in df1.columns if 'lifetime' in col]
        features_with_count = [col for col in df1.columns if 'count' in col]
        features_with_age = [col for col in df1.columns if 'age' in col]
        features_with_active_time = [col for col in df1.columns if 'active_time' in col]
        features_with_period = [col for col in df1.columns if 'period' in col]

        selected_features = features_with_validity_len + features_with_lifetime + features_with_age + features_with_active_time + features_with_period 
        selected_features_without_age = [
            col for col in selected_features if col not in ['rdap_domain_age', 'rdap_domain_active_time', 'rdap_registration_period']
        ]

        ratios = {}
        for feature in selected_features_without_age:
            benign_count = df1[feature].count()
            phishing_count = df2[feature].count()
            
            if phishing_count != 0:
                ratio = benign_count / phishing_count
                ratios[feature] = ratio

        # Convert timedelta columns to numeric representation
        for col in ['rdap_domain_age', 'rdap_domain_active_time', 'rdap_registration_period']:
            df1[col] = df1[col].dt.days
            df2[col] = df2[col].dt.days

        # Calculate counts after converting timedelta columns to days
        for col in ['rdap_domain_age', 'rdap_domain_active_time', 'rdap_registration_period']:
            benign_count = df1[col].count()
            phishing_count = df2[col].count()
            
            if phishing_count != 0:
                ratio = benign_count / phishing_count
                ratios[col] = ratio
        for key in ratios:
            self.logger.info(self.color_log(f'Possibly time demanding feature: {key}', Fore.YELLOW))

        # Sort features by ratio in descending order
        sorted_features = sorted(ratios, key=ratios.get, reverse=True)[:9]  # Select top 9 features
        custom_colors = ['#1f77b4', '#ff7f0e']  # Blue for Benign, Orange for Phishing
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

        for i, feature in enumerate(sorted_features):
            ax = axes[i // 3, i % 3]  # Calculate subplot index
            
            lts_benign = df1[feature].fillna(-1.0)
            lts_phishing = df2[feature].fillna(-1.0)
            
            # Plot histograms using the custom color palette
            ax.hist(lts_benign, bins=30, alpha=0.7, label='Benign', density=True, color=custom_colors[0])
            ax.hist(lts_phishing, bins=30, alpha=0.7, label='Phishing', density=True, color=custom_colors[1])
            ax.set_xlabel('Value', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'{feature}', fontsize=14)
            ax.legend(prop={'size': 10})

            # Customize tick labels and font sizes
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=8)

            # Adding gridlines and removing spines
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)


        fig.suptitle('Relative Frequency Histograms for Top Features', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
        plt.savefig('../false_positives/images/relative_freq_histograms.png', dpi=300) 
        plt.close()

        self.logger.info(self.color_log('\n', Fore.GREEN))
        self.logger.info(self.color_log('Choosing most relevant features based on feature_importances_', Fore.YELLOW))
        feature_importances = model.feature_importances_
        normalized_importances = (feature_importances - np.min(feature_importances)) / (np.max(feature_importances) - np.min(feature_importances))
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_20_indices = sorted_indices[:20]

        # Plotting the top 20 feature importances
        plt.figure(figsize=(10, 8))
        sns.barplot(x=normalized_importances[top_20_indices], y=X_train.columns[top_20_indices], orient='h')
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout(pad=1.0)
        plt.savefig('../false_positives/images/top_20_feature_importances.png', dpi=300)  # Save the plot as a high-resolution PNG

        self.print_header("Saving Results")
        
        self.logger.info(self.color_log('Saved possible time-dependent features to:'
                         ' ../false_positives/images/relative_freq_histograms.png', Fore.GREEN))
        self.logger.info(self.color_log('Saved top 20 feature importances plot to:'
                         ' ../false_positives/images/top_20_feature_importances.png', Fore.GREEN))

    def sort_and_color_vif(self, vif_data):
        # Sort VIF data by VIF values in descending order
        vif_data = vif_data.sort_values(by='VIF', ascending=False)

        # VIF >= 5: High multicollinearity. The variable is highly correlated with other independent variables and may cause issues in the model.
        vif_data['VIF'] = vif_data['VIF'].apply(lambda x: f"{Fore.RED}{x:.2f}{Style.RESET_ALL}" if x > 5 else f"{x:.2f}")

        return vif_data
        
    def remove_features(self, df: pd.DataFrame, features_to_remove: list) -> pd.DataFrame:
        return df.drop(columns=features_to_remove, errors='ignore')
    
    

    def explore_data(self, df: pd.DataFrame, dataset_name: str) -> list:
        """
        Perform Exploratory Data Analysis (EDA) on the dataset.
        
        Returns:
            list: List of potentially useless features.
        """
        self.print_header(f"Exploratory Data Analysis (EDA) - {dataset_name}")

        # Display basic information about the dataset
        self.logger.info(self.color_log("Basic Info of the Dataset:", Fore.YELLOW))
        self.logger.info(df.info())

        # Summary statistics of the dataset
        self.logger.info(self.color_log("Summary Statistics of the Dataset:", Fore.YELLOW))
        self.logger.info(df.describe())

        # Check for missing values
        self.logger.info(self.color_log("Missing Values:", Fore.YELLOW))
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]  # Filter columns with non-zero missing values
        # Sort them in descending order
        missing_values = missing_values.sort_values(ascending=False)
        if not missing_values.empty:
            missing_values_table = tabulate(missing_values.reset_index(), headers=[""], tablefmt="plain")
            # Log the tabulated missing values with count in red
            lines = missing_values_table.split('\n')
            self.logger.info(lines[0])  # Log the header line
            for line in lines[1:]:
                if line.strip():  # Skip empty lines
                    column, count = line.rsplit(maxsplit=1)  # Split at the last space
                    column = column.strip()  # Remove extra spaces
                    count = count.strip()  # Remove extra spaces
                    self.logger.info(f"{column.ljust(50)}{self.color_log(count, Fore.RED)}")
        else:
            self.logger.info("No missing values found.")

        self.logger.info(self.color_log('\n', Fore.GREEN))
        constant_features = df.columns[(df.nunique() == 1) & (df.columns != 'label')]
        self.logger.info(self.color_log("Constant Features (same value for whole dataset):", Fore.YELLOW))
    
        for feature in constant_features:
            self.logger.info(self.color_log(feature, Fore.GREEN))

        self.logger.info(self.color_log('\n', Fore.GREEN))
        # Filter numeric columns for outlier detection
        numerical_columns = df.select_dtypes(include=[np.number]).columns


        # Convert inf values to NaN before operating
        df[numerical_columns] = df[numerical_columns].replace([np.inf, -np.inf], np.nan)


        # Convert timedelta columns to numeric (e.g., days or seconds)
        for col in df.select_dtypes(include=['timedelta64']).columns:
            df[col] = df[col].dt.total_seconds()

        # Ensure all numeric columns are of the same data type (float)
        numeric_df = df.select_dtypes(include=[np.number]).astype(float)

        # Apply Variance Thresholding to numeric data
        self.logger.info(self.color_log("Applying Variance Thresholding to Numeric Data:", Fore.YELLOW))
        threshold = 0.001  # Threshold for variance
        sel = VarianceThreshold(threshold=threshold)
        sel.fit(numeric_df)
        mask = sel.get_support()

        # Identify low variance features in numeric data
        low_variance_features = numeric_df.columns[~mask]
        self.logger.info(self.color_log(f"Suggested Low Variance Numeric Features for Review (Threshold = {threshold}):", Fore.YELLOW))
        for feature in low_variance_features:
            self.logger.info(self.color_log(feature, Fore.GREEN))
        self.logger.info(self.color_log('\n', Fore.GREEN))


        # Detect and handle outliers for numerical columns
        self.logger.info(self.color_log("Detecting Outliers [%]:", Fore.YELLOW))
        outliers_summary = {}
        for column in numerical_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))
            outlier_percentage = outliers.sum() / len(df[column]) * 100  # Calculate outlier percentage
            outliers_summary[column] = outlier_percentage

        # Print outlier detection summary
        for column, percentage in outliers_summary.items():
            self.logger.info(f"Outliers in {self.color_log(f'{column}: {percentage:.2f}%', Fore.RED)}")
        self.logger.info(self.color_log('\n', Fore.RED))

        #get scaler recommendation
        print(Fore.YELLOW + "Get Scaler recommendation based on Data Statistics: ", end='\n')
        
        scaler_recommendations = self.scaler_recommendation(df)
        for method, scaler in scaler_recommendations.items():
            message = f"{method}: {scaler}"
            self.logger.info(self.color_log(message, Fore.GREEN))
        self.logger.info(self.color_log('\n', Fore.GREEN))



        potentially_useless_features = list(constant_features) + list(low_variance_features)
        return potentially_useless_features
    

    def cast_timestamp(df: DataFrame):
        for col in df.columns:
            if com.is_timedelta64_dtype(df[col]):
                df[col] = df[col].dt.total_seconds()  # This converts timedelta to float (seconds)
            elif com.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(np.int64) // 10**9  # Converts datetime64 to Unix timestamp (seconds)

        return df

    def perform_eda(self, model=None, apply_scaling=False) -> None:
        """
        Perform Exploratory Data Analysis (EDA) on the dataset.
        """
        benign_path = os.path.join(self.DEFAULT_INPUT_DIR, self.benign_path) if self.benign_path else None
        malign_path = os.path.join(self.DEFAULT_INPUT_DIR, self.malign_path) if self.malign_path else None
        
        self.logger.info(self.color_log(f'Benign dataset path: {benign_path}', Fore.GREEN))
        self.logger.info(self.color_log(f'Malign dataset path: {malign_path}', Fore.GREEN))

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

        # Train a decision tree to get class probabilities
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_categorical, labels)

        # Extract probability of the positive class
        probabilities = decision_tree.predict_proba(X_categorical)[:, 1]  # Get probability of class 1

        # Add this probability as a new feature
        combined_df['dtree_prob'] = probabilities
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

        labels = combined_df['label'].apply(lambda x: class_map.get(x, -1))  # -1 for any label not in class_map
        features = combined_df.drop('label', axis=1).copy()
        for col in features.columns:
            if com.is_timedelta64_dtype(features[col]):
                features[col] = features[col].dt.total_seconds()
            elif com.is_datetime64_any_dtype(features[col]):
                features[col] = features[col].astype(np.int64) // 10**9

       
        for column in features.columns:
            if features[column].dtype == 'bool':
                features[column] = features[column].astype('float64')

        features = features.drop(features.columns[0], axis=1)

        # Handling missing values in features
        features.fillna(-1, inplace=True)
        potentially_useless = self.explore_data(combined_df, "Combined Dataset")
        scaler_recommendations = self.scaler_recommendation(features)

        response = input(self.color_log("Do you want to implement the mentioned suggestions (handling missing values, removing outliers, categorical features encoding)? (yes/no): ", Fore.YELLOW)).strip().lower()
        if response == 'yes':
            # Remove potentially useless features
            features = self.remove_features(features, potentially_useless)

            #drop those same labels
            labels = self.remove_features(labels, potentially_useless)

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
                    features.drop(outlier_index, inplace=True)
                    labels.drop(outlier_index, inplace=True)

                    # Calculating the number of rows removed
                    removed_count = original_len - len(features)
                    self.logger.info(f"Outliers removed from {column}: {self.color_log(removed_count, Fore.RED)} rows")

            #IRQ not suitable, too strict, removes too many records
                
            # for column in features.select_dtypes(include=[np.number]).columns:
            #     original_len = len(features)
                
            #     # Calculate Q1, Q3, and IQR
            #     Q1 = features[column].quantile(0.25)
            #     Q3 = features[column].quantile(0.75)
            #     IQR = Q3 - Q1

            #     # Define the outlier boundaries
            #     lower_bound = Q1 - 1.5 * IQR
            #     upper_bound = Q3 + 1.5 * IQR

            #     # Find outlier indices
            #     outlier_index = features[(features[column] < lower_bound) | (features[column] > upper_bound)].index
                
            #     # Remove outliers
            #     features.drop(outlier_index, inplace=True)
            #     labels.drop(outlier_index, inplace=True)  # Also remove corresponding labels

            #     removed_count = original_len - len(features)
            #     self.logger.info(f"Outliers removed from {column}: {removed_count} rows")

            # self.logger.info(self.color_log("Outlier Removal Completed for Combined Dataset\n", Fore.GREEN))


            # Apply scaling if requested
            if apply_scaling:
                # self.logger.info(self.color_log(f"Applying {scaler_recommendations} scaling to the features.", Fore.YELLOW))
                scaler_type = scaler_recommendations.get(model.lower(), 'StandardScaler')
                features = self.apply_scaling(features, scaler_type)
                    
            self.logger.info(self.color_log("Outlier Removal Completed for Combined Dataset\n", Fore.GREEN))
            self.logger.info(self.color_log("Scaling applied to the features\n", Fore.GREEN))

            # Save the modified dataset as a Parquet file
            modified_data = pa.Table.from_pandas(features)
            output_path = os.path.join(self.DEFAULT_INPUT_DIR, 'modified_dataset.parquet')
            pq.write_table(modified_data, output_path)
            self.logger.info(self.color_log(f"Modified combined dataset saved to {output_path}", Fore.GREEN))

            self.logger.info(self.color_log("Head of modified combined dataset:", Fore.YELLOW))
            self.logger.info(features)

            return torch.tensor(features.values).float(), torch.tensor(labels.values).float()
        else:
            self.logger.info(self.color_log("No changes made to the dataset.", Fore.YELLOW))
            return None, None


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

@click.command()
@click.option('-eda', is_flag=True, help='Perform Exploratory Data Analysis')
@click.option('--model', type=click.Choice(['svm', 'xgboost', 'adaboost', 'cnn'], case_sensitive=False), default=None, help='Classification model to use')
@click.option('--scaling', is_flag=True, help='Scale data based on model')
def feature_engineering(eda: bool, model: str, scaling: bool):
    floor_folder = "../floor"
    parquet_files = os.listdir(floor_folder)
    dataset_files = [file for file in parquet_files if file.endswith('.parquet')]

    chosen_benign = choose_dataset(dataset_files, "benign")
    chosen_malign = choose_dataset(dataset_files, "malign")

    fe_cli = FeatureEngineeringCLI(benign_path=chosen_benign, malign_path=chosen_malign)

    if eda:
        features, labels = fe_cli.perform_eda(model, scaling)
        if model is None and scaling is not None:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            benign_name = ''.join(chosen_benign.split('_')[:2])
            malign_name = ''.join(chosen_malign.split('_')[:2])
            dataset_name = f"dataset_{benign_name}_{malign_name}_{current_date}"
            dataset_name = dataset_name.replace('.parquet', '') + '.parquet'

            dataset = {
                'name': dataset_name,
                'features': features,
                'labels': labels,
                'dimension': features.shape[1]
            }        

            directory = 'datasets'
            file_path = os.path.join(directory, 'malware_dataset.pickle')
            
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

    else:
        fe_cli.perform_feature_engineering()

if __name__ == '__main__':
    feature_engineering()