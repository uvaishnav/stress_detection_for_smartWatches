import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureSelector:
    """
    A class for selecting the most relevant features from a feature matrix.
    
    This class provides methods for feature selection using various techniques:
    - Correlation analysis
    - Statistical tests (ANOVA F-value, mutual information)
    - Model-based selection (Random Forest importance)
    - Dimensionality reduction (PCA)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the feature selector.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.feature_importances_ = None
        self.selected_features_ = None
        self.pca_components_ = None
        self.pca_explained_variance_ = None
    
    def correlation_selection(self, X: pd.DataFrame, threshold: float = 0.8) -> List[str]:
        """
        Select features by removing highly correlated features.
        
        Args:
            X: DataFrame containing the features
            threshold: Correlation threshold above which to remove features
            
        Returns:
            List of selected feature names
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Create a mask for the upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Select features to keep
        selected_features = [col for col in X.columns if col not in to_drop]
        
        return selected_features
    
    def plot_correlation_heatmap(self, X: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot a correlation heatmap of the features.
        
        Args:
            X: DataFrame containing the features
            figsize: Figure size as (width, height)
        """
        plt.figure(figsize=figsize)
        corr = X.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=.5, annot=False, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    def statistical_selection(self, X: pd.DataFrame, y: pd.Series, 
                             method: str = 'f_classif', k: int = 20) -> List[str]:
        """
        Select features using statistical tests.
        
        Args:
            X: DataFrame containing the features
            y: Series containing the target variable
            method: Statistical test to use ('f_classif' or 'mutual_info')
            k: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        # Choose the scoring function
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError("Method must be one of 'f_classif' or 'mutual_info'")
        
        # Apply feature selection
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        selector.fit(X, y)
        
        # Get feature scores
        scores = selector.scores_
        
        # Create a DataFrame with feature names and scores
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': scores
        })
        
        # Sort by score in descending order
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        
        # Select top k features
        selected_features = feature_scores.head(k)['Feature'].tolist()
        
        return selected_features
    
    def plot_feature_scores(self, X: pd.DataFrame, y: pd.Series, 
                           method: str = 'f_classif', top_n: int = 20) -> None:
        """
        Plot feature importance scores from statistical tests.
        
        Args:
            X: DataFrame containing the features
            y: Series containing the target variable
            method: Statistical test to use ('f_classif' or 'mutual_info')
            top_n: Number of top features to display
        """
        # Choose the scoring function
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError("Method must be one of 'f_classif' or 'mutual_info'")
        
        # Apply feature selection
        selector = SelectKBest(score_func=score_func, k='all')
        selector.fit(X, y)
        
        # Get feature scores
        scores = selector.scores_
        
        # Create a DataFrame with feature names and scores
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': scores
        })
        
        # Sort by score in descending order
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Score', y='Feature', data=feature_scores.head(top_n))
        plt.title(f'Top {top_n} Features by {method}')
        plt.tight_layout()
        plt.show()
    
    def model_based_selection(self, X: pd.DataFrame, y: pd.Series, 
                             n_features: int = 20, method: str = 'importance') -> List[str]:
        """
        Select features using a model-based approach.
        
        Args:
            X: DataFrame containing the features
            y: Series containing the target variable
            n_features: Number of features to select
            method: Method to use ('importance' or 'rfe')
            
        Returns:
            List of selected feature names
        """
        # Initialize model
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        if method == 'importance':
            # Fit the model
            model.fit(X, y)
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Create a DataFrame with feature names and importances
            feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            })
            
            # Sort by importance in descending order
            feature_importances = feature_importances.sort_values('Importance', ascending=False)
            
            # Store feature importances
            self.feature_importances_ = feature_importances
            
            # Select top n_features
            selected_features = feature_importances.head(n_features)['Feature'].tolist()
            
        elif method == 'rfe':
            # Initialize RFE
            rfe = RFE(estimator=model, n_features_to_select=n_features)
            
            # Fit RFE
            rfe.fit(X, y)
            
            # Get selected features
            selected_features = [X.columns[i] for i in range(len(X.columns)) if rfe.support_[i]]
            
        else:
            raise ValueError("Method must be one of 'importance' or 'rfe'")
        
        # Store selected features
        self.selected_features_ = selected_features
        
        return selected_features
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Plot feature importances from model-based selection.
        
        Args:
            top_n: Number of top features to display
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available. Run model_based_selection first.")
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=self.feature_importances_.head(top_n))
        plt.title(f'Top {top_n} Features by Importance')
        plt.tight_layout()
        plt.show()
    
    def pca_selection(self, X: pd.DataFrame, n_components: Optional[int] = None, 
                     variance_threshold: float = 0.95) -> pd.DataFrame:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: DataFrame containing the features
            n_components: Number of components to keep (if None, use variance_threshold)
            variance_threshold: Minimum cumulative explained variance
            
        Returns:
            DataFrame containing the PCA-transformed features
        """
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine number of components
        if n_components is None:
            # Start with all components
            pca_all = PCA(random_state=self.random_state)
            pca_all.fit(X_scaled)
            
            # Calculate cumulative explained variance
            cumulative_variance = np.cumsum(pca_all.explained_variance_ratio_)
            
            # Find number of components that explain at least variance_threshold of variance
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        # Apply PCA with selected number of components
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # Store PCA components and explained variance
        self.pca_components_ = pca.components_
        self.pca_explained_variance_ = pca.explained_variance_ratio_
        
        # Create DataFrame with PCA components
        pca_df = pd.DataFrame(
            X_pca, 
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=X.index
        )
        
        return pca_df
    
    def plot_pca_variance(self) -> None:
        """
        Plot explained variance ratio of PCA components.
        """
        if self.pca_explained_variance_ is None:
            raise ValueError("PCA not performed. Run pca_selection first.")
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(self.pca_explained_variance_) + 1), self.pca_explained_variance_)
        plt.plot(range(1, len(self.pca_explained_variance_) + 1), 
                np.cumsum(self.pca_explained_variance_), 'r-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Principal Components')
        plt.xticks(range(1, len(self.pca_explained_variance_) + 1))
        plt.tight_layout()
        plt.show()
    
    def plot_pca_components(self, X: pd.DataFrame, n_features: int = 10) -> None:
        """
        Plot feature contributions to top PCA components.
        
        Args:
            X: Original DataFrame containing the features
            n_features: Number of top contributing features to display per component
        """
        if self.pca_components_ is None:
            raise ValueError("PCA not performed. Run pca_selection first.")
        
        # Get feature names
        feature_names = X.columns
        
        # Number of components to plot
        n_components = min(4, len(self.pca_components_))
        
        # Create figure
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 4 * n_components))
        
        # If only one component, axes is not an array
        if n_components == 1:
            axes = [axes]
        
        # Plot each component
        for i, ax in enumerate(axes):
            if i < n_components:
                # Get absolute loadings
                loadings = pd.Series(np.abs(self.pca_components_[i]), index=feature_names)
                
                # Sort and get top features
                top_features = loadings.sort_values(ascending=False).head(n_features)
                
                # Plot
                sns.barplot(x=top_features.values, y=top_features.index, ax=ax)
                ax.set_title(f'Top {n_features} Features in PC{i+1} (Explained Variance: {self.pca_explained_variance_[i]:.2%})')
                ax.set_xlabel('Absolute Loading')
                ax.set_ylabel('Feature')
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_recommendations(self, X: pd.DataFrame, y: pd.Series, 
                                   n_features: int = 20) -> Dict[str, List[str]]:
        """
        Get feature recommendations using multiple selection methods.
        
        Args:
            X: DataFrame containing the features
            y: Series containing the target variable
            n_features: Number of features to recommend
            
        Returns:
            Dictionary mapping selection method to list of recommended features
        """
        recommendations = {}
        
        # Correlation-based selection
        recommendations['correlation'] = self.correlation_selection(X, threshold=0.8)
        
        # Statistical selection
        recommendations['f_classif'] = self.statistical_selection(X, y, method='f_classif', k=n_features)
        recommendations['mutual_info'] = self.statistical_selection(X, y, method='mutual_info', k=n_features)
        
        # Model-based selection
        recommendations['random_forest'] = self.model_based_selection(X, y, n_features=n_features)
        
        # Find common features across methods
        all_features = set()
        for features in recommendations.values():
            all_features.update(features)
        
        # Count occurrences of each feature
        feature_counts = {}
        for feature in all_features:
            count = sum(1 for features in recommendations.values() if feature in features)
            feature_counts[feature] = count
        
        # Sort features by occurrence count
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Get consensus features (appearing in at least 2 methods)
        consensus_features = [feature for feature, count in sorted_features if count >= 2]
        recommendations['consensus'] = consensus_features[:n_features]
        
        return recommendations 