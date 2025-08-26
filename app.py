import os
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq
from tempfile import NamedTemporaryFile, mkdtemp
from pydantic import BaseModel
import logging
import shutil
import zipfile
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = os.getenv("MODEL", "")  # Updated to valid Groq model

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CleaningRequest(BaseModel):
    temp_path: str
    instruction: str

class EDARequest(BaseModel):
    temp_path: str
    target_column: str = None  # Optional target variable for supervised analysis
    max_categories: int = 20  # Maximum categories to show in categorical plots

class CSVProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV with shape: {self.df.shape}")

    def apply_script(self, script):
        """Apply a pandas script to the dataframe with safety measures."""
        try:
            # Create a safe environment for execution
            safe_globals = {
                "pd": pd,
                "np": np,
                "__builtins__": {},  # Restrict built-in functions for security
            }
            safe_locals = {"df": self.df.copy()}
            
            # Execute the script
            exec(script, safe_globals, safe_locals)
            
            # Update the dataframe if it was modified
            if "df" in safe_locals:
                self.df = safe_locals["df"]
                logger.info(f"Script applied successfully. New shape: {self.df.shape}")
                return True, None
        except Exception as e:
            logger.error(f"Error applying script: {e}")
            return False, str(e)
        return True, None

    def export_csv(self, output_path):
        """Export the processed dataframe to CSV."""
        try:
            self.df.to_csv(output_path, index=False)
            logger.info(f"CSV exported to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            return False

    def get_basic_info(self):
        """Get comprehensive information about the dataset."""
        df = self.df
        
        # Basic info
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "duplicate_rows": int(df.duplicated().sum()),
        }
        
        # Column analysis
        column_info = {}
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": float(round(df[col].isnull().sum() / len(df) * 100, 2)),
                "unique_count": int(df[col].nunique()),
                "unique_percentage": float(round(df[col].nunique() / len(df) * 100, 2)),
            }
            
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                col_info.update({
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "median": float(df[col].median()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                    "skewness": float(df[col].skew()) if not df[col].isna().all() else None,
                    "kurtosis": float(df[col].kurtosis()) if not df[col].isna().all() else None,
                })
                
                # Detect potential outliers using IQR method
                if not df[col].isna().all():
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
                    col_info["outlier_count"] = int(len(outliers))
                    col_info["outlier_percentage"] = float(round(len(outliers) / len(df) * 100, 2))
            
            elif df[col].dtype == 'object':
                if not df[col].isna().all():
                    top_values = df[col].value_counts().head(5).to_dict()
                    col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                    col_info["avg_string_length"] = float(round(df[col].astype(str).str.len().mean(), 2))
            
            column_info[col] = col_info
        
        info["column_details"] = column_info
        
        # Correlation analysis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            # Find highly correlated pairs
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        high_correlations.append({
                            "var1": str(correlation_matrix.columns[i]),
                            "var2": str(correlation_matrix.columns[j]),
                            "correlation": float(round(corr_val, 3))
                        })
            info["high_correlations"] = high_correlations
        
        return info

class CustomEDAGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.plot_files = []
        
    def generate_comprehensive_eda(self, df, target_column=None, max_categories=20):
        """Generate comprehensive EDA plots."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Dataset Overview
        self._create_dataset_overview(df)
        
        # 2. Missing Values Analysis
        self._create_missing_values_analysis(df)
        
        # 3. Numeric Variables Analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            self._create_numeric_analysis(df, numeric_cols)
            
        # 4. Categorical Variables Analysis
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            self._create_categorical_analysis(df, categorical_cols, max_categories)
            
        # 5. Correlation Analysis
        if len(numeric_cols) > 1:
            self._create_correlation_analysis(df, numeric_cols)
            
        # 6. Distribution Analysis
        self._create_distribution_analysis(df, numeric_cols)
        
        # 7. Target Variable Analysis (if specified)
        if target_column and target_column in df.columns:
            self._create_target_analysis(df, target_column, numeric_cols, categorical_cols)
            
        return self.plot_files
    
    def _save_plot(self, filename, title=None):
        """Save plot and add to files list."""
        filepath = os.path.join(self.output_dir, filename)
        if title:
            plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        self.plot_files.append(filepath)
        
    def _create_dataset_overview(self, df):
        """Create dataset overview plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Data types
        dtype_counts = df.dtypes.value_counts()
        ax1.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        ax1.set_title('Data Types Distribution')
        
        # Missing values
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            missing_data.plot(kind='bar', ax=ax2, color='coral')
            ax2.set_title('Missing Values by Column')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Missing Values by Column')
        
        # Dataset shape info
        info_text = f"""Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
        
Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
Duplicated Rows: {df.duplicated().sum():,}
        
Numeric Columns: {len(df.select_dtypes(include=[np.number]).columns)}
Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}
        """
        ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace')
        ax3.axis('off')
        ax3.set_title('Dataset Summary')
        
        # Unique values per column
        unique_counts = df.nunique().sort_values(ascending=False)
        if len(unique_counts) <= 20:
            unique_counts.plot(kind='bar', ax=ax4, color='skyblue')
            ax4.set_title('Unique Values per Column')
            ax4.tick_params(axis='x', rotation=45)
        else:
            # Show top 20 columns with most unique values
            unique_counts.head(20).plot(kind='bar', ax=ax4, color='skyblue')
            ax4.set_title('Top 20 Columns by Unique Values')
            ax4.tick_params(axis='x', rotation=45)
        
        self._save_plot('01_dataset_overview.png', 'Dataset Overview')
    
    def _create_missing_values_analysis(self, df):
        """Create missing values heatmap and analysis."""
        if df.isnull().sum().sum() == 0:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Missing values heatmap
        sns.heatmap(df.isnull(), cbar=True, cmap='viridis', ax=ax1)
        ax1.set_title('Missing Values Heatmap')
        
        # Missing values percentage
        missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_percent = missing_percent[missing_percent > 0]
        
        if len(missing_percent) > 0:
            missing_percent.plot(kind='barh', ax=ax2, color='coral')
            ax2.set_title('Missing Values Percentage by Column')
            ax2.set_xlabel('Percentage Missing')
        
        self._save_plot('02_missing_values_analysis.png', 'Missing Values Analysis')
    
    def _create_numeric_analysis(self, df, numeric_cols):
        """Create analysis for numeric variables."""
        if not numeric_cols:
            return
            
        # Distributions
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                df[col].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'{col}\nMean: {df[col].mean():.2f}, Std: {df[col].std():.2f}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].axis('off')
        
        self._save_plot('03_numeric_distributions.png', 'Numeric Variables Distribution')
        
        # Box plots for outlier detection
        if len(numeric_cols) <= 10:
            fig, ax = plt.subplots(figsize=(12, 6))
            df[numeric_cols].boxplot(ax=ax)
            ax.set_title('Box Plots - Outlier Detection')
            ax.tick_params(axis='x', rotation=45)
            self._save_plot('04_numeric_boxplots.png', 'Numeric Variables Box Plots')
    
    def _create_categorical_analysis(self, df, categorical_cols, max_categories):
        """Create analysis for categorical variables."""
        if not categorical_cols:
            return
            
        for i, col in enumerate(categorical_cols[:6]):  # Limit to first 6 categorical columns
            if df[col].nunique() <= max_categories:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Value counts bar plot
                value_counts = df[col].value_counts()
                value_counts.plot(kind='bar', ax=ax1, color='lightcoral')
                ax1.set_title(f'{col} - Value Counts')
                ax1.tick_params(axis='x', rotation=45)
                
                # Pie chart
                if len(value_counts) <= 10:
                    ax2.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                    ax2.set_title(f'{col} - Distribution')
                else:
                    # Show top 10 and group others
                    top_values = value_counts.head(9)
                    others_sum = value_counts.iloc[9:].sum()
                    if others_sum > 0:
                        plot_data = pd.concat([top_values, pd.Series({'Others': others_sum})])
                    else:
                        plot_data = top_values
                    ax2.pie(plot_data.values, labels=plot_data.index, autopct='%1.1f%%')
                    ax2.set_title(f'{col} - Top Categories')
                
                self._save_plot(f'05_categorical_{i+1}_{col.replace(" ", "_")}.png')
    
    def _create_correlation_analysis(self, df, numeric_cols):
        """Create correlation analysis."""
        if len(numeric_cols) < 2:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Correlation heatmap
        correlation_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, ax=ax1, fmt='.2f')
        ax1.set_title('Correlation Matrix')
        
        # Correlation with target (if exists) or strongest correlations
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                corr_pairs.append({
                    'pair': f"{correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}",
                    'correlation': corr_val
                })
        
        # Sort by absolute correlation value
        corr_pairs = sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
        top_correlations = corr_pairs[:10]
        
        pairs = [item['pair'] for item in top_correlations]
        correlations = [item['correlation'] for item in top_correlations]
        
        colors = ['red' if abs(c) > 0.7 else 'orange' if abs(c) > 0.5 else 'green' for c in correlations]
        ax2.barh(range(len(pairs)), correlations, color=colors)
        ax2.set_yticks(range(len(pairs)))
        ax2.set_yticklabels(pairs)
        ax2.set_title('Top 10 Variable Correlations')
        ax2.set_xlabel('Correlation Coefficient')
        
        self._save_plot('06_correlation_analysis.png', 'Correlation Analysis')
    
    def _create_distribution_analysis(self, df, numeric_cols):
        """Create distribution analysis with statistical tests."""
        if not numeric_cols:
            return
            
        # Statistical distribution analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:4]):
            if i < 4:
                # Q-Q plot for normality
                stats.probplot(df[col].dropna(), dist="norm", plot=axes[i])
                axes[i].set_title(f'{col} - Q-Q Plot (Normality Test)')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), 4):
            axes[i].axis('off')
        
        self._save_plot('07_distribution_analysis.png', 'Statistical Distribution Analysis')
    
    def _create_target_analysis(self, df, target_column, numeric_cols, categorical_cols):
        """Create target variable analysis."""
        if target_column not in df.columns:
            return
            
        target_is_numeric = df[target_column].dtype in [np.number, 'int64', 'float64']
        
        if target_is_numeric:
            # Target vs numeric variables
            if len(numeric_cols) > 1:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                other_numeric = [col for col in numeric_cols if col != target_column][:4]
                for i, col in enumerate(other_numeric):
                    if i < 4:
                        axes[i].scatter(df[col], df[target_column], alpha=0.6)
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel(target_column)
                        axes[i].set_title(f'{target_column} vs {col}')
                
                self._save_plot('08_target_vs_numeric.png', f'Target Variable ({target_column}) vs Numeric Variables')
        else:
            # Target distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            target_counts = df[target_column].value_counts()
            target_counts.plot(kind='bar', ax=ax1, color='lightblue')
            ax1.set_title(f'{target_column} - Distribution')
            ax1.tick_params(axis='x', rotation=45)
            
            # Target vs first categorical variable
            if categorical_cols and len(categorical_cols) > 1:
                other_cat = [col for col in categorical_cols if col != target_column][0]
                pd.crosstab(df[other_cat], df[target_column]).plot(kind='bar', ax=ax2, stacked=True)
                ax2.set_title(f'{target_column} vs {other_cat}')
                ax2.tick_params(axis='x', rotation=45)
            
            self._save_plot('08_target_analysis.png', f'Target Variable ({target_column}) Analysis')

class GroqIntegration:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_cleaning_script(self, instruction, csv_sample, column_info):
        """Generate a data cleaning script using Groq API."""
        prompt = f"""
Act as a data cleaning expert. You are provided with a CSV file sample and need to write a Python script using pandas to clean the data.

INSTRUCTION: {instruction}

CSV SAMPLE:
{csv_sample}

COLUMN INFORMATION:
{column_info}

REQUIREMENTS:
1. Write ONLY executable Python code using pandas
2. The dataframe variable is called 'df'
3. Modify 'df' in-place or reassign it
4. Do not include any explanatory text, comments, or markdown
5. Do not use any imports (pandas is already available as 'pd', numpy as 'np')
6. Focus on the specific instruction given

Example format:
df = df.dropna()
df['column'] = df['column'].str.strip()

Your script:
"""
        
        try:
            response = groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            script = response.choices[0].message.content.strip()
            
            # Clean up the script (remove markdown formatting if present)
            if "```python" in script:
                script = script.split("```python")[1].split("```")[0].strip()
            elif "```" in script:
                script = script.split("```")[1].strip()
                
            logger.info(f"Generated script: {script}")
            return script
        except Exception as e:
            logger.error(f"Error getting cleaning script: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate cleaning script: {str(e)}")

    def analyze_eda_results(self, df_info, plot_descriptions):
        """Generate comprehensive EDA analysis and insights using Groq API."""
        prompt = f"""
You are a senior data scientist with expertise in exploratory data analysis. Analyze the following dataset information and visualization descriptions to provide comprehensive insights and actionable recommendations.

DATASET INFORMATION:
{df_info}

GENERATED VISUALIZATIONS:
{plot_descriptions}

Please provide a detailed analysis report with the following sections:

## 1. EXECUTIVE SUMMARY
- Brief overview of the dataset and key findings
- Most critical insights that require immediate attention

## 2. DATA QUALITY ASSESSMENT
- Missing data patterns and their potential impact
- Outliers and anomalies detected
- Data consistency issues
- Recommendations for data quality improvements

## 3. KEY STATISTICAL INSIGHTS
- Distribution patterns of key variables
- Significant correlations and relationships
- Statistical anomalies or interesting patterns
- Variability and central tendency insights

## 4. BUSINESS INSIGHTS (Based on Data Patterns)
- What the data reveals about business operations
- Customer/user behavior patterns (if applicable)
- Performance indicators and trends
- Risk factors or opportunities identified

## 5. ACTIONABLE RECOMMENDATIONS
### Data Processing:
- Priority data cleaning steps
- Feature engineering opportunities
- Data collection improvements

### Analysis & Modeling:
- Suggested analytical approaches
- Variables of interest for deeper analysis
- Potential modeling strategies

### Business Actions:
- Immediate actions based on findings
- Long-term strategic considerations
- Areas requiring further investigation

## 6. TECHNICAL NOTES
- Assumptions made during analysis
- Limitations of current analysis
- Suggestions for advanced analysis

Format your response clearly with headers and bullet points for easy reading.
"""
        
        try:
            response = groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000
            )
            
            analysis = response.choices[0].message.content.strip()
            logger.info("Generated comprehensive EDA analysis")
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating EDA analysis: {e}")
            return f"Error generating analysis: {str(e)}"

    def explain_specific_chart(self, chart_name, df_info, chart_context=""):
        """Explain a specific chart in detail."""
        prompt = f"""
You are a data visualization expert. Explain the following chart in detail based on the dataset context.

CHART: {chart_name}
CHART CONTEXT: {chart_context}

DATASET CONTEXT:
{df_info}

Please provide a detailed explanation including:

1. **What This Chart Shows**: Clear explanation of what the visualization represents
2. **Key Patterns & Trends**: Specific patterns, trends, or relationships visible
3. **Statistical Insights**: What the data distribution/relationship tells us statistically
4. **Business Implications**: What these patterns mean for business decisions
5. **Notable Observations**: Any anomalies, outliers, or interesting findings
6. **Actionable Insights**: Specific recommendations based on this visualization
7. **Next Steps**: Suggested follow-up analysis or actions

Keep the explanation practical and actionable, focusing on insights that can drive decisions.
"""
        
        try:
            response = groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining chart: {e}")
            return f"Error explaining chart: {str(e)}"

# Initialize FastAPI app
app = FastAPI(title="Advanced CSV Analysis Service", version="2.0.0")

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file for processing."""
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Check file size (50MB limit for EDA)
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    
    try:
        # Save uploaded file to temporary location
        with NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Validate that it's actually a CSV by trying to read it
            try:
                df_test = pd.read_csv(temp_file.name)
                logger.info(f"Uploaded CSV validated. Shape: {df_test.shape}")
            except Exception as e:
                os.unlink(temp_file.name)  # Clean up
                raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
            
            return {
                "message": "File uploaded successfully",
                "filename": file.filename,
                "temp_path": temp_file.name,
                "rows": len(df_test),
                "columns": len(df_test.columns),
                "column_names": list(df_test.columns)
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/clean-csv")
async def clean_csv(request: CleaningRequest):
    """Clean CSV based on provided instructions."""
    temp_path = request.temp_path
    instruction = request.instruction
    
    if not os.path.exists(temp_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    
    try:
        # Load and sample CSV data for context
        df = pd.read_csv(temp_path)
        csv_sample = df.head(5).to_string()
        column_info = f"Columns: {list(df.columns)}\nData types: {df.dtypes.to_dict()}"
        
        # Get cleaning script from Groq API
        groq = GroqIntegration(MODEL)
        script = groq.get_cleaning_script(instruction, csv_sample, column_info)
        
        # Apply the script to the CSV
        processor = CSVProcessor(temp_path)
        success, error = processor.apply_script(script)
        
        if success:
            # Generate output path
            output_path = temp_path.replace(".csv", "_cleaned.csv")
            if processor.export_csv(output_path):
                return {
                    "status": "success",
                    "message": "CSV cleaned successfully",
                    "output_path": output_path,
                    "script_used": script,
                    "original_rows": len(df),
                    "cleaned_rows": len(processor.df)
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to export cleaned CSV")
        else:
            raise HTTPException(status_code=500, detail=f"Script execution failed: {error}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleaning process failed: {str(e)}")

@app.post("/generate-eda")
async def generate_eda(request: EDARequest):
    """Generate comprehensive EDA using custom plotting and provide AI analysis."""
    temp_path = request.temp_path
    target_column = request.target_column
    max_categories = request.max_categories
    
    if not os.path.exists(temp_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    try:
        # Load the CSV
        processor = CSVProcessor(temp_path)
        df = processor.df
        
        # Get basic dataset information
        df_info = processor.get_basic_info()
        
        # Create temporary directory for plots
        plot_dir = mkdtemp(prefix="eda_plots_")
        
        # Generate EDA plots using custom generator
        eda_generator = CustomEDAGenerator(plot_dir)
        generated_plots = eda_generator.generate_comprehensive_eda(
            df, target_column=target_column, max_categories=max_categories
        )
        
        # Create plot descriptions for AI analysis
        plot_descriptions = []
        for plot_path in generated_plots:
            plot_name = os.path.basename(plot_path)
            plot_descriptions.append(f"Generated plot: {plot_name}")
        
        # Generate AI analysis
        groq = GroqIntegration(MODEL)
        comprehensive_analysis = groq.analyze_eda_results(
            df_info, "\n".join(plot_descriptions)
        )
        
        # Create zip file with all plots
        zip_path = temp_path.replace(".csv", "_eda_plots.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for plot_path in generated_plots:
                arcname = os.path.basename(plot_path)
                zipf.write(plot_path, arcname)
        
        return {
            "status": "success",
            "message": "EDA generated successfully",
            "dataset_info": df_info,
            "plots_generated": len(generated_plots),
            "plots_zip_path": zip_path,
            "comprehensive_analysis": comprehensive_analysis,
            "plot_directory": plot_dir,
            "individual_plots": [os.path.basename(p) for p in generated_plots]
        }
        
    except Exception as e:
        logger.error(f"EDA generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"EDA generation failed: {str(e)}")

@app.post("/explain-chart")
async def explain_chart(chart_name: str, temp_path: str):
    """Get detailed explanation of a specific chart."""
    if not os.path.exists(temp_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    try:
        # Load dataset info
        processor = CSVProcessor(temp_path)
        df_info = processor.get_basic_info()
        
        # Generate explanation
        groq = GroqIntegration(MODEL)
        explanation = groq.explain_specific_chart(chart_name, df_info)
        
        return {
            "status": "success",
            "chart_name": chart_name,
            "explanation": explanation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart explanation failed: {str(e)}")

@app.get("/download-csv")
async def download_csv(output_path: str):
    """Download the cleaned CSV file."""
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        return FileResponse(
            output_path, 
            filename="cleaned.csv", 
            media_type="application/csv",
            headers={"Content-Disposition": "attachment; filename=cleaned.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/download-eda-plots")
async def download_eda_plots(zip_path: str):
    """Download all EDA plots as a zip file."""
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="EDA plots zip not found")
    
    try:
        return FileResponse(
            zip_path,
            filename="eda_plots.zip",
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=eda_plots.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.delete("/cleanup")
async def cleanup_temp_files(temp_path: str):
    """Clean up temporary files and directories."""
    try:
        files_to_remove = [
            temp_path, 
            temp_path.replace(".csv", "_cleaned.csv"),
            temp_path.replace(".csv", "_eda_plots.zip")
        ]
        removed_count = 0
        
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    removed_count += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    removed_count += 1
                    
        return {"message": f"Cleaned up {removed_count} temporary files/directories"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Advanced CSV Analysis API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)