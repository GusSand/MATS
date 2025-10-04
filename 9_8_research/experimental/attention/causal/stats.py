import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.api import Logit

def statistical_analysis(results):
    """
    Perform rigorous statistical tests on causal relationship
    """
    df = pd.DataFrame(results)
    
    # 1. Correlation analysis
    correlation = stats.spearmanr(df['begin_attention'], df['error_rate'])
    print(f"Spearman correlation: r={correlation.correlation:.3f}, p={correlation.pvalue:.4f}")
    
    # 2. Logistic regression with confidence intervals
    from statsmodels.api import Logit
    import statsmodels.api as sm
    
    X = sm.add_constant(df['begin_attention'])
    y = df['error_rate']
    
    model = Logit(y, X)
    result = model.fit()
    
    print("\nLogistic Regression Results:")
    print(result.summary())
    
    # 3. Threshold analysis
    thresholds = np.linspace(0.1, 0.5, 20)
    best_threshold = None
    best_accuracy = 0
    
    for t in thresholds:
        predicted = (df['begin_attention'] < t).astype(int)
        actual = df['error_rate']
        accuracy = (predicted == actual).mean()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t
    
    print(f"\nOptimal BEGIN attention threshold: {best_threshold:.1%}")
    print(f"Classification accuracy at threshold: {best_accuracy:.1%}")
    
    return {
        'correlation': correlation,
        'logistic_model': result,
        'threshold': best_threshold,
        'threshold_accuracy': best_accuracy
    }