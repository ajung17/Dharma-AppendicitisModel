import shap
import pandas as pd

def explanation(model, x, max_display=10, class_index=1):

    explainer=shap.TreeExplainer(model)

    shap_values=explainer.shap_values(x)
    shap.summary_plot(shap_values[:,:,class_index],x,plot_type="bar",max_display=10)
    shap.summary_plot(shap_values[:,:,class_index],x,plot_type="dot",max_display=10)
    base_value=explainer.expected_value[class_index]

    importances = model.feature_importances_
    feature_importances_df=pd.DataFrame({
        'Feature': x.columns,
        'Importance': importances
    })
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

    return base_value, feature_importances_df
