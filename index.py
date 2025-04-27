import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

def label_encode_features(df):
    le = LabelEncoder()
    cat_cols = df.select_dtypes(exclude=["number"]).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def drop_correlated_features(df, limit=0.8):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > limit)]

    print("Признаки, которые будут удалены из-за высокой корреляции:")
    for col in drop_cols:
        print("-", col)
    
    return df.drop(columns=drop_cols)

def main():
    data = pd.read_csv('AmesHousing.csv')

    numeric = data.select_dtypes(include=["number"])
    categorical = data.select_dtypes(exclude=["number"])

    categorical_encoded = label_encode_features(categorical)

    full_data = pd.concat([numeric, categorical_encoded], axis=1)

    full_data = drop_correlated_features(full_data)

    imputer = SimpleImputer(strategy='mean')
    X = full_data.drop(columns=["SalePrice"])
    X = imputer.fit_transform(X)
    y = full_data["SalePrice"]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], y, c=y, cmap='plasma')
    ax.set_xlabel('Первая компонента PCA')
    ax.set_ylabel('Вторая компонента PCA')
    ax.set_zlabel('SalePrice')
    plt.title('3D визуализация данных (PCA + SalePrice)')
    fig.colorbar(scatter, shrink=0.5, aspect=10)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    alpha_values = np.logspace(-3, 2, 50)
    rmse_list = []

    for alpha in alpha_values:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        rmse_list.append(rmse)

    best_alpha = alpha_values[np.argmin(rmse_list)]
    print(f"Лучшее значение alpha: {best_alpha:.5f}")

    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, rmse_list, marker='o')
    plt.xscale('log')
    plt.xlabel('Alpha (коэффициент регуляризации)')
    plt.ylabel('RMSE')
    plt.title('Влияние регуляризации на ошибку (Lasso)')
    plt.axvline(best_alpha, color='red', linestyle='--', label=f'Оптимальное alpha = {best_alpha:.5f}')
    plt.legend()
    plt.grid(True)
    plt.show()

    final_lasso = Lasso(alpha=best_alpha, max_iter=10000)
    final_lasso.fit(X_train, y_train)

    features = full_data.drop(columns=["SalePrice"]).columns
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': final_lasso.coef_,
        'Abs_Coefficient': np.abs(final_lasso.coef_)
    })

    top_feature = coef_df.sort_values(by='Abs_Coefficient', ascending=False).iloc[0]
    print(f"Самый влиятельный признак: {top_feature.Feature} (вес: {top_feature.Coefficient:.2f})")

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Abs_Coefficient', y='Feature', data=coef_df.sort_values(by='Abs_Coefficient', ascending=False).head(10), palette='mako')
    plt.title('Топ-10 наиболее влияющих признаков по Lasso')
    plt.show()

if __name__ == "__main__":
    main()
