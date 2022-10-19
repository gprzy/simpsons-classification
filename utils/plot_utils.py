import matplotlib.pyplot as plt
import seaborn as sns

# função para exibir f1 scores em conjunto de uma coluna desejada
def plot_scores_by_col(col, df, figsize=(14,10)):
    fig, ax = plt.subplots(1,2, figsize=figsize);

    sns.barplot(
        data=df,
        x='f1-score',
        y=col,
        palette='viridis',
        ax=ax[0]
    );

    df_agg = df.groupby(col).agg('max')
    df_agg = df_agg.sort_values(by='f1-score', ascending=False)

    sns.barplot(
        data=df_agg,
        x='f1-score',
        y=df_agg.index,
        palette='viridis',
        ax=ax[1]
    );

    plt.tight_layout();
    ax[0].set_title(f'f1-scores médios por {col}');
    ax[1].set_title(f'f1-scores máximos por {col}');