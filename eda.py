import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("spark/output")

print("=" * 50)
print("🔍 ANÁLISE EXPLORATÓRIA DOS DADOS")
print("=" * 50)

# 1. Estatísticas básicas
print("\n📊 Estatísticas do 'amount' por classe:")
print(df.groupby('fraud')['amount'].describe())

# 2. Comparar distribuições
print("\n📈 Média de 'amount' por classe:")
print(df.groupby('fraud')['amount'].mean())

print("\n📈 Mediana de 'amount' por classe:")
print(df.groupby('fraud')['amount'].median())

# 3. Visualização
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histograma
df[df['fraud']==0]['amount'].hist(bins=30, alpha=0.5, label='Normal', ax=axes[0], color='blue')
df[df['fraud']==1]['amount'].hist(bins=30, alpha=0.5, label='Fraud', ax=axes[0], color='red')
axes[0].set_xlabel('Amount')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribuição de Amount por Classe')
axes[0].legend()

# Boxplot
df.boxplot(column='amount', by='fraud', ax=axes[1])
axes[1].set_xlabel('Fraud (0=Normal, 1=Fraud)')
axes[1].set_ylabel('Amount')
axes[1].set_title('Boxplot de Amount por Classe')

plt.tight_layout()
plt.savefig('data_analysis.png')
print("\n✅ Gráfico salvo: data_analysis.png")

# 4. Teste estatístico
from scipy.stats import ttest_ind

normal_amounts = df[df['fraud']==0]['amount']
fraud_amounts = df[df['fraud']==1]['amount']

t_stat, p_value = ttest_ind(normal_amounts, fraud_amounts)
print(f"\n📊 Teste T (diferença entre médias):")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Há diferença significativa entre fraude e normal!")
else:
    print("❌ NÃO há diferença significativa - dados podem ser aleatórios!")

# 5. Correlação com target
print(f"\n🔗 Correlação de 'amount' com 'fraud': {df['amount'].corr(df['fraud']):.4f}")

# 6. Ver alguns exemplos
print("\n👀 Exemplos de transações NORMAIS:")
print(df[df['fraud']==0].head(10))

print("\n👀 Exemplos de transações FRAUDULENTAS:")
print(df[df['fraud']==1].head(10))