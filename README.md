# multi-agent-financial-risk
Multi-agent LLM pipeline for financial risk assessment using AutoGen. Compares single-agent vs safeguard-validated multi-agent reliability on portfolio volatility and Value-at-Risk calculations.

# Multi-Agent LLM for Financial Risk Assessment

**March 2025 – Present | University of the Pacific | Independent Research**

I built this to explore whether adding a validation layer to an LLM pipeline actually improves reliability on quantitative tasks. Financial risk calculations are a good test case because the math is well-defined — there's a correct answer — so you can measure exactly how much the agent screws up, and whether a second agent reviewing the output catches those errors.

The system uses AutoGen to orchestrate three agents: a Commander that manages the workflow, a Financial Analyst that pulls live market data and computes risk metrics, and a Risk Safeguard that reviews the output before it gets accepted. I then ran both the multi-agent pipeline and a single-agent baseline across multiple trials to compare accuracy and error rates against ground-truth calculations.

---

## Architecture

```
Commander (UserProxy)
    │
    ├──► Financial Analyst
    │       Pulls live price data via yfinance
    │       Calculates annualized volatility and Value-at-Risk
    │
    └──► Risk Safeguard
            Checks for implausible values, missing fields,
            and internal consistency (e.g. 10d VaR ≈ 3.16× 1d VaR)
            Approves or rejects with explanation
```

## Research question

Does a Safeguard validation agent reduce output errors in automated financial risk calculations compared to a single-agent model?

## Metrics computed

- Annualized portfolio volatility
- 1-day 95% Historical Value-at-Risk
- 10-day 95% Value-at-Risk

All agent outputs are compared against ground-truth values computed directly in NumPy.

## Setup

```bash
git clone https://github.com/BathHarman/multi-agent-financial-risk.git
cd multi-agent-financial-risk
pip install -r requirements.txt

cp .env.example .env
# Add your OpenAI API key to .env
```

Run the full evaluation:

```bash
python financial_risk_agent.py
```

Custom portfolio:

```python
from financial_risk_agent import run_evaluation

my_portfolio = {"NVDA": 0.40, "AMZN": 0.35, "BRK-B": 0.25}
df, truth = run_evaluation(n_trials=3, portfolio=my_portfolio)
```

## Output files

| File | Description |
|------|-------------|
| `results/evaluation_results.csv` | Per-trial results for both agent types |
| `results/agent_comparison.png` | Mean % error vs ground truth: single vs multi-agent |
| `results/parse_error_rate.png` | Output parse failure rate by agent type |

## Stack

Python · AutoGen · yfinance · NumPy · Pandas · Matplotlib

## Reference

Wu Q, et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv:2308.08155.
