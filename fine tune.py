pip install gradientai
import os
os.environ['GRADIENT_ACCESS_TOKEN'] = "bW5OzpPIxIsqu5cmBSyK3OzJeivQ8pI5"##i have revoked these
os.environ['GRADIENT_WORKSPACE_ID'] = "664820f8-5741-41e7-8b1e-8b8f7ea84bd7_workspace"

from gradientai import Gradient

def main():
  with Gradient() as gradient:
      base_model = gradient.get_base_model(base_model_slug="nous-hermes2")

      new_model_adapter = base_model.create_model_adapter(
          name="test model 3"
      )
      print(f"Created model adapter with id {new_model_adapter.id}")
      sample_query = "### Instruction: what is a dividend in stock market? \n\n### Response:"
      print(f"Asking: {sample_query}")

      # before fine-tuning
      completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
      print(f"Generated (before fine-tune): {completion}")

      samples = [
  {
    "inputs": "### Instruction: What is a stock market index? \n\n### Response: A stock market index is a measure of the value of a section of the stock market. It is calculated from the prices of selected stocks. Common examples include the S&P 500 and the Dow Jones Industrial Average.",
  },
  {
    "inputs": "### Instruction: Can you explain compound interest? \n\n### Response: Compound interest is the addition of interest to the principal sum of a loan or deposit, or in other words, interest on interest. It allows for exponential growth of wealth over time.",
  },
  {
    "inputs": "### Instruction: How does diversification work in investing? \n\n### Response: Diversification in investing involves spreading your investments across different assets to reduce risk. By investing in various asset classes, such as stocks, bonds, and real estate, you can minimize the impact of any single investment's poor performance on your overall portfolio.",
  },
  {
    "inputs": "### Instruction: What is the difference between stocks and bonds? \n\n### Response: Stocks represent ownership in a company, entitling the holder to a share of the company's profits. Bonds, on the other hand, are debt securities where investors lend money to the issuer, who promises to repay the loan with interest over a specified period.",
  },
  {
    "inputs": "### Instruction: Explain the concept of risk-return tradeoff in investing. \n\n### Response: The risk-return tradeoff states that investments with higher potential returns typically come with greater risk. Investors must weigh the potential for higher returns against the likelihood of losses when making investment decisions.",
  },
  {
    "inputs": "### Instruction: What are mutual funds? \n\n### Response: Mutual funds are investment vehicles that pool money from multiple investors to purchase a diversified portfolio of stocks, bonds, or other securities. They are managed by professional portfolio managers and offer investors access to a diversified investment strategy.",
  },
  {
    "inputs": "### Instruction: How does the Federal Reserve influence the economy? \n\n### Response: The Federal Reserve, also known as the Fed, influences the economy through monetary policy. It can adjust interest rates, control the money supply, and implement other measures to stabilize prices, promote full employment, and regulate economic growth.",
  },
  {
    "inputs": "### Instruction: Explain the concept of 'buy low, sell high' in investing. \n\n### Response: 'Buy low, sell high' is a fundamental principle in investing that advocates purchasing assets when their prices are low and selling them when prices rise. It aims to capitalize on market fluctuations and generate profits from price appreciation.",
  },
  {
    "inputs": "### Instruction: What is a 401(k) retirement plan? \n\n### Response: A 401(k) retirement plan is a tax-advantaged retirement account offered by employers to their employees. Employees can contribute a portion of their pre-tax income to the plan, which is then invested in a variety of investment options, typically including stocks, bonds, and mutual funds, with the goal of building a retirement nest egg.",
  },
  {
    "inputs": "### Instruction: Can you explain the concept of asset allocation? \n\n### Response: Asset allocation involves dividing an investment portfolio among different asset classes, such as stocks, bonds, and cash, based on an investor's risk tolerance, investment goals, and time horizon. It aims to optimize returns while managing risk by diversifying across various types of assets.",
  },
  {
    "inputs": "### Instruction: What are the key financial ratios used in fundamental analysis? \n\n### Response: Some key financial ratios used in fundamental analysis include the price-to-earnings ratio (P/E ratio), debt-to-equity ratio, return on equity (ROE), and earnings per share (EPS). These ratios help investors evaluate a company's financial health, profitability, and valuation.",
  },
  {
    "inputs": "### Instruction: Explain the concept of dollar-cost averaging. \n\n### Response: Dollar-cost averaging is an investment strategy where an investor regularly invests a fixed amount of money into a particular asset, regardless of its price fluctuations. This approach allows investors to buy more shares when prices are low and fewer shares when prices are high, potentially reducing the impact of market volatility over time.",
  },
  {
    "inputs": "### Instruction: What are the different types of orders in stock trading? \n\n### Response: Some common types of orders in stock trading include market orders, limit orders, stop orders, and trailing stop orders. Each type of order has its own specific instructions for executing trades and managing risk in the stock market.",
  },
  {
    "inputs": "### Instruction: How do exchange-traded funds (ETFs) work? \n\n### Response: Exchange-traded funds (ETFs) are investment funds that trade on stock exchanges like individual stocks. They typically track a specific index, commodity, or basket of assets and offer investors exposure to diversified portfolios with lower expense ratios compared to traditional mutual funds.",
  }
]


      
      num_epochs = 3
      count = 0
      while count < num_epochs:
          print(f"Fine-tuning the model, iteration {count + 1}")
          new_model_adapter.fine_tune(samples=samples)
          count = count + 1

      # after fine-tuning
      completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
      print(f"Generated (after fine-tune): {completion}")



if __name__ == "__main__":
    main()

def is_finance_related(question):
    finance_keywords = ["stock", "investment", "finance", "retirement", "portfolio"]
    for keyword in finance_keywords:
        if keyword in question.lower():
            return True
    return False

def get_model_response(question):
    if is_finance_related(question):
        # Pass the finance-related question to the model for a response
        model_response = completion.generate_response(question)
        return model_response
    else:
        return "Please ask a finance-related question."

# Example usage:
user_question = input("Ask a question: ")
response = get_model_response(user_question)
print(response)    